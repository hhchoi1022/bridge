# toolkit/gwportal_client.py

import os
import requests
import time
from tqdm import tqdm
from requests.exceptions import RequestException, JSONDecodeError
import sys # Import sys for stderr
from urllib.parse import urljoin # Use urljoin for robust URL construction

class GWPortalClient:
    """
    Client for interacting with the function-based GWPortal API.
    Handles authentication, pagination, and error handling.
    
    This client uses client-side streaming (stream_all_results) for
    bulk data handling.
    """

    def __init__(self, base_url=None, api_key=None):
        """
        Initialize the client.
        Reads from environment variables if parameters are not provided.
        """
        self.base_url = base_url or os.getenv('GWPORTAL_BASE_URL')
        self.api_key = api_key or os.getenv('GWPORTAL_API_KEY')

        if not self.base_url:
            raise ValueError("GWPORTAL_BASE_URL is not set.")
        if not self.api_key:
            raise ValueError("GWPORTAL_API_KEY is not set.")

        # Ensure base_url ends with a slash
        if not self.base_url.endswith('/'):
            self.base_url += '/'
        
        self.headers = {
            # Use X-API-Key as expected by the api_key_required decorator
            'X-API-Key': self.api_key, 
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        try:
            # Test connection by querying the /api/tiles/ endpoint
            # We must prepend 'api/' here for the test URL
            test_url = urljoin(self.base_url, 'api/tiles/')
            response = self.session.get(test_url, params={'page_size': 1}, timeout=10)
            response.raise_for_status()
            print("GWPortalClient: API connection successful.", file=sys.stderr)
        except RequestException as e:
            print(f"GWPortalClient: API connection test failed. Error: {e}", file=sys.stderr)
            # Do not raise error, allow script to continue
            
    def _make_request(self, method, endpoint, params=None, data=None, stream=False):
        """
        Helper function to make requests.
        This function is responsible for prepending the 'api/' prefix.
        """
        # Prepend 'api/' to the endpoint, as defined in bohrspec/urls.py
        api_endpoint = f"api/{endpoint.lstrip('/')}"
        url = urljoin(self.base_url, api_endpoint)
        
        try:
            response = self.session.request(method, url, params=params, json=data, stream=stream, timeout=60) # 60s timeout
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            if stream:
                return response # Return the response object for streaming

            if not response.content:
                 return {} # Return empty dict for 204 No Content or empty responses
            
            return response.json()
        
        except JSONDecodeError:
             raise RequestException(f"API returned non-JSON response. Status: {response.status_code}. Text: {response.text[:200]}... URL: {url}")
        
        except RequestException as e:
            if e.response is not None:
                try:
                    # Parse error details, expecting {'error': '...'}
                    error_data = e.response.json()
                    detail = error_data.get('error', 'Unknown API Error') 
                    raise RequestException(f"API Error ({e.response.status_code}): {detail} (URL: {url})")
                except JSONDecodeError:
                     raise RequestException(f"API Error ({e.response.status_code}): {e.response.text[:200]}... (URL: {url})")
            else:
                 raise RequestException(f"Network Error: {e} (URL: {url})")

    def _fetch_paginated_data(self, query_func, desc="Fetching", **kwargs):
        """
        Fetches all pages of data from a paginated endpoint (e.g., in-memory fetch).
        This is the memory-intensive function.
        """
        all_results = []
        page = 1
        page_size = kwargs.get('page_size', 5000)
        
        try:
            kwargs['page'] = page
            kwargs['page_size'] = page_size
            response = query_func(**kwargs)
        except RequestException as e:
            print(f"Error on first page fetch: {e}", file=sys.stderr)
            return []

        total_count = response.get('count', 0)
        if total_count == 0:
            return []

        results = response.get('results', [])
        all_results.extend(results)

        total_pages = response.get('num_pages', 1)
        
        if total_pages > 1:
            with tqdm(total=total_count, desc=desc, unit=" items", initial=len(all_results), file=sys.stdout) as pbar:
                for page in range(2, total_pages + 1):
                    try:
                        kwargs['page'] = page
                        response = query_func(**kwargs)
                        results = response.get('results', [])
                        
                        if not results:
                            print(f"\nWarning: Page {page} returned no results. Stopping pagination.", file=sys.stderr)
                            break
                            
                        all_results.extend(results)
                        pbar.update(len(results))
                    
                    except RequestException as e:
                        print(f"\nError fetching page {page}: {e}", file=sys.stderr)
                        break
                    except KeyboardInterrupt:
                         print(f"\nPagination interrupted by user at page {page}.", file=sys.stderr)
                         break

        return all_results

    def stream_all_results(self, query_func, desc="Streaming", **kwargs):
        """
        Streams all pages of data from a paginated endpoint using a generator.
        This is the memory-efficient method.
        
        Yields:
            int: The total count of items (yielded first).
            dict: The individual result items.
        """
        page = 1
        page_size = kwargs.get('page_size', 500)
        
        total_count = 0
        total_pages = 1
        try:
            kwargs['page'] = page
            kwargs['page_size'] = page_size
            response = query_func(**kwargs)
            total_count = response.get('count', 0)
            total_pages = response.get('num_pages', 1)
        except RequestException as e:
            print(f"Error on first page fetch: {e}", file=sys.stderr)
            yield 0
            return

        yield total_count

        if total_count == 0:
            return

        results = response.get('results', [])
        yield from results
        
        if total_pages > 1:
            for page in range(2, total_pages + 1):
                try:
                    kwargs['page'] = page
                    response = query_func(**kwargs)
                    results = response.get('results', [])
                    
                    if not results:
                        print(f"\nWarning: Page {page} returned no results. Stopping pagination.", file=sys.stderr)
                        break
                        
                    yield from results
                
                except RequestException as e:
                    print(f"\nError fetching page {page}: {e}", file=sys.stderr)
                    break
                except KeyboardInterrupt:
                     print(f"\nStreaming interrupted by user at page {page}.", file=sys.stderr)
                     break

    # --- Query Functions ---
    # These functions map to the endpoints in api/urls.py
    # They pass the path *after* the 'api/' prefix.
    
    def query_raw(self, **kwargs):
        """Query the 'raw' frames endpoint."""
        # Path: frames/raw/
        return self._make_request('get', 'frames/raw/', params=kwargs)

    def query_processed(self, **kwargs):
        """Query the 'processed' frames endpoint."""
        # Path: frames/processed/
        return self._make_request('get', 'frames/processed/', params=kwargs)

    def query_combined(self, **kwargs):
        """Query the 'combined' frames endpoint."""
        # Path: frames/combined/
        return self._make_request('get', 'frames/combined/', params=kwargs)

    def query_tiles(self, **kwargs):
        """Query the 'tiles' endpoint."""
        # Path: tiles/
        return self._make_request('get', 'tiles/', params=kwargs)
        
    def query_targets(self, **kwargs):
        """Query the 'targets' endpoint."""
        # Path: targets/
        return self._make_request('get', 'targets/', params=kwargs)

    # --- Original In-Memory Fetch Function (Kept for small console queries) ---
    def get_all_results(self, query_func, desc="Fetching", **kwargs):
        """
        Fetches all pages of data and returns them as a single list.
        (This is the original, memory-intensive function)
        """
        return self._fetch_paginated_data(query_func, desc, **kwargs)
