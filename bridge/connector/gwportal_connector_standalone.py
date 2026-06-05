
#%%
from typing import List
import time
import sys
import requests
from astropy.table import Table
import os
from tqdm import tqdm
from requests.exceptions import RequestException, JSONDecodeError
from urllib.parse import urljoin # Use urljoin for robust URL construction
#%%
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
#%%


class GWPortalConnector:
    
    def __init__(self,
                 query_type: str = 'processed',
                 base_url: str = None,
                 api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.query_type = query_type
        self.client = GWPortalClient(base_url = self.base_url, api_key = self.api_key)

    def __repr__(self):
        return f"GWPortalConnector(query_type = {self.query_type})"
    
    @property
    def query_func(self):
        if self.query_type.lower() == 'raw':
            return self.client.query_raw
        elif self.query_type.lower() == 'processed':
            return self.client.query_processed
        elif self.query_type.lower() == 'combined':
            return self.client.query_combined
        elif self.query_type.lower() == 'tile':
            return self.client.query_tiles
        elif self.query_type.lower() == 'target':
            return self.client.query_targets
        else:
            raise ValueError(f"Invalid query_type: {self.query_type}")
    
    
    # TODO: Implement parameters when updated
    def query(self,
              since_days: float = None,
              obs_start_date: str = None,
              obs_end_date: str = None,
              filter_name: str = None,
              unit_name: str = None,
              object_name: str = None,
              obsnote_contains: str = None,
              target_type: str = None,
              tile_name: str = None,
              target_name_contains: str = None,
              coord_sys: str = None,
              ra: float = None,
              dec: float = None,
              gl: float = None,
              gb: float = None,
              radius: float = None,
              polygon: List[List[float]] = None,
              ):
        """
        Query the GWPortal API and return the results as an Astropy Table.
        Parameters:
            since_days: (float) = Number of days to query data from the current date.
            obs_start_date: (str) = Start date of the observation.
            obs_end_date: (str) = End date of the observation.
            filter_name: (str) = Name of the filter.
            unit_name: (str) = Name of the unit.
            object_name: (str) = Name of the object.
            obsnote_contains: (str) = String to filter the observation notes.
            target_type: (str) = Type of the target.
            tile_name: (str) = Name of the tile.
            target_name_contains: (str) = String to filter the target name.
            coord_sys: (str) = Coordinate system.
            ra: (float) = Right ascension.
            dec: (float) = Declination.
            gl: (float) = Galactic longitude.
            gb: (float) = Galactic latitude.
            radius: (float) = Radius of the target.
            polygon: (List[List[float]]) = Polygon of the target.
        Returns:
            tbl: Astropy Table
        """
        script_start_time = time.time()

        query_params = {
            'days': since_days,
            'date_start': obs_start_date,
            'date_end': obs_end_date,
            'filter_name': filter_name,
            'unit_name': unit_name,
            'object_name': object_name,
            'obsnote_contains': obsnote_contains,
            'target_type': target_type,
            'tile_name': tile_name,
            'target_name_contains': target_name_contains,
            'coord_sys': coord_sys,
            'ra': ra,
            'dec': dec,
            'gl': gl,
            'gb': gb,
            'radius': radius,
            'polygon': polygon
        }
        query_params = {k: v for k, v in query_params.items() if v is not None}

        # --- Execute Based on Output Mode ---
        try:
            fetch_start_time = time.time()

            # --- MODE 1: Client-Side Streaming (for --output-csv) ---
            print(f"\n--- Mode: Client-Side Streaming to CSV ---")
            all_items_fetched = self.client.get_all_results(
                self.query_func,
                desc=f"Fetching {self.query_type}",
                **query_params
            )
            
            fetch_duration = time.time() - fetch_start_time
            print(f"\nData fetching took {fetch_duration:.2f} seconds. {len(all_items_fetched)} items fetched.")

            tbl = Table(all_items_fetched)
            return tbl
            
        # --- Error Handling ---
        except ValueError as e: # Specific client init errors
            print(f"\nClient Configuration Error: {e}", file=sys.stderr)
            sys.exit(1)
        except requests.exceptions.RequestException as e: # Network or API errors
            print(f"\nNetwork/API Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e: # Other unexpected errors
            print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
        finally:
            script_end_time = time.time()
            total_duration = script_end_time - script_start_time
            print("-" * 30)
            print(f"Script finished in {total_duration:.2f} seconds.")

#%% Example Usage
if __name__ == "__main__":
    base_url = None
    api_key = None
    connector_raw = GWPortalConnector(query_type = 'raw', base_url = base_url, api_key = api_key)
    tbl_raw = connector_raw.query(
        obs_start_date = '2025-10-01', 
        obs_end_date = '2025-10-05')
    print(tbl_raw)
    connector_processed = GWPortalConnector(query_type = 'processed', base_url = base_url, api_key = api_key)
    tbl_processed = connector_processed.query(
        filter_name = 'm525')
    print(tbl_processed)
    tbl_processed_with_gwevent = connector_processed.query(
        obsnote_contains = 'S251112cm')

        
        
# %%
