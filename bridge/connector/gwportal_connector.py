
#%%
from typing import List
import numpy as np
from bridge.connector import GWPortalClient
from bridge.configuration import Configuration
import time
import sys
import requests
from astropy.table import Table
#%%
class GWPortalConnector:
    def __init__(self,
                 query_type: str = 'processed'):
        self.config = Configuration(config_filenames=['gwportalconnector.config'])
        self.base_url = None
        self.api_key = None
        self.query_type = query_type
        
        if self.config.base_url is not None:
            self.base_url = self.config.base_url
        if self.config.api_key is not None:
            self.api_key = self.config.api_key
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
              verbose: bool = True
              ):
        """
        
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
            if verbose:
                print(f"\n--- Mode: Client-Side Streaming to CSV ---")
            all_items_fetched = self.client.get_all_results(
                self.query_func,
                desc=f"Fetching {self.query_type}",
                **query_params
            )
            
            fetch_duration = time.time() - fetch_start_time
            if verbose:
                print(f"\nData fetching took {fetch_duration:.2f} seconds. {len(all_items_fetched)} items fetched.")

            tbl = Table(all_items_fetched)
            return tbl
            
        # --- Error Handling ---
        except ValueError as e: # Specific client init errors
            if verbose:
                print(f"\nClient Configuration Error: {e}", file=sys.stderr)
            sys.exit(1)
        except requests.exceptions.RequestException as e: # Network or API errors
            if verbose:
                print(f"\nNetwork/API Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e: # Other unexpected errors
            if verbose:
                print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
        finally:
            script_end_time = time.time()
            total_duration = script_end_time - script_start_time
            if verbose:
                print("-" * 30)
            if verbose:
                print(f"Script finished in {total_duration:.2f} seconds.")


#%%
if __name__ == "__main__":
    self = GWPortalConnector()
    self.query_type = 'processed'
#%%
if __name__ == "__main__":
    since_days = 150
    obs_start_date = None
    obs_end_date = None
    filter_name = None
    unit_name = None
    object_name = None
    obsnote_contains = None
    target_type = None
    tile_name = None
    target_name_contains = None
    coord_sys = None
    ra = None
    dec = None
    gl = None
    gb = None
    radius = None
    polygon = None
    tbl_processed = self.query(since_days = since_days, obs_start_date = obs_start_date, obs_end_date = obs_end_date, filter_name = filter_name, unit_name = unit_name, object_name = object_name, obsnote_contains = obsnote_contains, target_type = target_type, tile_name = tile_name, target_name_contains = target_name_contains, coord_sys = coord_sys, ra = ra, dec = dec, gl = gl, gb = gb, radius = radius, polygon = polygon)