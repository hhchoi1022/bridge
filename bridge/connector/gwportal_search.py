# toolkit/gwportal_search.py

import os
import sys
import json # Ensure json is imported for serializing polygon data
import argparse
from datetime import datetime, timedelta, timezone # Import timezone
import requests # Added for handling potential client errors
from tqdm import tqdm # Import tqdm
import time # Import time for timing execution
import csv # Import csv module

# --- Import the updated client ---
try:
    from gwportal_client import GWPortalClient
except ImportError:
    print("Error: Could not import GWPortalClient. Make sure gwportal_client.py is accessible.", file=sys.stderr)
    sys.exit(1)

# --- Helper Function ---
def format_and_print_results(fetched_results_sample, total_fetched_count, args):
    """
    Formats and prints a sample of the fetched items to the console
    in the specified format, including filepath.

    Args:
        fetched_results_sample (list): A list of the first N items.
        total_fetched_count (int): The total number of items found.
        args (argparse.Namespace): The script arguments.
    """
    total_fetched = total_fetched_count
    query_type = args.type.lower()
    print_limit = args.print_limit

    print(f"\n--- Query Report ({query_type.capitalize()}) ---")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Add applied filters to the report
    print("\nApplied Filters:")
    # (Date range printing)
    if args.days:
        print(f"  - Time Range: Last {args.days} day(s)")
    else:
        date_range_parts = []
        if args.date_start:
            date_range_parts.append(f"from {args.date_start}")
        if args.date_end:
            date_range_parts.append(f"until {args.date_end}")
        if date_range_parts:
             date_range = " ".join(date_range_parts)
        # --- MODIFIED: Removed 1-day default reporting ---
        # elif not args.days and query_type in ['raw', 'processed', 'combined']:
        #     date_range = "Default (last 1 day)"
        else:
             date_range = "All time"
        print(f"  - Time Range: {date_range}")

    # (Attribute filters)
    if args.filter_name: print(f"  - Filter: {args.filter_name}")
    if args.unit_name: print(f"  - Unit: {args.unit_name}")
    if args.night_date: print(f"  - Night Date: {args.night_date}")
    if args.target_type: print(f"  - Target Type: {args.target_type}")

    # (Spatial/Name filters)
    if args.tile_name: print(f"  - Tile Name: {args.tile_name}")
    if args.target_name: print(f"  - Target Name: {args.target_name}")
    if args.object_name: print(f"  - Object Name (contains): {args.object_name}")
    if args.obsnote_contains: print(f"  - ObsNote (contains): {args.obsnote_contains}")

    # (Spatial coordinate filters)
    if args.coord_sys and (args.ra is not None or args.gl is not None or args.polygon is not None):
         print(f"  - Coordinate System: {args.coord_sys}")
    if args.ra is not None and args.dec is not None:
         print(f"  - RA/Dec (radec): {args.ra}, {args.dec}")
    if args.gl is not None and args.gb is not None:
         print(f"  - GL/GB (galactic): {args.gl}, {args.gb}")
    if args.radius:
         print(f"  - Cone Search Radius: {args.radius} deg")
    if args.polygon:
         print(f"  - Polygon Search: {args.polygon[:50]}...")


    print(f"\nTotal Items Found: {total_fetched}")

    if total_fetched > 0 and fetched_results_sample:
        results_to_print = fetched_results_sample
        print(f"\n--- Sample Results (displaying up to {len(results_to_print)}) ---")

        # Define header based on frame type
        # NOTE: Polygon data is NOT added to the console printout
        # as it would make the fixed-width format unreadable.
        # It is only added to the CSV output.
        if query_type == 'combined':
            header = "Filename | ObsStart (UTC) | Unit | Filter | Seeing | Ellip | Depth(ul5) | MJD | Filepath"
        elif query_type == 'processed':
            header = "Filename | ObsTime (UTC) | Unit | Filter | Seeing | Ellip | Depth(ul5) | MJD | UnifiedFile"
        elif query_type == 'raw':
            header = "Filename | ObsTime (UTC) | Unit | Filter | Seeing | Airmass | Object | MJD | UnifiedFile"
        elif query_type == 'tile':
             header = "Name | RA | Dec | L | B | Priority | Obs Count"
        elif query_type == 'target':
             header = "Name | RA | Dec | L | B | Type | Obs Count"

        print(header)
        print("-" * (len(header) + 5))

        iterable = tqdm(results_to_print, unit=" item", desc="Formatting Sample", disable=len(results_to_print)<100)

        for item_data in iterable:
            line = "N/A"
            try:
                # Format based on the query type
                if query_type == 'combined':
                    time_utc_str = item_data.get('obs_start', 'N/A').split('.')[0].replace('T', ' ')
                    line = (
                        f"{item_data.get('filename', 'N/A'):<30.30} | {time_utc_str} | "
                        f"{item_data.get('unit', 'N/A'):<5} | {item_data.get('filter', 'N/A'):<6} | "
                        f"{item_data.get('seeing', 0.0):<6.2f} | {item_data.get('ellip', 0.0):<5.3f} | "
                        f"{item_data.get('ul5', 0.0):<5.2f} | {item_data.get('mjd', 0.0):<10.4f} | "
                        f"{item_data.get('filepath', 'N/A')}"
                    )
                elif query_type == 'processed':
                    time_utc_str = item_data.get('obstime', 'N/A').split('.')[0].replace('T', ' ')
                    line = (
                        f"{item_data.get('filename', 'N/A'):<30.30} | {time_utc_str} | "
                        f"{item_data.get('unit', 'N/A'):<5} | {item_data.get('filter', 'N/A'):<6} | "
                        f"{item_data.get('seeing', 0.0):<6.2f} | {item_data.get('ellip', 0.0):<5.3f} | "
                        f"{item_data.get('ul5', 0.0):<5.2f} | {item_data.get('mjd', 0.0):<10.4f} | "
                        f"{item_data.get('unified_filename', 'N/A')}"
                    )
                elif query_type == 'raw':
                    time_utc_str = item_data.get('obstime', 'N/A').split('.')[0].replace('T', ' ')
                    line = (
                        f"{item_data.get('filename', 'N/A'):<30.30} | {time_utc_str} | "
                        f"{item_data.get('unit', 'N/A'):<5} | {item_data.get('filter', 'N/A'):<6} | "
                        f"{item_data.get('seeing', 0.0):<6.2f} | {item_data.get('airmass', 0.0):<5.3f} | "
                        f"{item_data.get('object_name', 'N/A'):<15.15} | {item_data.get('mjd', 0.0):<10.4f} | "
                        f"{item_data.get('unified_filename', 'N/A')}"
                    )
                elif query_type == 'tile':
                    line = (
                        f"{item_data.get('name', 'N/A'):<12.12} | {item_data.get('ra', 0.0):<8.4f} | "
                        f"{item_data.get('dec', 0.0):<8.4f} | {item_data.get('l', 0.0):<8.4f} | "
                        f"{item_data.get('b', 0.0):<8.4f} | {item_data.get('priority', 0):<8} | "
                        f"{item_data.get('observation_count', 0)}"
                    )
                elif query_type == 'target':
                    line = (
                        f"{item_data.get('name', 'N/A'):<20.20} | {item_data.get('ra', 0.0):<8.4f} | "
                        f"{item_data.get('dec', 0.0):<8.4f} | {item_data.get('l', 0.0):<8.4f} | "
                        f"{item_data.get('b', 0.0):<8.4f} | {item_data.get('target_type', 'N/A'):<8} | "
                        f"{item_data.get('observation_count', 0)}"
                    )
            except Exception as e:
                line = f"Error formatting item: {e}"

            print(line)

        if print_limit is not None and total_fetched > print_limit:
            print(f"\n... displayed first {print_limit} of {total_fetched} found items.")

    elif total_fetched > 0:
         print(f"\n... {total_fetched} items were processed (e.g., saved to CSV), but console display was skipped (limit=0).")
    else:
        print("\nNo items found matching the criteria.")

# --- CSV Row Formatting Helper ---
def _format_item_for_csv(item_data, query_type):
    """Formats a single data dictionary into a list for a CSV row."""
    
    if query_type == 'combined':
        time_val = item_data.get('obs_start', '')
        time_utc_str = time_val.split('.')[0].replace('T', ' ') if time_val else ''
        return [
            item_data.get('id', ''), item_data.get('filename', ''), time_utc_str,
            item_data.get('unit', ''), item_data.get('filter', ''),
            item_data.get('seeing', ''), item_data.get('ellip', ''),
            item_data.get('ul5', ''), item_data.get('mjd', ''),
            item_data.get('filepath', ''),
            item_data.get('catalog_filepath', ''),
            item_data.get('tile', ''), item_data.get('target', ''),
            item_data.get('ra_center', ''), item_data.get('dec_center', ''),
            item_data.get('l_center', ''), item_data.get('b_center', ''),
            # NEW: Add polygon data, serialized as JSON strings
            # The API returns these as dicts, so we dump them back to strings for the CSV
            json.dumps(item_data.get('footprint')),
            json.dumps(item_data.get('footprint_galactic'))
        ]
    elif query_type == 'processed':
        time_val = item_data.get('obstime', '')
        time_utc_str = time_val.split('.')[0].replace('T', ' ') if time_val else ''
        return [
            item_data.get('id', ''), item_data.get('filename', ''), time_utc_str,
            item_data.get('unit', ''), item_data.get('filter', ''),
            item_data.get('seeing', ''), item_data.get('ellip', ''),
            item_data.get('ul5', ''), item_data.get('mjd', ''),
            item_data.get('filepath', ''),
            item_data.get('unified_filename', ''),
            item_data.get('tile', ''), item_data.get('target', ''),
            item_data.get('obsnote', ''),
            item_data.get('ra_center', ''), item_data.get('dec_center', ''),
            item_data.get('l_center', ''), item_data.get('b_center', ''),
            # NEW: Add polygon data, serialized as JSON strings
            json.dumps(item_data.get('poly')),
            json.dumps(item_data.get('poly_galactic'))
        ]
    elif query_type == 'raw':
        time_val = item_data.get('obstime', '')
        time_utc_str = time_val.split('.')[0].replace('T', ' ') if time_val else ''
        return [
            item_data.get('id', ''), item_data.get('filename', ''), time_utc_str,
            item_data.get('unit', ''), item_data.get('filter', ''),
            item_data.get('seeing', ''), item_data.get('airmass', ''),
            item_data.get('object_name', ''), item_data.get('mjd', ''),
            item_data.get('filepath', ''),
            item_data.get('unified_filename', ''),
            item_data.get('tile', ''), item_data.get('target', ''),
            item_data.get('obsnote', ''),
            item_data.get('object_ra', ''), item_data.get('object_dec', ''),
            # NEW: Add polygon data, serialized as JSON strings
            json.dumps(item_data.get('vertices')),
            json.dumps(item_data.get('vertices_galactic'))
        ]
    elif query_type == 'tile':
        return [
            item_data.get('id', ''), item_data.get('name', ''), item_data.get('ra', ''),
            item_data.get('dec', ''), item_data.get('l', ''), item_data.get('b', ''),
            item_data.get('priority', ''), item_data.get('observation_count', ''),
            item_data.get('first_observed', ''), item_data.get('last_observed', ''),
            # NEW: Add polygon data, serialized as JSON strings
            json.dumps(item_data.get('vertices')),
            json.dumps(item_data.get('vertices_galactic'))
        ]
    elif query_type == 'target':
        return [
            item_data.get('id', ''), item_data.get('name', ''), item_data.get('ra', ''),
            item_data.get('dec', ''), item_data.get('l', ''), item_data.get('b', ''),
            item_data.get('target_type', ''), item_data.get('observation_count', ''),
            item_data.get('first_observed', ''), item_data.get('last_observed', ''),
            # NEW: Add polygon data, serialized as JSON strings
            json.dumps(item_data.get('vertices')),
            json.dumps(item_data.get('vertices_galactic'))
        ]
    return []

# --- CSV Header Helper ---
def _get_csv_header(query_type):
    """Gets the appropriate CSV header list for the query type."""
    if query_type == 'combined':
        return [
            "id", "filename", "obs_start", "unit", "filter", "seeing", "ellip", "ul5", "mjd", 
            "filepath", "catalog_filepath", "tile", "target", "ra_center", "dec_center", "l_center", "b_center",
            "footprint_geojson", "footprint_galactic_geojson" # NEW: Add polygon header columns
        ]
    elif query_type == 'processed':
        return [
            "id", "filename", "obstime", "unit", "filter", "seeing", "ellip", "ul5", "mjd", 
            "filepath", "unified_filename", "tile", "target", "obsnote", "ra_center", "dec_center", "l_center", "b_center",
            "poly_geojson", "poly_galactic_geojson" # NEW: Add polygon header columns
        ]
    elif query_type == 'raw':
        return [
            "id", "filename", "obstime", "unit", "filter", "seeing", "airmass", "object_name", "mjd", 
            "filepath", "unified_filename", "tile", "target", "obsnote", "object_ra", "object_dec",
            "vertices_geojson", "vertices_galactic_geojson" # NEW: Add polygon header columns
        ]
    elif query_type == 'tile':
        return [
            "id", "name", "ra", "dec", "l", "b", "priority", "observation_count", "first_observed", "last_observed",
            "vertices_geojson", "vertices_galactic_geojson" # NEW: Add polygon header columns
        ]
    elif query_type == 'target':
        return [
            "id", "name", "ra", "dec", "l", "b", "target_type", "observation_count", "first_observed", "last_observed",
            "vertices_geojson", "vertices_galactic_geojson" # NEW: Add polygon header columns
        ]
    return []

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="GWPortal Search Tool: Query frames, tiles, or targets using various attribute and spatial filters."
    )
    parser.add_argument(
        "--type", type=str, choices=['raw', 'processed', 'combined', 'tile', 'target'], default='processed',
        help="Type of data to query. (Default: processed)"
    )
    
    # --- Date/Time Filters ---
    time_group = parser.add_argument_group('Date/Time Filters (for frames)')
    time_group.add_argument(
        "--days", type=int, default=None,
        help="Number of recent days to query (UTC obstime/obs_start). Overrides --date_start/--date_end."
    )
    time_group.add_argument(
        "--date_start", type=str, default=None,
        help="Start date for query range (YYYY-MM-DD, UTC)."
    )
    time_group.add_argument(
        "--date_end", type=str, default=None,
        help="End date for query range (YYYY-MM-DD, UTC)."
    )
    time_group.add_argument(
        "--night_date", type=str, default=None,
        # --- MODIFICATION: Updated help text ---
        help="Filter by a specific local observation night (YYYY-MM-DD). (Frames only)"
    )

    # --- Attribute Filters ---
    attr_group = parser.add_argument_group('Attribute Filters')
    attr_group.add_argument(
        "--filter", type=str, default=None, dest='filter_name',
        help="Filter by filter name (e.g., 'g', 'r'). (Frames only)"
    )
    attr_group.add_argument(
        "--unit", type=str, default=None, dest='unit_name',
        help="Filter by unit name (e.g., '7DT01'). (Frames only)"
    )
    attr_group.add_argument(
        "--object_name", type=str, default=None,
        # --- MODIFICATION: Updated help text ---
        help="Filter by object name (contains, case-insensitive). (Frames only)"
    )
    attr_group.add_argument(
        "--obsnote", type=str, default=None, dest='obsnote_contains',
        # --- MODIFICATION: Updated help text ---
        help="Filter by keyword in observation notes (contains, case-insensitive, FTS-enabled). (Frames only)"
    )
    attr_group.add_argument(
        "--target_type", type=str, default=None,
        help="Filter by target type (e.g., 'TOO', 'STD'). (Target only)"
    )

    # --- Spatial/Name Filters (for all types) ---
    spatial_group = parser.add_argument_group('Spatial & Name Filters')
    spatial_group.add_argument(
        "--tile_name", type=str, default=None,
        help="Filter by specific Tile name (exact match, case-insensitive). (Frames or Tile query)"
    )
    spatial_group.add_argument(
        "--target_name", type=str, default=None,
        help="Filter by specific Target name (contains, case-insensitive). (Frames or Target query)"
    )
    spatial_group.add_argument(
        "--coord_sys", type=str, choices=['radec', 'galactic'], default='radec',
        help="Coordinate system for spatial queries. (Default: radec)"
    )
    spatial_group.add_argument(
        "--ra", type=float, default=None, help="Right Ascension for cone/point search (degrees)."
    )
    spatial_group.add_argument(
        "--dec", type=float, default=None, help="Declination for cone/point search (degrees)."
    )
    spatial_group.add_argument(
        "--gl", type=float, default=None, help="Galactic Longitude for cone/point search (degrees)."
    )
    spatial_group.add_argument(
        "--gb", type=float, default=None, help="Galactic Latitude for cone/point search (degrees)."
    )
    spatial_group.add_argument(
        "--radius", type=float, default=None,
        help="Radius for cone search (degrees). If omitted, performs a point-in-polygon search."
    )
    spatial_group.add_argument(
        "--polygon", type=str, default=None,
        help="JSON string of a polygon for intersection search (e.g., '[[lon1,lat1], [lon2,lat2], ...]'). Coords must match --coord_sys."
    )

    # --- Output Control ---
    output_group = parser.add_argument_group('Output Control')
    output_group.add_argument(
        "--limit", type=int, default=10, dest='print_limit', metavar='N',
        help="Limit the number of results displayed in the console output (default: 10). Set to 0 or negative for no limit."
    )
    output_group.add_argument(
        "--output-csv", type=str, default=None, metavar='FILENAME.CSV',
        help="Save ALL fetched results to CSV using client-side streaming (memory efficient, uses API pagination)."
    )
    
    # --- MODIFICATION: Check if no arguments were provided ---
    # If only the script name is in sys.argv (length is 1), print help and exit.
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    # --- End of modification ---

    args = parser.parse_args()

    # --- Validation ---
    if args.days is not None:
        if args.date_start is not None or args.date_end is not None:
            print("Warning: --days parameter overrides --date_start and --date_end.", file=sys.stderr)
            args.date_start = None
            args.date_end = None
        if args.days <= 0:
             parser.error("--days must be a positive integer.")
             
    date_format = "%Y-%m-%d"
    if args.date_start:
        try: datetime.strptime(args.date_start, date_format)
        except ValueError: parser.error("Invalid --date_start format. Use YYYY-MM-DD.")
    if args.date_end:
        try: datetime.strptime(args.date_end, date_format)
        except ValueError: parser.error("Invalid --date_end format. Use YYYY-MM-DD.")
    if args.night_date:
        try: datetime.strptime(args.night_date, date_format)
        except ValueError: parser.error("Invalid --night_date format. Use YYYY-MM-DD.")

    if args.print_limit is not None and args.print_limit <= 0:
        args.print_limit = None # Use None to signify no limit internally

    # --- Spatial param validation ---
    if (args.ra is not None and args.dec is None) or (args.ra is None and args.dec is not None):
        parser.error("--ra and --dec must be used together.")
    if (args.gl is not None and args.gb is None) or (args.gl is None and args.gb is not None):
        parser.error("--gl and --gb must be used together.")
    if args.coord_sys == 'radec' and (args.gl or args.gb):
        parser.error("Cannot use --gl/--gb with --coord_sys=radec.")
    if args.coord_sys == 'galactic' and (args.ra or args.dec):
        parser.error("Cannot use --ra/--dec with --coord_sys=galactic.")
    if args.polygon and (args.ra or args.radius or args.gl):
        parser.error("Cannot use --polygon with cone search parameters (--ra/--dec/--radius or --gl/--gb/--radius).")
    
    # Check for incompatible filters
    # (This logic is already correct, as it allows 'combined' type)
    frame_only_args = [args.filter_name, args.unit_name, args.object_name, args.obsnote_contains, args.night_date]
    if args.type in ['tile', 'target'] and any(frame_only_args):
        parser.error("--filter, --unit, --object_name, --obsnote, --night_date are only for frame queries (raw, processed, combined).")
    
    tile_only_args = [] # --priority was removed
    if args.type != 'tile' and any(tile_only_args):
        parser.error("--priority is only for tile queries.")
        
    target_only_args = [args.target_type]
    if args.type != 'target' and any(target_only_args):
        parser.error("--target_type is only for target queries.")

    return args

# --- Main Script Logic ---
def main():
    script_start_time = time.time()
    args = parse_args()

    query_type = args.type.lower()

    # --- Initialize GWPortal Client ---
    try:
        client = GWPortalClient() # Reads from env vars
    except ValueError as e:
        print(f"Error initializing client: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Select Query Function ---
    query_func = None
    if query_type == 'raw':
        query_func = client.query_raw
    elif query_type == 'processed':
        query_func = client.query_processed
    elif query_type == 'combined':
        query_func = client.query_combined
    elif query_type == 'tile':
        query_func = client.query_tiles
    elif query_type == 'target':
        query_func = client.query_targets

    if not query_func:
         print(f"Error: Invalid type '{query_type}' specified.", file=sys.stderr)
         sys.exit(1)

    # --- Combine all query parameters ---
    # Map script arguments to the API parameter names defined in api/views.py
    query_params = {
        # Date/Time
        'days': args.days,
        'date_start': args.date_start,
        'date_end': args.date_end,
        'night_date': args.night_date,
        
        # Attributes
        'filter_name': args.filter_name,
        'unit_name': args.unit_name,
        'object_name': args.object_name,
        'obsnote_contains': args.obsnote_contains, # Maps to _build_base_queryset
        # 'priority' was removed
        'target_type': args.target_type,
        
        # Spatial/Name
        'tile_name': args.tile_name,
        'target_name': args.target_name,
        'coord_sys': args.coord_sys,
        'ra': args.ra,
        'dec': args.dec,
        'gl': args.gl, # Added
        'gb': args.gb, # Added
        'radius': args.radius,
        'polygon': args.polygon,
    }
    # Filter out None values
    query_params = {k: v for k, v in query_params.items() if v is not None}
    
    # --- Execute Based on Output Mode ---
    try:
        fetch_start_time = time.time()

        # --- MODE 1: Client-Side Streaming (for --output-csv) ---
        if args.output_csv:
            print(f"\n--- Mode: Client-Side Streaming to CSV ---")
            results_generator = client.stream_all_results(
                query_func,
                desc=f"Streaming {query_type}",
                **query_params
            )
            
            total_items_found = next(results_generator, 0)
            
            sample_items_for_report = []
            print_limit_count = args.print_limit if args.print_limit is not None else 0
            
            csv_writer = None
            csv_file = None
            processed_count = 0
            try:
                csv_file = open(args.output_csv, 'w', newline='', encoding='utf-8')
                csv_writer = csv.writer(csv_file)
                header = _get_csv_header(query_type) # Get dynamic header
                if not header:
                    raise ValueError(f"No CSV header defined for type '{query_type}'")
                csv_writer.writerow(header)
                print(f"Streaming {total_items_found} items to {args.output_csv}...")

                with tqdm(total=total_items_found, unit=" item", desc=f"Writing CSV", file=sys.stdout) as pbar:
                    for item_data in results_generator:
                        if len(sample_items_for_report) < print_limit_count:
                            sample_items_for_report.append(item_data)
                        
                        row = _format_item_for_csv(item_data, query_type)
                        csv_writer.writerow(row)
                        processed_count += 1
                        pbar.update(1)

            except IOError as e:
                 print(f"\nERROR: Could not write to CSV file {args.output_csv}: {e}", file=sys.stderr)
                 sys.exit(1)
            finally:
                if csv_file:
                    csv_file.close()
                    if processed_count == total_items_found:
                         print(f"\nSuccessfully saved {processed_count} results to {args.output_csv}")
                    else:
                         print(f"\nSaved {processed_count}/{total_items_found} results to {args.output_csv} (stream might have been interrupted).")

            fetch_duration = time.time() - fetch_start_time
            print(f"\nClient-side streaming and CSV writing took {fetch_duration:.2f} seconds.")
            
            # Show console report *after* CSV is written
            format_and_print_results(sample_items_for_report, total_items_found, args)


        # --- MODE 2: Standard In-Memory Fetch (default, no CSV output) ---
        else:
            print(f"\n--- Mode: Standard In-Memory Fetch (Console Output Only) ---")
            all_items_fetched = client.get_all_results(
                query_func,
                desc=f"Fetching {query_type}",
                **query_params
            )
            
            fetch_duration = time.time() - fetch_start_time
            print(f"\nData fetching took {fetch_duration:.2f} seconds.")

            total_items = len(all_items_fetched)
            
            # Get sample for console
            sample_to_print = all_items_fetched[:args.print_limit] if args.print_limit is not None else all_items_fetched
            
            # Show console report
            format_and_print_results(sample_to_print, total_items, args)

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


if __name__ == "__main__":
    main()
