# %%
from pathlib import Path
import threading

import pandas as pd
from astropy.table import Table
from astropy.io import ascii
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

from bridge.utils.avro_writer import AvroWriter
from bridge.configuration import Configuration
from bridge.connector import SQLConnector
from bridge.alertquerier import TNSQuerier

#%%
class TNSConsumer:
    def __init__(self):
        self.config = Configuration(config_filenames=['tnsconsumer.config'])
        self.tnsconfig = self.config._dict.get('queryconfig', {})

        self.sql = SQLConnector()
        self.querier = TNSQuerier()
        self.avro_writer = AvroWriter()

        self.is_consuming = False
        self.is_monitoring = False

        self.receiver_thread = None
        self.monitor_thread = None

        self._stop_event = threading.Event()
        self._receiver_lock = threading.Lock()

    def __repr__(self):
        return (
            f"TNSConsumer("
            f"is_consuming={self.is_consuming}, "
            f"is_monitoring={self.is_monitoring})"
        )

    def run(self):
        """
        Start receiver loop and monitoring loop.
        Safe against repeated calls.
        """
        if self.is_consuming or self.is_monitoring:
            print("TNSConsumer is already running.")
            return

        outdir = Path(self.config._dict['outdir'])
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / 'processed').mkdir(parents=True, exist_ok=True)

        self._stop_event.clear()
        self.is_consuming = True
        self.is_monitoring = True

        print("Starting TNSConsumer...")

        self.receiver_thread = threading.Thread(
            target=self._receive_loop,
            name="TNSReceiver",
            daemon=True,
        )
        self.monitor_thread = threading.Thread(
            target=self.monitor_alert_folder,
            name="TNSMonitor",
            daemon=True,
        )

        self.receiver_thread.start()
        self.monitor_thread.start()

        print("TNSConsumer started.")

    def stop(self, join_timeout=10):
        """
        Stop receiver loop and monitoring loop gracefully.
        """
        if not self.is_consuming and not self.is_monitoring:
            print("TNSConsumer is not running.")
            return

        print("Stopping TNSConsumer...")

        self._stop_event.set()
        self.is_consuming = False
        self.is_monitoring = False

        if self.receiver_thread is not None and self.receiver_thread.is_alive():
            self.receiver_thread.join(timeout=join_timeout)
            if self.receiver_thread.is_alive():
                print("Warning: receiver_thread did not stop within timeout.")

        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=join_timeout)
            if self.monitor_thread.is_alive():
                print("Warning: monitor_thread did not stop within timeout.")

        self.receiver_thread = None
        self.monitor_thread = None

        print("TNSConsumer stopped.")

    def _receive_loop(self):
        """
        Periodically query TNS and save them as AVRO files until stop is requested.
        """
        search_cadence = self.config._dict.get('search_cadence', 300)
        verbose = self.config._dict.get('verbose', True)

        print("Receiver loop started.")

        while not self._stop_event.is_set():
            try:
                with self._receiver_lock:
                    num_alerts = self.receive_alerts(verbose=verbose)
                    if num_alerts > 0:
                        print(f"[TNS] Received {num_alerts} alerts")
                    else:
                        print(f"[TNS] No new alerts in the last {search_cadence} seconds")

            except Exception as e:
                print(f"Error in receive loop: {e}")

            if self._stop_event.wait(timeout=search_cadence):
                break

        print("Receiver loop stopped.")

    def monitor_alert_folder(self):
        outdir = Path(self.config._dict['outdir'])
        processed_folder = outdir / 'processed'
        processed_folder.mkdir(parents=True, exist_ok=True)

        print("Monitoring alert folder...")

        while not self._stop_event.is_set():
            try:
                for file in outdir.glob('*.txt'):
                    if self._stop_event.is_set():
                        break

                    processed_path = processed_folder / file.name

                    if processed_path.exists():
                        try:
                            file.unlink()
                        except Exception as e:
                            print(f"Error deleting duplicated file {file.name}: {e}")
                        continue

                    try:
                        result = self.add_alert_to_db(file)
                        if result:
                            file.rename(processed_path)
                    except Exception as e:
                        print(f"Error processing {file.name}: {e}")

            except Exception as e:
                print(f"Error in monitor loop: {e}")

            if self._stop_event.wait(timeout=5):
                break

        print("Monitoring loop stopped.")

    def receive_alerts(self, verbose: bool = True):
        """
        Query TNS, convert results to per-object AVRO files, and save only new objects.
        """
        outdir = Path(self.config._dict['outdir'])
        outdir.mkdir(parents=True, exist_ok=True)
        
        tbl = self.query_alerts(verbose=verbose)

        if tbl is None or len(tbl) == 0:
            return 0

        if not isinstance(tbl, Table):
            tbl = Table.from_pandas(pd.DataFrame(tbl))

        num_alerts = len(tbl)
        return num_alerts

    def query_alerts(self, verbose: bool = True):
        """
        Build TNS query parameters from config and search for recent objects.
        """
        search_days = self.config._dict.get('search_days', 1)
        timeout_seconds = self.config._dict.get('timeout_seconds', 600)
        outdir = self.config._dict.get('outdir', 'tns_search')

        date_end = Time.now().datetime.strftime('%Y-%m-%d')
        date_start = (Time.now() - search_days).datetime.strftime('%Y-%m-%d')

        url_parameters = dict(self.tnsconfig)
        url_parameters['discovery_date_start'] = date_start
        url_parameters['discovery_date_end'] = date_end

        if verbose:
            print("Querying recent objects from TNS")
            print("===== Configuration =====")
            for key, value in url_parameters.items():
                print(f"{key}: {value}")
            print("=========================")

        file_ = self.querier._search_tns(
            url_parameters=url_parameters,
            save_dir=outdir,
            timeout_seconds=timeout_seconds,
            verbose=verbose,
        )

        if file_ is None:
            return Table()

        try:
            tbl = ascii.read(file_, format='csv')
        except Exception:
            # some TNS outputs may still parse better with pandas
            df = pd.read_csv(file_)
            tbl = Table.from_pandas(df)

        return tbl

    def add_alert_to_db(self, filepath: str):
        tbl = Table.read(filepath, format='csv')
        try:
            alert_df = tbl.to_pandas()

            remove_columns = []
            alert_df = alert_df.drop(columns=remove_columns, errors='ignore')

            flat_df = self._flatten_alert_dataframe(alert_df)
            flat_df = self._add_radec_deg(flat_df)

            if 'ID' in flat_df.columns:
                flat_df = flat_df.drop_duplicates(subset=['ID'])

            table_name = self.config._dict['db_table_name']

            # dtype 보존된 상태로 테이블 생성
            self.sql.create_table_from_dataframe(table_name, flat_df)

            # insert 직전에만 None 처리
            insert_df = self._sanitize_for_sql(flat_df)
            self.sql.insert_dataframe(table_name, insert_df)

            return True

        except Exception as e:
            print(f"Error adding alert to database: {e}")
            return False
        
    def _flatten_alert_dataframe(self, df):
        """
        Flatten nested dict columns if any.
        TNS CSV is usually already flat, but keep same interface as ALERCEConsumer.
        """
        df = df.copy()
        cols_to_drop = []

        for col in df.columns:
            first_valid = df[col].dropna()
            if len(first_valid) == 0:
                continue

            if isinstance(first_valid.iloc[0], dict):
                nested = pd.json_normalize(df[col])
                nested.columns = [f"{col}_{subcol}" for subcol in nested.columns]
                df = pd.concat([df, nested], axis=1)
                cols_to_drop.append(col)

        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        return df
    
    def _sanitize_for_sql(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # astropy masked / pandas nullable dtype / NaN / NaT -> Python None
        df = df.convert_dtypes()

        for col in df.columns:
            # object로 바꿔야 None이 제대로 들어감
            df[col] = df[col].astype(object)
            df[col] = df[col].where(pd.notna(df[col]), None)

        return df
    
    def _add_radec_deg(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        ra_deg_list = []
        dec_deg_list = []

        # 후보 컬럼명들
        ra_candidates = ['RA', 'ra', 'radeg', 'ra_deg']
        dec_candidates = ['DEC', 'Dec', 'dec', 'decdeg', 'dec_deg']

        ra_col = next((col for col in ra_candidates if col in df.columns), None)
        dec_col = next((col for col in dec_candidates if col in df.columns), None)

        if ra_col is None or dec_col is None:
            return df

        for ra_val, dec_val in zip(df[ra_col], df[dec_col]):
            try:
                if pd.isna(ra_val) or pd.isna(dec_val):
                    ra_deg_list.append(None)
                    dec_deg_list.append(None)
                    continue

                coord = SkyCoord(
                    ra_val,
                    dec_val,
                    unit=(u.hourangle, u.deg)
                )
                ra_deg_list.append(float(coord.ra.deg))
                dec_deg_list.append(float(coord.dec.deg))

            except Exception:
                ra_deg_list.append(None)
                dec_deg_list.append(None)

        df['ra_deg'] = ra_deg_list
        df['dec_deg'] = dec_deg_list

        return df

