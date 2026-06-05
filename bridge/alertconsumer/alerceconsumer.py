
'''
This code manages the FINK consumer for the ToO trigger pipeline in BRIDGE. 

1. Manage configuration of the FINK consumer
2. Manage connection to the FINK broker
3. Store the alerts in a database
'''


#%%
from pathlib import Path
import threading
import pandas as pd
from astropy.table import vstack, Table
from astropy.time import Time
from fastavro import writer
import numpy as np

from fink_client.avro_utils import AlertReader
from bridge.utils.avro_writer import AvroWriter
from bridge.configuration import Configuration
from bridge.connector import SQLConnector
from bridge.alertquerier import ALERCEQuerier


class ALERCEConsumer:
    def __init__(self, survey_type='ztf'):
        self.survey_type = survey_type
        self.config = Configuration(config_filenames=['alerceconsumer.config'])
        self.alerceconfig = self.config._dict[self.survey_type]['queryconfig']

        self.sql = SQLConnector()
        self.querier = ALERCEQuerier(survey_type=self.survey_type)
        self.querier.config._dict[self.survey_type]['queryconfig'].update(self.alerceconfig)
        self.avro_writer = AvroWriter()

        self.is_consuming = False
        self.is_monitoring = False

        self.receiver_thread = None
        self.monitor_thread = None

        self._stop_event = threading.Event()
        self._receiver_lock = threading.Lock()

    def __repr__(self):
        return (
            f"ALERCEConsumer(survey_type={self.survey_type}, "
            f"is_consuming={self.is_consuming}, is_monitoring={self.is_monitoring})"
        )

    def run(self):
        """
        Start receiver loop and monitoring loop.
        Safe against repeated calls.
        """
        if self.is_consuming or self.is_monitoring:
            print("ALERCEConsumer is already running.")
            return

        outdir = Path(self.config._dict[self.survey_type]['outdir'])
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / 'processed').mkdir(parents=True, exist_ok=True)

        self._stop_event.clear()
        self.is_consuming = True
        self.is_monitoring = True

        print(f"Starting ALERCEConsumer for survey_type={self.survey_type}...")

        self.receiver_thread = threading.Thread(
            target=self._receive_loop,
            name=f"ALERCEReceiver-{self.survey_type}",
            daemon=True,
        )
        self.monitor_thread = threading.Thread(
            target=self.monitor_alert_folder,
            name=f"ALERCEMonitor-{self.survey_type}",
            daemon=True,
        )

        self.receiver_thread.start()
        self.monitor_thread.start()

        print("ALERCEConsumer started.")

    def stop(self, join_timeout=10):
        """
        Stop receiver loop and monitoring loop gracefully.
        """
        if not self.is_consuming and not self.is_monitoring:
            print("ALERCEConsumer is not running.")
            return

        print(f"Stopping ALERCEConsumer for survey_type={self.survey_type}...")

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

        print("ALERCEConsumer stopped.")

    def _receive_loop(self):
        """
        Periodically query alerts and save them as AVRO files until stop is requested.
        """
        search_cadence = self.config._dict[self.survey_type].get('search_cadence', 30)
        verbose = self.config._dict[self.survey_type].get('verbose', True)

        print("Receiver loop started.")

        while not self._stop_event.is_set():
            try:
                with self._receiver_lock:
                    num_alerts = self.receive_alerts(verbose=verbose)
                    if num_alerts > 0:
                        print(f"Received {num_alerts} alerts")
                    else:
                        print(f"[ALERCE, {self.survey_type}] No alerts the last {search_cadence} seconds")
                        
            except Exception as e:
                print(f"Error in receive loop: {e}")

            # sleep cooperatively so stop() can interrupt quickly
            if self._stop_event.wait(timeout=search_cadence):
                break

        print("Receiver loop stopped.")
        
    def monitor_alert_folder(self):
        outdir = Path(self.config._dict[self.survey_type]['outdir'])
        processed_folder = outdir / 'processed'
        processed_folder.mkdir(parents=True, exist_ok=True)

        print("Monitoring alert folder...")

        while not self._stop_event.is_set():
            try:
                for file in outdir.glob('*.avro'):
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
                        result = self.add_alert_to_db(file.name)
                        if result:
                            file.rename(processed_path)
                    except Exception as e:
                        print(f"Error processing {file.name}: {e}")

            except Exception as e:
                print(f"Error in monitor loop: {e}")

            if self._stop_event.wait(timeout=5):
                break

        print("Monitoring loop stopped.")
        

    def receive_alerts(self, 
                       verbose: bool = True):
        # helper: convert numpy scalars → python scalars
        alert_tbl = self.querier.query_alerts(firstmjd = Time.now().mjd - self.config._dict[self.survey_type]['search_days'], verbose = verbose)
        all_oids = [alert['oid'] for alert in alert_tbl]
        all_oids = list(set(all_oids))
        all_detection_tbl = Table()
        if self.survey_type == 'lsst':

            detection_tbl = self.querier.query_detections(all_oids, verbose = verbose)
            
            if len(detection_tbl) == 0:
                return 0

            # build alert lookup
            alert_dict = {str(row['oid']): row for row in alert_tbl}

            rows = []

            for oid in all_oids:

                det = detection_tbl[detection_tbl['oid'] == oid]

                if len(det) == 0:
                    continue

                alert_row = alert_dict.get(str(oid))

                # add alert columns to detection table
                for col in alert_tbl.colnames:
                    if col == 'oid':
                        continue
                    det[col] = [alert_row[col]] * len(det)

                rows.append(det)

            if rows:
                all_detection_tbl = vstack(rows)
                
            object_id_key = 'oid'
            source_id_key = 'measurement_id'

        elif self.survey_type == 'ztf':
            for oid in all_oids:
                detection_tbl = self.querier.query_detections(oid, verbose = False)
                if len(detection_tbl) == 0:
                    continue
                alert_tbl_target = alert_tbl[alert_tbl['oid'] == oid]
                for col in alert_tbl_target.columns:
                    detection_tbl[col] = alert_tbl_target[col]
                all_detection_tbl = vstack([all_detection_tbl, detection_tbl])

            object_id_key = 'oid'
            source_id_key = 'candid'
        else:
            raise RuntimeError(f"Survey type {self.survey_type} not supported")
        
        schema = self.avro_writer.make_schema(all_detection_tbl)

        Path(self.config._dict[self.survey_type]['outdir']).mkdir(exist_ok=True)
        num_alerts = 0
        for detection in all_detection_tbl:
            filename = f"{Path(self.config._dict[self.survey_type]['outdir'])}/{detection[object_id_key]}_{detection[source_id_key]}.avro"
            if self.is_alert_saved(Path(filename).name):
                continue
            field_types = {f["name"]: f["type"] for f in schema["fields"]}

            record = {
                col: self.avro_writer.normalize_value(detection[col], field_types[col])
                for col in all_detection_tbl.colnames
            }

            with open(filename, "wb") as out:
                writer(out, schema, [record])
            num_alerts += 1
        return num_alerts
            
    def read_alert(self, filename: str):
        filepath = Path(self.config._dict[self.survey_type]['outdir']) / filename
        if not filepath.exists():
            filepath = Path(self.config._dict[self.survey_type]['outdir']) / 'processed' / filename
            if not filepath.exists():
                raise FileNotFoundError(f"File {filepath} not found.")
        r = AlertReader(str(filepath))
        return r.to_pandas()
    
    def add_alert_to_db(self, filename: str):
        
        try:
            alert_df = self.read_alert(filename)

            # Remove heavy / nested columns safely
            remove_columns = [
            ]

            alert_df = alert_df.drop(columns=remove_columns, errors='ignore')

            flat_df = self._flatten_alert_dataframe(alert_df)

            table_name = self.config._dict[self.survey_type]['db_table_name']

            # Always safe due to IF NOT EXISTS
            self.sql.create_table_from_dataframe(table_name, flat_df)

            # Insert (must use ON CONFLICT DO NOTHING)
            self.sql.insert_dataframe(table_name, flat_df)
            return True
        
        except Exception as e:
            print(f"Error adding alert to database: {e}")
            return False
    
    def is_alert_saved(self, filename: str):
        filepath = Path(self.config._dict[self.survey_type]['outdir']) / filename
        if not filepath.exists():
            filepath = Path(self.config._dict[self.survey_type]['outdir']) / 'processed' / filename
            if not filepath.exists():
                return False
        return True
        
    @property
    def alertlist(self):
        alert_dict = dict()
        alert_dict['unprocessed'] = [file.name for file in list(Path(self.config._dict[self.survey_type]['outdir']).glob('*.avro'))]
        alert_dict['processed'] = [file.name for file in list((Path(self.config._dict[self.survey_type]['outdir']) / 'processed').glob('*.avro'))]
        return alert_dict
        
    def _flatten_alert_dataframe(self, df):

        df = df.copy()
        cols_to_drop = []

        for col in df.columns:

            # ==============================
            # 🔥 Special handling for lc_features
            # ==============================
            if col == "lc_features":

                filters = ["u", "g", "r", "i", "z", "y"]

                # 1️⃣ 모든 feature key 수집
                feature_keys = set()
                for val in df[col].dropna():
                    if isinstance(val, dict):
                        for filt in val:
                            if isinstance(val[filt], dict):
                                feature_keys.update(val[filt].keys())

                feature_keys = list(feature_keys)

                # 2️⃣ 강제 expansion
                expanded_rows = []

                for val in df[col]:

                    row_dict = {}

                    if not isinstance(val, dict):
                        val = {}

                    for filt in filters:

                        subdict = val.get(filt, {})

                        for feat in feature_keys:
                            keyname = f"{col}_{filt}.{feat}"
                            row_dict[keyname] = (
                                subdict.get(feat, None)
                                if isinstance(subdict, dict)
                                else None
                            )

                    expanded_rows.append(row_dict)

                expanded_df = pd.DataFrame(expanded_rows)

                df = pd.concat([df, expanded_df], axis=1)
                cols_to_drop.append(col)

                continue  # 🔥 skip normal dict logic

            # ==============================
            # 🔹 Normal dict columns
            # ==============================

            first_valid = df[col].dropna()
            if len(first_valid) == 0:
                continue

            if isinstance(first_valid.iloc[0], dict):

                nested = pd.json_normalize(df[col])
                nested.columns = [f"{col}_{subcol}" for subcol in nested.columns]

                df = pd.concat([df, nested], axis=1)
                cols_to_drop.append(col)

        df.drop(columns=cols_to_drop, inplace=True)

        return df
# %%
if __name__ == '__main__':
    self = ALERCEConsumer('ztf')
    self.run()