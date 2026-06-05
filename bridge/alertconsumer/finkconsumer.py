
'''
This code manages the FINK consumer for the ToO trigger pipeline in BRIDGE. 

1. Manage configuration of the FINK consumer
2. Manage connection to the FINK broker
3. Store the alerts in a database
'''
#%%
from pathlib import Path
import yaml
import subprocess
import time
import pandas as pd
import threading

from fink_client.avro_utils import AlertReader

from bridge.configuration import Configuration
from bridge.connector import SQLConnector
#%%
class FINKConsumer:
    def __init__(self,
                 survey_type = 'ztf', # ZTF or LSST
                 ):
        self.survey_type = survey_type
        self.config = Configuration(config_filenames=['finkconsumer.config'])
        self.queryconfig = self.config._dict[self.survey_type]['queryconfig']
        self.is_consuming = False
        self.is_monitoring = False
        self.process = None
        self.monitor_process = None
        self.sql = SQLConnector()
        
    def __repr__(self):
        return f'FINKConsumer(survey_type = {self.survey_type}, is_consuming = {self.is_consuming}, is_monitoring = {self.is_monitoring})'

    def run(self):
        if self.is_consuming:
            print("FINKConsumer is already running.")
            return

        self._save_queryconfig()

        num_alert_limit = self.config._dict[self.survey_type]['num_alerts_limit']
        verbose = self.config._dict[self.survey_type]['verbose']
        outdir = self.config._dict[self.survey_type]['outdir']

        command = ["fink_consumer", "-survey", self.survey_type]

        if verbose:
            command.append("--display")

        if num_alert_limit != 999:
            command.extend(["-limit", str(num_alert_limit)])

        command.extend([
            "-start_at", "latest",
            "--save",
            "-outdir", outdir
        ])

        print("Starting FINKConsumer...")
        print("Command:", " ".join(command))

        self.process = subprocess.Popen(command)

        self.is_consuming = True

        # 🔥 Start monitoring thread
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self.monitor_alert_folder,
            daemon=True
        )
        self.monitor_thread.start()

        print("FINKConsumer Monitoring started.")

    def stop(self):

        if not self.is_consuming:
            print("FINKConsumer is not running.")
            return

        print("Stopping FINKConsumer...")

        # Stop consumer process
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()

        self.process = None
        self.is_consuming = False

        # 🔥 Stop monitoring thread
        self.is_monitoring = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        self.monitor_thread = None

        print("FINKConsumer and Monitoring stopped.")
        
    def monitor_alert_folder(self):
        outdir = Path(self.config._dict[self.survey_type]['outdir'])
        processed_folder = outdir / 'processed'
        processed_folder.mkdir(exist_ok=True)

        print("Monitoring alert folder...")

        while self.is_monitoring:

            for file in outdir.glob('*.avro'):
                try:
                    result = self.add_alert_to_db(file.name)

                    if result:
                        file.rename(processed_folder / file.name)

                except Exception as e:
                    print(f"Error processing {file.name}: {e}")

            time.sleep(5)   # reduce latency
            
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
                # For ZTF
                'prv_candidates',
                # For LSST
                'prvDiaSources',
                'prvDiaForcedSources',
                # 'diaObject',
                'ssSource',
                'mpc_orbits',
                # For both
                'cutoutScience',
                'cutoutTemplate',
                'cutoutDifference',
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
    
    def _save_queryconfig(self):
        queryconfig_path = Path.home() / '.finkclient' / f'{self.survey_type}_credentials.yml'
        with open(queryconfig_path, 'w') as f:
            yaml.dump(self.queryconfig, f)
        print(f'Fink Configuration Saved to: {queryconfig_path}')