

#%%

from pathlib import Path
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from tqdm import tqdm
import time
from collections import deque
import datetime
from astropy.table import unique
import math
import numpy as np
import datetime
from astropy.time import Time

from uuid import uuid4
from bridge.configuration import Configuration
from bridge.alertfilter import ZTF_Alerce_Filter, LSST_Alerce_Filter, ZTF_Fink_Filter, LSST_Fink_Filter, TNS_Filter
from bridge.utils import HostGalaxyCatalog, ExposureCalculator
from bridge.objects import Alert
from bridge.connector import SQLConnector, SDSConnector, GmailConnector
from bridge.objects import Template
from bridge.alertquerier import ALERCEQuerier, FINKQuerier, TNSQuerier
#%%
ZTF_FILTER_MAP = {1 : 'g', 2 : 'r', 3 : 'i'}
#%%
class AlertTrigger:
    def __init__(self):
        self.config = Configuration(config_filenames=['alerttrigger.config'])
        self.initialize()
        self.trigger_history = deque(maxlen=20)  # keep last 20 trigger timestamps
        
    def initialize(self):
        print('Start initializing the AlertTrigger...')
        self.hostgalaxycatalog = HostGalaxyCatalog()
        self.exposurecalculator = ExposureCalculator()
        self.exposurecalculator.load_model()
        
        # Set queriers
        self.querier = dict()
        self.querier['ztf'] = dict()
        self.querier['ztf']['alerce'] = ALERCEQuerier(survey_type = 'ztf')
        self.querier['ztf']['fink'] = FINKQuerier(survey_type = 'ztf')
        self.querier['lsst'] = dict()
        self.querier['lsst']['alerce'] = ALERCEQuerier(survey_type = 'lsst')
        self.querier['lsst']['fink'] = FINKQuerier(survey_type = 'lsst')
        self.querier['tns'] = TNSQuerier()
        
        # Set filter
        self.filter = dict()
        self.filter['ztf'] = dict()
        self.filter['ztf']['alerce'] = ZTF_Alerce_Filter(hostgalaxycatalog = self.hostgalaxycatalog)
        self.filter['ztf']['fink'] = ZTF_Fink_Filter(hostgalaxycatalog = self.hostgalaxycatalog)
        self.filter['lsst'] = dict()
        self.filter['lsst']['alerce'] = LSST_Alerce_Filter(hostgalaxycatalog = self.hostgalaxycatalog)
        self.filter['lsst']['fink'] = LSST_Fink_Filter(hostgalaxycatalog = self.hostgalaxycatalog)
        self.filter['tns'] = TNS_Filter(hostgalaxycatalog = self.hostgalaxycatalog)
        
        # Set Connectors
        self.db = SQLConnector()
        self.sds = SDSConnector()
        self.mail = GmailConnector()
        print('AlertTrigger initialized successfully!')
        
    def receive_alerts(self, utctime: Time = None, verbose: bool = False):
        if utctime is None:
            utctime = Time.now()

        # ALERCE ZTF Alert broker. (Receive -> Update DB)
        if self.config.receive_ztf_alerce:
            ztf_alerce_tbl = self.db.get_data(tbl_name = self.config.db_name['ztf_alerce'], select_key = '*', where_key = 'status', where_value = 'new')
            if len(ztf_alerce_tbl) == 0:
                print('No ZTF ALERCE alerts found!')
            else:
                input_count = 0
                ztf_alerce_tbl_filtered = self.filter['ztf']['alerce'].apply(ztf_alerce_tbl, verbose = False)
                filtered_id_set = set(str(row['candid']) for row in ztf_alerce_tbl_filtered)
                rejected_id_set = set(str(row['candid']) for row in ztf_alerce_tbl) - filtered_id_set
                
                for rejected_id in rejected_id_set:
                    self.db.update_row(tbl_name = self.config.db_name['ztf_alerce'], update_key = 'status', update_value = 'rejected', id_key = 'candid', id_value = rejected_id)
                
                if len(ztf_alerce_tbl_filtered) != 0:
                    ztf_alerce_tbl_filtered.sort('mjd')
                    ztf_alerce_tbl_filtered = unique(ztf_alerce_tbl_filtered, keys = 'oid')
                    if verbose:
                        ztf_alerce_iterator = tqdm(ztf_alerce_tbl_filtered, desc = 'Checking alerts...')
                        print(f"[ZTF, ALERCE] {len(ztf_alerce_tbl_filtered)} new alerts are found!")
                    else:
                        ztf_alerce_iterator = ztf_alerce_tbl_filtered
                    
                    for row in ztf_alerce_iterator:
                        row_dict = {k: v for k, v in dict(row).items() if k not in ['ra', 'dec']}
                        alert_instance = Alert(
                            ra = float(row['ra']),
                            dec = float(row['dec']),
                            objname = str(row['oid']),
                            alert_time = Time(row['firstmjd'], format='mjd'),
                            source = 'ALERCE_ZTF',
                            **row_dict
                            )
                        
                        alert_already_in_db = self.get_alert_db_info(ra = alert_instance.ra, 
                                                                     dec = alert_instance.dec,
                                                                     alert_time=alert_instance.alert_time)
                        if alert_already_in_db is not None:
                            self.db.update_row(tbl_name = self.config.db_name['ztf_alerce'], update_key = 'status', update_value = 'passed', id_key = 'oid', id_value = str(row['oid']))
                            continue
                            
                        input_count +=1
                        if verbose:
                            print("NEW ALERT FOUND!")
                        self.insert_alert_db_info(alert_instance)
                        self.db.update_row(tbl_name = self.config.db_name['ztf_alerce'], update_key = 'status', update_value = 'passed', id_key = 'oid', id_value = str(row['oid']))
                        
                print(f"[ZTF, ALERCE] {input_count}/{len(ztf_alerce_tbl)} new alerts")
            
        # FINK ZTF Alert broker. (Receive -> Update DB)
        if self.config.receive_ztf_fink:
            ztf_fink_tbl = self.db.get_data(tbl_name = self.config.db_name['ztf_fink'], select_key = '*', where_key = 'status', where_value = 'new')
            if len(ztf_fink_tbl) == 0:
                print('No ZTF FINK alerts found!')
            else:
                input_count = 0
                ztf_fink_tbl_filtered = self.filter['ztf']['fink'].apply(ztf_fink_tbl, verbose = False)
                # ztf_fink_tbl_filtered = self.filter['ztf']['fink'].apply_catalog_filter(ztf_fink_tbl_filtered, verbose = True)
                # ztf_fink_tbl_filtered = unique(ztf_fink_tbl_filtered, keys = 'objectId')
                # ztf_fink_tbl_filtered = self.filter['ztf']['fink'].apply_hostgalaxy_filter(ztf_fink_tbl_filtered, verbose = True)

                filtered_id_set = set(str(row['candid']) for row in ztf_fink_tbl_filtered)
                rejected_id_set = set(str(row['candid']) for row in ztf_fink_tbl) - filtered_id_set
                
                for rejected_id in rejected_id_set:
                    self.db.update_row(tbl_name = self.config.db_name['ztf_fink'], update_key = 'status', update_value = 'rejected', id_key = 'candid', id_value = rejected_id)
                
                if len(ztf_fink_tbl_filtered) != 0:
                    ztf_fink_tbl_filtered.sort('candidate_jd')
                    ztf_fink_tbl_filtered = unique(ztf_fink_tbl_filtered, keys = 'objectId')
                    if verbose:
                        ztf_fink_iterator = tqdm(ztf_fink_tbl_filtered, desc = 'Checking alerts...')
                        print(f"[ZTF, FINK] {len(ztf_fink_tbl_filtered)} new alerts are found!")
                    else:
                        ztf_fink_iterator = ztf_fink_tbl_filtered
                    alert_instances = []
                    for row in ztf_fink_iterator:
                        alert_instance = Alert(
                            ra = float(row['candidate_ra']),
                            dec = float(row['candidate_dec']),
                            objname = str(row['objectId']),
                            alert_time = Time(row['candidate_jd'], format='jd'),
                            source = 'FINK_ZTF', 
                            **row
                        )
                        alert_instances.append(alert_instance)
                        alert_already_in_db = self.get_alert_db_info(ra = alert_instance.ra, 
                                                                     dec = alert_instance.dec,
                                                                     alert_time=alert_instance.alert_time)
                        if alert_already_in_db is not None:
                            self.db.update_row(tbl_name = self.config.db_name['ztf_fink'], update_key = 'status', update_value = 'passed', id_key = 'objectId', id_value = str(row['objectId']))
                            continue
                            
                        input_count += 1
                        if verbose:
                            print("NEW ALERT FOUND!")
                        self.insert_alert_db_info(alert_instance)
                        self.db.update_row(tbl_name = self.config.db_name['ztf_fink'], update_key = 'status', update_value = 'passed', id_key = 'objectId', id_value = str(row['objectId']))
                print(f"[ZTF, FINK] {input_count}/{len(ztf_fink_tbl)} new alerts")
            
        # ALERCE LSST Alert broker. (Receive -> Update DB)
        if self.config.receive_lsst_alerce:
            lsst_alerce_tbl = self.db.get_data(tbl_name = self.config.db_name['lsst_alerce'], select_key = '*', where_key = 'status', where_value = 'new')
            if len(lsst_alerce_tbl) == 0:
                print('No LSST ALERCE alerts found!')
            else:
                input_count = 0
                lsst_alerce_tbl_filtered = self.filter['lsst']['alerce'].apply(lsst_alerce_tbl, verbose = False)
                filtered_id_set = set(str(row['measurement_id']) for row in lsst_alerce_tbl_filtered) ########## VARIABLE COLUMN NAME
                rejected_id_set = set(str(row['measurement_id']) for row in lsst_alerce_tbl) - filtered_id_set ########## VARIABLE COLUMN NAME
                
                for rejected_id in rejected_id_set:
                    self.db.update_row(tbl_name = self.config.db_name['lsst_alerce'], update_key = 'status', update_value = 'rejected', id_key = 'measurement_id', id_value = rejected_id)
                
                if len(lsst_alerce_tbl_filtered) != 0:
                    lsst_alerce_tbl_filtered.sort('mjd')
                    lsst_alerce_tbl_filtered = unique(lsst_alerce_tbl_filtered, keys = 'diaObjectId')
                    
                    if verbose:
                        lsst_alerce_iterator = tqdm(lsst_alerce_tbl_filtered, desc = 'Checking alerts...')
                        print(f"[LSST, ALERCE] {len(lsst_alerce_tbl_filtered)} new alerts are found!")
                    else:
                        lsst_alerce_iterator = lsst_alerce_tbl_filtered
                    
                    for row in lsst_alerce_iterator:
                        row_dict = {k: v for k, v in dict(row).items() if k not in ['ra', 'dec']}
                        alert_instance = Alert(
                            ra = float(row['ra']),
                            dec = float(row['dec']),
                            objname = str(row['diaObjectId']),
                            alert_time = Time(row['firstmjd'], format='mjd'),
                            source = 'ALERCE_LSST',
                            **row_dict
                        )
                        
                        alert_already_in_db = self.get_alert_db_info(ra = alert_instance.ra, 
                                                                     dec = alert_instance.dec,
                                                                     alert_time=alert_instance.alert_time)
                        if alert_already_in_db is not None:
                            self.db.update_row(tbl_name = self.config.db_name['lsst_alerce'], update_key = 'status', update_value = 'passed', id_key = 'diaObjectId', id_value = str(row['diaObjectId']))
                            continue
                            
                        input_count += 1
                        if verbose:
                            print("NEW ALERT FOUND!")
                        self.insert_alert_db_info(alert_instance)
                        self.db.update_row(tbl_name = self.config.db_name['lsst_alerce'], update_key = 'status', update_value = 'passed', id_key = 'diaObjectId', id_value = str(row['diaObjectId']))
                print(f"[LSST, ALERCE] {input_count}/{len(lsst_alerce_tbl)} new alerts")
            
        # FINK LSST Alert broker. (Receive -> Update DB)
        if self.config.receive_lsst_fink:
            lsst_fink_tbl = self.db.get_data(tbl_name = self.config.db_name['lsst_fink'], select_key = '*', where_key = 'status', where_value = 'new')
            if len(lsst_fink_tbl) == 0:
                print('No LSST FINK alerts found!')
            else:
                input_count = 0
                lsst_fink_tbl_filtered = self.filter['lsst']['fink'].apply(lsst_fink_tbl, verbose = False)
                filtered_id_set = set(str(row['diaSourceId']) for row in lsst_fink_tbl_filtered) ########## VARIABLE COLUMN NAME
                rejected_id_set = set(str(row['diaSourceId']) for row in lsst_fink_tbl) - filtered_id_set ########## VARIABLE COLUMN NAME
                
                for rejected_id in tqdm(rejected_id_set, desc = 'Updating status of rejected alerts...'):
                    self.db.update_row(tbl_name = self.config.db_name['lsst_fink'], update_key = 'status', update_value = 'rejected', id_key = 'diaSourceId', id_value = rejected_id) ########## VARIABLE COLUMN NAME
                
                if len(lsst_fink_tbl_filtered) != 0:
                    lsst_fink_tbl_filtered.sort('diaSource_midpointMjdTai') ########## VARIABLE COLUMN NAME
                    lsst_fink_tbl_filtered = unique(lsst_fink_tbl_filtered, keys = 'diaObject_diaObjectId') ########## VARIABLE COLUMN NAME
                    
                    if verbose:
                        lsst_fink_iterator = tqdm(lsst_fink_tbl_filtered, desc = 'Checking alerts...')
                        print(f"[LSST, FINK] {len(lsst_fink_tbl_filtered)} new alerts are found!")
                    else:
                        lsst_fink_iterator = lsst_fink_tbl_filtered
                    
                    for row in lsst_fink_iterator:
                        alert_instance = Alert(
                            ra = float(row['diaObject_ra']), ########## VARIABLE COLUMN NAME
                            dec = float(row['diaObject_dec']), ########## VARIABLE COLUMN NAME
                            objname = str(row['diaObject_diaObjectId']), ########## VARIABLE COLUMN NAME
                            alert_time = Time(row['diaSource_midpointMjdTai'], format='mjd'), ########## VARIABLE COLUMN NAME
                            source = 'FINK_LSST',
                            **row
                        )
                        
                        alert_already_in_db = self.get_alert_db_info(ra = alert_instance.ra, 
                                                                     dec = alert_instance.dec,
                                                                     alert_time=alert_instance.alert_time)
                        if alert_already_in_db is not None:
                            self.db.update_row(tbl_name = self.config.db_name['lsst_fink'], update_key = 'status', update_value = 'passed', id_key = 'diaObject_diaObjectId', id_value = str(row['diaObject_diaObjectId'])) ########## VARIABLE COLUMN NAME
                            continue
                            
                        input_count += 1
                        if verbose:
                            print("NEW ALERT FOUND!")
                        self.insert_alert_db_info(alert_instance)
                        self.db.update_row(tbl_name = self.config.db_name['lsst_fink'], update_key = 'status', update_value = 'passed', id_key = 'diaObject_diaObjectId', id_value = str(row['diaObject_diaObjectId'])) ########## VARIABLE COLUMN NAME
                print(f"[LSST, FINK] {input_count}/{len(lsst_fink_tbl)} new alerts")

        # TNS Alert broker. (Receive -> Update DB -> Trigger)
        if self.config.receive_tns:
            tns_tbl = self.db.get_data(tbl_name = self.config.db_name['tns'], select_key = '*', where_key = 'status', where_value = 'new')
            if len(tns_tbl) == 0:
                print('No TNS alerts found!')
            else:
                input_count = 0
                tns_tbl_filtered = self.filter['tns'].apply(tns_tbl, verbose = False)
                filtered_id_set = set(str(row['Name']) for row in tns_tbl_filtered)
                rejected_id_set = set(str(row['Name']) for row in tns_tbl) - filtered_id_set
                
                for rejected_id in rejected_id_set:
                    self.db.update_row(tbl_name = self.config.db_name['tns'], update_key = 'status', update_value = 'rejected', id_key = 'Name', id_value = rejected_id)
                
                if len(tns_tbl_filtered) != 0:
                    if verbose:
                        tns_iterator = tqdm(tns_tbl_filtered, desc = 'Checking alerts...')
                        print(f"[TNS] {len(tns_tbl_filtered)} new alerts are found!")
                    else:
                        tns_iterator = tns_tbl_filtered
                    for row in tns_iterator:
                        alert_instance = Alert(
                            ra = float(row['ra_deg']),
                            dec = float(row['dec_deg']),
                            objname = str(row['Name']),
                            alert_time = Time(row['Discovery Date (UT)'], format='iso'),
                            source = f'TNS_{row["Reporting Group/s"]}',
                            **row
                        )
                        
                        alert_already_in_db = self.get_alert_db_info(ra = alert_instance.ra, 
                                                                     dec = alert_instance.dec,
                                                                     alert_time=alert_instance.alert_time)
                        if alert_already_in_db is not None:
                            self.db.update_row(tbl_name = self.config.db_name['tns'], update_key = 'status', update_value = 'passed', id_key = 'Name', id_value = str(row['Name']))
                            continue
                                                
                        input_count +=1
                        if verbose:
                            print("NEW ALERT FOUND!")
                        self.insert_alert_db_info(alert_instance)
                        self.db.update_row(tbl_name = self.config.db_name['tns'], update_key = 'status', update_value = 'passed', id_key = 'Name', id_value = str(row['Name']))
                print(f"[TNS] {input_count}/{len(tns_tbl)} new alerts")
            
    def trigger_alerts(self, verbose: bool = True):
        db_data = self.db.get_data(tbl_name = 'transient_info', select_key = '*')
        if len(db_data) == 0:
            print("No alerts to trigger!")
            return
        
        db_data_untriggered = db_data[(db_data['is_triggered'] == False) &
                                      (db_data['create_time'] > Time.now() - self.config.trigger_constraints["since_days"]*u.day)&
                                      (db_data['priority'] < 99)]
        
        if len(db_data_untriggered) == 0:
            print("No alerts to trigger!")
            return
            
        alert_instances_to_check = []
        for row in db_data_untriggered:
            input_dict = dict(row)
            # If isinstance(Time), convert to isot
            input_dict = {key: Time(value).isot if isinstance(value, (Time, datetime.datetime)) else value for key, value in input_dict.items()}
            # If not float, int, str, bool, None, remove
            input_dict = {key: value for key, value in input_dict.items() if (np.isscalar(value)) or (value is None)}
            alert_instance = Alert(**input_dict)
            alert_instances_to_check.append(alert_instance)
        
        while alert_instances_to_check:
            alert_instance = alert_instances_to_check.pop(0)
            checked = self.check_alert(alert_instance, verbose=verbose)
            if checked is None:
                continue
            self.trigger_alert(checked)

            now = time.time()
            self.trigger_history.append(now)

            # Count how many triggers in last 10 minutes
            recent_triggers = [t for t in self.trigger_history if now - t <= 600]

            if len(recent_triggers) > 2:
                if verbose:
                    print("⚠️ More than 2 triggers in 10 minutes → waiting 1 day")
                time.sleep(86400)  # wait 1 day
            else:
                if verbose:
                    print("Sleeping 10 minutes before next trigger")
                time.sleep(600)   # Wait 10 minutes

    def trigger_alert(self, alert_instance: Alert):
        # Input ToO status 
        # Send ToO observation mail (Trigger ToO observation)
        self.send_mail(alert_instance)
        alert_instance.is_triggered = True
        alert_instance.trigger_time = Time.now()
        alert_instance.update_time = Time.now()
        self.insert_alert_db_status(alert_instance)
        self.update_alert_db_info(alert_instance)
        
    def check_alert(self, alert_instance: Alert, verbose: bool = True):
        """
        Check the alert and return the alert instance if it is valid.
        Trigger criteria:
        - Visibility (now/tonight)
        - Host galaxy association (double check)
        - Detection
        - Magnitude range
        """

        alert_filter = None
        alert_querier = None
        alert_source = alert_instance.source
        if 'TNS' in alert_source:
            alert_filter = self.filter['tns']
            alert_querier = self.querier['tns']
        elif alert_source  == 'FINK_ZTF':
            alert_filter = self.filter['ztf']['fink']
            alert_querier = self.querier['ztf']['fink']
        elif alert_source  == 'FINK_LSST':
            alert_filter = self.filter['lsst']['fink']
            alert_querier = self.querier['lsst']['fink']
        elif alert_source  == 'ALERCE_ZTF':
            alert_filter = self.filter['ztf']['alerce']
            alert_querier = self.querier['ztf']['alerce']
        elif alert_source  == 'ALERCE_LSST':
            alert_filter = self.filter['lsst']['alerce']
            alert_querier = self.querier['lsst']['alerce']
        else:
            alert_instance.reject_note = "Unsupported alert source"
            alert_instance.priority = 99
            alert_instance.priority_note = "Unsupported alert source"
            self.update_alert_db_info(alert_instance)
            self._print_progress(
                title = alert_instance.objname, 
                progress_dict = progress, 
                stop="Unsupported alert source", 
                verbose = verbose)
            return None
        now = Time.now()
        mjd_now = now.mjd

        progress = {
            "Host galaxy association": False,
            "Visibility (now/tonight)": False,
            "Detection from the alert source": False,
            "Magnitude/Phase Check": False,
            'Exposure time calculation': False
        }

        # 1. Host galaxy association       
        """
        TO REJECT FOLLOWING CASES:
        1. No host galaxy found
        2. Host galaxy distance is not found
        3. Host galaxy distance is out or range
        """     
        hostgalaxy_association_rejected = False
        if self.config.trigger_constraints['hostgalaxy_association']:
            if alert_instance.hostgalaxy_name is None:
                alert_instance.match_host(hostgalaxy_catalog = self.hostgalaxycatalog, 
                                        search_radius_arcsec = alert_filter.config.hostgalaxy_constraints['search_radius_arcsec'], 
                                        max_dell = alert_filter.config.hostgalaxy_constraints['max_dell'], 
                                        return_all = False,
                                        plot = False,
                                        save_path = None)
            if alert_instance.hostgalaxy_dist is None:
                reject_note = "Host galaxy distance is not found"
                hostgalaxy_association_rejected = True
            else:
                if alert_instance.hostgalaxy_dist > self.config.trigger_constraints['hostgalaxy_association_kwargs']['distance_upper']:
                    reject_note = "Host galaxy distance is out of range"
                    hostgalaxy_association_rejected = True
                if alert_instance.hostgalaxy_dist < self.config.trigger_constraints['hostgalaxy_association_kwargs']['distance_lower']:
                    hostgalaxy_association_rejected = True
                    reject_note = "Host galaxy distance is out of range"
            if hostgalaxy_association_rejected:
                alert_instance.reject_note = reject_note
                alert_instance.priority = 99
                alert_instance.priority_note = reject_note
                self.update_alert_db_info(alert_instance)
                self._print_progress(title = alert_instance.objname, progress_dict = progress, stop=reject_note, verbose = verbose)
                return None

        progress["Host galaxy association"] = True

        # 1. Visibility
        """
        TO REJECT FOLLOWING CASES:
        1. Alert that is not observable tonight
        """
        if not alert_instance.is_observable_tonight(now):
            alert_instance.reject_note = "Not observable tonight"
            self.update_alert_db_info(alert_instance)
            self._print_progress(
                title = alert_instance.objname,
                progress_dict = progress, 
                stop="Not observable tonight",
                verbose = verbose)
            return None
        progress["Visibility (now/tonight)"] = True
        savepath_visibility = Path(self.config.save_dir) / alert_instance.objname / f'{alert_instance.objname}_visibility.png' if self.config.save_visibility else None
        if (self.config.show_visibility) or (self.config.save_visibility):
            alert_instance.plot_visibility(utctime = now, 
                                           show = self.config.show_visibility, 
                                           save_path = savepath_visibility)

        # 3. Detection
        """
        TO REJECT FOLLOWING CASES:
        1. Alert source (TNS, FINK, ALERCE) changed the alert as null
        2. Alert source (TNS, FINK, ALERCE) has only non-detections
        3. Alert that has no recent detection
        4. Alert that was detected long time ago (since_days)
        """

        detection_rejected = False
        trend = 'rising'
        
        history_days = float(self.config.trigger_constraints["since_days"])
        magnitude = None
        filter_ = None
        mjd = None
        source = alert_instance.source.split('_')[0]
        facility = alert_instance.source.split('_')[1]
        detection_tbl = alert_querier.query_detections(alert_instance.objname, verbose=verbose)

        if len(detection_tbl) == 0:
            rejection_note = "No detections found"
            detection_rejected = True
        else:
            # --------------------------------------------------
            # Normalize columns by alert source
            # --------------------------------------------------
            if 'TNS' in alert_source:
                if 'mjd' not in detection_tbl.colnames:
                    detection_tbl['mjd'] = Time(detection_tbl['jd'], format='jd').mjd
                detection_tbl['mag'] = np.array(detection_tbl['mag'], dtype=float)
                detection_tbl['filter_'] = [
                    str(f).split('-')[0] if f is not None else None
                    for f in detection_tbl['filter']
                ]

            elif alert_source == 'ALERCE_ZTF':
                detection_tbl['mag'] = np.array(detection_tbl['magpsf'], dtype=float)
                detection_tbl['filter_'] = [
                    ZTF_FILTER_MAP.get(int(fid), None) if fid is not None else None
                    for fid in detection_tbl['fid']
                ]
                source = 'ALERCE_ZTF'
                facility = 'ZTF'

            elif alert_source == 'ALERCE_LSST':
                detection_tbl['mag'] = -2.5 * np.log10(np.array(detection_tbl['psfFlux'], dtype=float)) + 31.4
                detection_tbl['filter_'] = np.array(detection_tbl['band_name']).astype(str)

            elif alert_source == 'FINK_ZTF':
                if 'mjd' not in detection_tbl.colnames:
                    detection_tbl['mjd'] = Time(detection_tbl['i:jdstarthist'], format='jd').mjd
                detection_tbl['mag'] = np.array(detection_tbl['i:magpsf'], dtype=float)
                detection_tbl['filter_'] = [
                    ZTF_FILTER_MAP.get(int(fid), None) if fid is not None else None
                    for fid in detection_tbl['i:fid']
                ]

            elif alert_source == 'FINK_LSST':
                if 'mjd' not in detection_tbl.colnames:
                    detection_tbl['mjd'] = np.array(detection_tbl['r:midpointMjdTai'], dtype=float)
                detection_tbl['mag'] = -2.5 * np.log10(np.array(detection_tbl['r:psfFlux'], dtype=float)) + 31.4
                detection_tbl['filter_'] = np.array(detection_tbl['r:band']).astype(str)

            # --------------------------------------------------
            # Common validation
            # --------------------------------------------------
            if not detection_rejected:
                valid_mask = (
                    np.isfinite(np.array(detection_tbl['mag'], dtype=float)) &
                    np.isfinite(np.array(detection_tbl['mjd'], dtype=float))
                )
                detection_tbl = detection_tbl[valid_mask]

                if len(detection_tbl) == 0:
                    rejection_note = f"No valid detections found for {alert_instance.objname}"
                    detection_rejected = True

            # --------------------------------------------------
            # Common summary and rejection logic
            # --------------------------------------------------
            if not detection_rejected:
                detection_tbl.sort('mjd', reverse=True)

                latest_row = detection_tbl[0]
                magnitude = float(latest_row['mag'])
                filter_ = latest_row['filter_'] if 'filter_' in detection_tbl.colnames else None
                mjd = float(latest_row['mjd'])

                if mjd < (mjd_now - history_days):
                    rejection_note = f"No recent detections found for {alert_instance.objname}"
                    detection_rejected = True

                oldest_mjd = np.nanmin(np.array(detection_tbl['mjd'], dtype=float))
                if oldest_mjd < (mjd_now - history_days):
                    rejection_note = f"Historical detections older than {history_days:.0f} days found for {alert_instance.objname}"
                    detection_rejected = True

            # --------------------------------------------------
            # Trend
            # --------------------------------------------------
            if not detection_rejected:
                if 'trend' in detection_tbl.colnames:
                    val = detection_tbl['trend'][0]
                    if val is not None and str(val).strip() != '':
                        trend = str(val).lower()
                else:
                    if filter_ is not None:
                        same_filter_tbl = detection_tbl[detection_tbl['filter_'] == filter_]
                    else:
                        same_filter_tbl = detection_tbl

                    same_filter_tbl = same_filter_tbl[
                        np.isfinite(np.array(same_filter_tbl['mag'], dtype=float))
                    ]
                    same_filter_tbl.sort('mjd', reverse=True)

                    if len(same_filter_tbl) >= 2:
                        mag_recent = float(same_filter_tbl['mag'][0])
                        mag_prev = float(same_filter_tbl['mag'][1])

                        if mag_recent < mag_prev:
                            trend = 'rising'
                        elif mag_recent > mag_prev:
                            trend = 'falling'

        if detection_rejected:
            alert_instance.reject_note = rejection_note
            alert_instance.priority = 99
            alert_instance.priority_note = rejection_note
            self.update_alert_db_info(alert_instance)
            self._print_progress(
                title = alert_instance.objname, 
                progress_dict = progress, 
                stop=rejection_note, 
                verbose = verbose)
            return None
        progress["Detection from the alert source"] = True

        # 4. Apply magnitude range
        """
        TO REJECT FOLLOWING CASES:
        Reject the alert with the following criteria
        1. Expected magnitude is not in the range of the magnitude range
        2. Expected absolute magnitude is not in the range of the absolute magnitude range
        3. Expected phase is out of range
        Magnitude cut will be applied differently depending on the priority
        - Priority
            Priority definition
            - Priority 0: Infant-phase SN candidate -> Priority 0
                trend == "rising"
                expected_absolute_magnitude > -14
                In this case, if too faint, wait until becoming bright.

            - Priority 1: Early-excess candidate -> Priority 0
                trend == "rising"
                -16 < expected_absolute_magnitude <= -14

            - Priority 2: Nearby bright SN around the peak -> Priority 5
                host_distance_mpc <= 50
                -20 <= expected_absolute_magnitude <= -17

            - Priority 3: Apparently bright SN at larger distance -> Priority 50
                host_distance_mpc > 50
                -20 <= expected_absolute_magnitude <= -17

            - Priority 4: Others -> Priority 50
        """

        # Update observation information
        magnitude_rejected = False
        distance_modulus = 5 * np.log10(alert_instance.hostgalaxy_dist) + 25
        absolute_magnitude = magnitude - distance_modulus
        facility = alert_source.split('_')[0]
        source = alert_source.split('_')[1]
        update_dict = dict()
        update_dict['last_observed_time'] = Time(mjd, format = 'mjd').isot
        update_dict['last_observed_mag'] = float(magnitude)
        update_dict['last_observed_absmag'] = float(absolute_magnitude)
        update_dict['last_observed_filter'] = str(filter_)
        update_dict['last_observed_source'] = str(source)
        update_dict['last_observed_facility'] = str(facility)
        update_dict['last_observed_nhist'] = len(detection_tbl)
        update_dict['last_expected_time'] = Time(mjd_now, format = 'mjd').isot
        update_dict['last_expected_trend'] = str(trend)
        for key, val in update_dict.items():
            setattr(alert_instance, key, val)

        if self.config.trigger_constraints['template_expectation']:

            # Estimated magnitude
            template = Template(transient_type = self.config.trigger_constraints['template_kwargs']['transient_type'],
                                observed_mjd = mjd, 
                                observed_absmag = absolute_magnitude, 
                                observed_mag = magnitude,
                                trend = trend, 
                                transient_source = self.config.trigger_constraints['template_kwargs']['transient_source'],
                                template_dir = self.config.trigger_constraints['template_kwargs']['template_dir'])
            estimate = template.estimate_from_mjd(now.mjd)
            estimate_dict = estimate.__dict__

            for key, val in estimate_dict.items():
                setattr(alert_instance, key, np.atleast_1d(val)[0])

            expected_magnitude = estimate_dict['last_expected_mag']
            expected_absolute_magnitude = estimate_dict['last_expected_absmag']
            expected_phase = estimate_dict['last_expected_phase']
        else:
            expected_magnitude = magnitude
            expected_absolute_magnitude = absolute_magnitude
            expected_phase = 0
            
        host_distance_mpc = alert_instance.hostgalaxy_dist
        # ------------------------------------------------------------------
        # Expected values
        # ------------------------------------------------------------------
        is_rising = (str(trend).lower() == 'rising')
        is_nearby = (
            host_distance_mpc is not None and
            np.isfinite(host_distance_mpc) and
            host_distance_mpc <= 50
        )

        priority = 50
        priority_note = "Other acceptable target"

        # Priority 0: Infant-phase SN candidate
        if is_rising and (expected_absolute_magnitude is not None) and np.isfinite(expected_absolute_magnitude):
            if expected_absolute_magnitude > -14:
                priority = 0
                priority_note = "Infant-phase SN candidate"

            elif -16 < expected_absolute_magnitude <= -14:
                priority = 1
                priority_note = "Early-excess candidate"

        # Priority 5: Nearby bright SN around peak
        if priority == 50:
            if (
                is_nearby and
                (expected_absolute_magnitude is not None) and
                np.isfinite(expected_absolute_magnitude) and
                (-20 <= expected_absolute_magnitude <= -16)
            ):
                priority = 5
                priority_note = "Nearby bright SN around peak"

        # Priority 10: Apparently bright SN at larger distance
        if priority == 50:
            if (
                (not is_nearby) and
                (expected_absolute_magnitude is not None) and
                np.isfinite(expected_absolute_magnitude) and
                (-20 <= expected_absolute_magnitude <= -17)
            ):
                priority = 10
                priority_note = "Apparently bright SN at larger distance"

        alert_instance.priority = priority
        alert_instance.priority_note = priority_note

        tc = self.config.trigger_constraints

        # Use defaults if some keys are not defined in config
        mag_min = tc['mag']['lower']
        mag_max = tc['mag']['upper']

        absmag_min = tc['absmag']['lower']
        absmag_max = tc['absmag']['upper']

        phase_min = tc['phase']['lower']
        phase_max = tc['phase']['upper']

        magnitude_rejected = False
        rejection_note = None

        # 1. apparent magnitude cut
        if (expected_magnitude is None) or (not np.isfinite(expected_magnitude)):
            if priority not in [0, 1]:
                magnitude_rejected = True
                rejection_note = "Expected magnitude is not finite"

        elif not (mag_min <= expected_magnitude <= mag_max):
            if priority not in [0, 1]:
                magnitude_rejected = True
                rejection_note = (
                    f"Expected magnitude {expected_magnitude:.2f} is out of range "
                    f"[{mag_min:.2f}, {mag_max:.2f}]"
                )

        # 2. absolute magnitude cut
        if not magnitude_rejected:
            if (expected_absolute_magnitude is None) or (not np.isfinite(expected_absolute_magnitude)):
                magnitude_rejected = True
                rejection_note = "Expected absolute magnitude is not finite"

            elif not (absmag_min <= expected_absolute_magnitude <= absmag_max):
                magnitude_rejected = True
                rejection_note = (
                    f"Expected absolute magnitude {expected_absolute_magnitude:.2f} is out of range "
                    f"[{absmag_min:.2f}, {absmag_max:.2f}]"
                )

        # 3. phase cut
        if not magnitude_rejected:
            if (expected_phase is None) or (not np.isfinite(expected_phase)):
                magnitude_rejected = True
                rejection_note = "Expected phase is not finite"

            elif not (phase_min <= expected_phase <= phase_max):
                magnitude_rejected = True
                rejection_note = (
                    f"Expected phase {expected_phase:.2f} is out of range "
                    f"[{phase_min:.2f}, {phase_max:.2f}]"
                )
                
        if magnitude_rejected:
            alert_instance.rejection_note = rejection_note
            self.update_alert_db_info(alert_instance)
            self._print_progress(
                title=alert_instance.objname,
                progress_dict=progress,
                stop=rejection_note,
                verbose=verbose
            )
            return None
        progress["Magnitude/Phase Check"] = True

        # 5. Exposure time calculation
        """
        TO REJECT FOLLOWING CASES:
        1. Exposure time is too long
        2. Exposure time is too short
        3. Exposure time is not finite
        4. Exposure time is not in the range of the exposure time range
        """

        if self.config.trigger_constraints["exposure_calculation"]:
            exptime_dict = self.exposurecalculator.calculate_exptime(filter = 'all', 
                                                                     magnitude = expected_magnitude, 
                                                                     snr = self.config.trigger_constraints["exposure_calculator_kwargs"]["snr"], 
                                                                     obsdate = now, 
                                                                     ra = alert_instance.ra, 
                                                                     dec = alert_instance.dec, 
                                                                     verbose = False)
            # 만약 Priority 0, 1 타겟이면 Spec - COlor - Deep 시도 
            exptime_rejected = False            
            exptime_upper = self.config.trigger_constraints['exptime']['upper_limit'][f'priority{priority}']
            exptime_lower = self.config.trigger_constraints['exptime']['lower_limit'][f'priority{priority}']

            if priority in [0, 1]:
                # First, calculate Spec mode exposure time
                spec_ntel = self.config.trigger_constraints['exposure_calculator_kwargs']['ntel_spec']
                spec_filter = self.config.trigger_constraints['exposure_calculator_kwargs']['reference_filter_spec']
                expected_exptime = exptime_dict[spec_filter]['t_exp']
                expected_exptime_per_tel = expected_exptime / spec_ntel
                obsmode = 'Spec'

                # If Spec mode exposure time is too long, try Color mode
                if expected_exptime_per_tel > exptime_upper:
                    color_ntel = self.config.trigger_constraints['exposure_calculator_kwargs']['ntel_color']
                    color_filter = self.config.trigger_constraints['exposure_calculator_kwargs']['reference_filter_color']
                    expected_exptime = exptime_dict[color_filter]['t_exp']
                    expected_exptime_per_tel = expected_exptime / color_ntel
                    obsmode = 'Color'

                    # If Color mode exposure time is too long, try Deep mode
                    if expected_exptime_per_tel > exptime_upper:
                        deep_ntel = self.config.trigger_constraints['exposure_calculator_kwargs']['ntel_deep']
                        deep_filter = self.config.trigger_constraints['exposure_calculator_kwargs']['reference_filter_deep']
                        expected_exptime = exptime_dict[deep_filter]['t_exp']
                        expected_exptime_per_tel = expected_exptime / deep_ntel
                        obsmode = 'Deep'

                        if expected_exptime_per_tel > exptime_upper:
                            exptime_rejected = True

            elif priority in [5, 10, 50]:
                # Calculate Spec mode exposure time
                spec_ntel = self.config.trigger_constraints['exposure_calculator_kwargs']['ntel_spec']
                spec_filter = self.config.trigger_constraints['exposure_calculator_kwargs']['reference_filter_spec']
                expected_exptime = exptime_dict[spec_filter]['t_exp']
                expected_exptime_per_tel = expected_exptime / spec_ntel
                obsmode = 'Spec'

                if expected_exptime_per_tel > exptime_upper:
                    exptime_rejected = True
            else:
                exptime_rejected = True

            alert_instance.obsmode = obsmode                        
            if expected_exptime < exptime_lower:
                alert_instance.count = int(np.ceil(exptime_lower / alert_instance.exptime))
                expected_exptime = exptime_lower
            else:
                alert_instance.count = int(np.ceil(expected_exptime / alert_instance.exptime))
            expected_snr = self.exposurecalculator.calculate_snr(filter = 'all', 
                                                                 magnitude = expected_magnitude, 
                                                                 exptime = expected_exptime,
                                                                 obsdate = now, 
                                                                 ra = alert_instance.ra, 
                                                                 dec = alert_instance.dec, 
                                                                 verbose = False)
            alert_instance.expected_observation = dict()
            alert_instance.expected_observation['snr'] = {filt: np.round(value['snr'], 1) for filt, value in expected_snr.items()}
            alert_instance.expected_observation['moonphase'] = np.round(list(expected_snr.values())[0]['moon_phase'], 1)
            alert_instance.expected_observation['moonseparation'] = np.round(list(expected_snr.values())[0]['moon_separation'], 1)

            if exptime_rejected:
                rejection_note = f'Estimated exposure time is too long: {alert_instance.exptime:.0f} seconds'
                alert_instance.rejection_note = rejection_note
                self.update_alert_db_info(alert_instance)
                self._print_progress(
                    title=alert_instance.objname,
                    progress_dict=progress,
                    stop=rejection_note,
                    verbose=verbose
                )
                return None
        progress["Exposure time calculation"] = True
        self.update_alert_db_info(alert_instance)
        self._print_progress(title = alert_instance.objname, progress_dict = progress, stop=None, verbose = verbose)
        return alert_instance
                
    def send_mail(self, alert_instance):
        """
        Send email notification for a new alert_instance,
        strictly following the ToO email format.
        """

        subject = f"[Automated] 7DT ToO Observation Request: {alert_instance.objname}"

        email_body = f"""
        ================================
        New ToO Request Submitted
        ================================

        **Observation Information**
        ----------------------
        - Requester: {self.config.mail_requester}
        - Target Name: {alert_instance.objname}
        - Right Ascension (R.A.): {alert_instance.ra:.5f}
        - Declination (Dec.): {alert_instance.dec:.5f}
        - Total Exposure Time (seconds): {alert_instance.exptime * alert_instance.count:.1f}
        - Single Exposure Time (seconds): {alert_instance.exptime:.1f}
        - # of images: {alert_instance.count}
        - Obsmode: {alert_instance.obsmode}

        **Detailed Settings**
        --------------------
        - Is_ToO: {alert_instance.is_ToO}
        - Is_rapid_ToO: {alert_instance.is_rapid_ToO}
        - Cadence: 1
        - Number of Observations: 1
        - ToO start time: None
        - ToO end time: None
        - Priority: {alert_instance.priority}
        - Gain: {alert_instance.gain}
        - Radius (arcsec): {alert_instance.radius * 3600}
        - Binning: {alert_instance.binning}
        - Observation Start Time: None
        - Comments: Triggered automatically with 7DT SN ToO Project [Host: {alert_instance.hostgalaxy_name} Distance: {alert_instance.hostgalaxy_dist:.2f} Mpc]
        
        ADDITIONAL ALERT INFORMATION
        | Expected Absolute Magnitude: {alert_instance.last_expected_absmag:.2f}
        | Expected Magnitude: {alert_instance.last_expected_mag:.2f}
        | Last Observed Magnitude: {alert_instance.last_observed_mag:.2f}
        | Last Observed Absolute Magnitude: {alert_instance.last_observed_absmag:.2f}
        | Last Observed Time: {alert_instance.last_observed_time}
        | Last Observed Facility: {alert_instance.last_observed_facility}
        """

        if 'expected_observation' in alert_instance.__dict__:
            snr_dict = alert_instance.expected_observation.get('snr', {})
            snr_text = ", ".join(
                f"{filt}({snr:.1f})" for filt, snr in snr_dict.items()
            ) if snr_dict else "N/A"

            moonphase = alert_instance.expected_observation.get('moonphase')
            moonsep = alert_instance.expected_observation.get('moonseparation')

            moonphase_text = f"{moonphase:.1%}" if isinstance(moonphase, (int, float)) else "N/A"
            moonsep_text = f"{moonsep:.1f} deg" if isinstance(moonsep, (int, float)) else "N/A"

            email_body += f"""| Expected SNR: {snr_text}
        | Moon phase: {moonphase_text}
        | Moon separation: {moonsep_text}"""

        try:
            self.mail.send_mail(to_users = self.config.mail_recipients, subject = subject, body = email_body)
            return True
        except Exception as e:
            print(f"❌ Failed to send email for {alert_instance.objname}: {e}")
            return False
        
    def insert_alert_db_status(self, alert_instance: Alert):
        tile_id = alert_instance.tile_id
        tile_ra = alert_instance.tile_ra
        tile_dec = alert_instance.tile_dec
        # When the alert is not covered by 7DS tiles
        if tile_id is None:
            alert_instance.status_id = str(uuid4())
            input_dict = alert_instance.__dict__
            # If isinstance(Time), convert to isot
            input_dict = {key: Time(value).isot if isinstance(value, (Time, datetime.datetime)) else value for key, value in input_dict.items()}
            # If not float, int, str, bool, None, remove
            input_dict = {key: value for key, value in input_dict.items() if (np.isscalar(value)) or (value is None)}
            result = self.db.insert_rows(tbl_name = 'transient_status', data = Table(names = input_dict.keys(), rows = [input_dict.values()]))
            
        if len(tile_id) == 1:
            alert_instance.tile_id = tile_id[0]
            alert_instance.tile_ra = tile_ra[0]
            alert_instance.tile_dec = tile_dec[0]
            
            input_dict = alert_instance.__dict__
            # If isinstance(Time), convert to isot
            input_dict = {key: Time(value).isot if isinstance(value, (Time, datetime.datetime)) else value for key, value in input_dict.items()}
            # If not float, int, str, bool, None, remove
            input_dict = {key: value for key, value in input_dict.items() if (np.isscalar(value)) or (value is None)}
            result = self.db.insert_rows(tbl_name = 'transient_status', data = Table(names = input_dict.keys(), rows = [input_dict.values()]))

        else:
            for tile_id, tile_ra, tile_dec in zip(tile_id, tile_ra, tile_dec):
                alert_instance.tile_id = tile_id
                alert_instance.tile_ra = tile_ra
                alert_instance.tile_dec = tile_dec
                alert_instance.status_id = str(uuid4())
                input_dict = alert_instance.__dict__
                # If isinstance(Time), convert to isot
                input_dict = {key: Time(value).isot if isinstance(value, (Time, datetime.datetime)) else value for key, value in input_dict.items()}
                # If not float, int, str, bool, None, remove
                input_dict = {key: value for key, value in input_dict.items() if (np.isscalar(value)) or (value is None)}
                result = self.db.insert_rows(tbl_name = 'transient_status', data = Table(names = input_dict.keys(), rows = [input_dict.values()]))
        return result


    def _to_python_scalar(self, value):
        # None
        if value is None:
            return None

        # Time / datetime -> isot string
        if isinstance(value, (Time, datetime.datetime)):
            return Time(value).isot

        # NumPy scalar -> native Python scalar
        if isinstance(value, np.generic):
            value = value.item()

        # Convert non-finite floats to None
        if isinstance(value, float):
            if not math.isfinite(value):
                return None
            return value

        # Native accepted scalar types
        if isinstance(value, (int, str, bool)):
            return value

        # Plain Python containers are not stored here
        return None


    def _sanitize_dict_for_db(self, input_dict):
        clean = {}
        for key, value in input_dict.items():
            converted = self._to_python_scalar(value)
            clean[key] = converted
        return clean

    def update_alert_db_info(self, alert_instance: Alert):
        input_dict = self._sanitize_dict_for_db(alert_instance.__dict__)

        # Usually do not update primary key itself
        input_dict.pop("alert_id", None)

        keys = list(input_dict.keys())
        values = list(input_dict.values())

        result = self.db.update_row(
            tbl_name='transient_info',
            update_key=keys,
            update_value=values,
            id_key='alert_id',
            id_value=str(alert_instance.alert_id),
        )
        return result
    
    def insert_alert_db_info(self, alert_instance: Alert):
        input_dict = self._sanitize_dict_for_db(alert_instance.__dict__)

        result = self.db.insert_rows(
            tbl_name='transient_info',
            data=Table(names=list(input_dict.keys()), rows=[list(input_dict.values())])
        )
        return result
    
    def get_alert_db_info(self, 
                         idx: int = None,
                         alert_id: str = None,
                         objname: str = None,
                         ra: float = None,
                         dec: float = None,
                         alert_time: Time = None):
        
        data = self.db.get_data(tbl_name = 'transient_info', select_key = '*')
        if len(data) == 0:
            return None
        
        target_row = data
        if idx is not None:
            target_row = target_row[idx]
        if len(target_row) == 0:
            return None
        if alert_id is not None:
            target_row = target_row[target_row['alert_id'] == alert_id]
        if len(target_row) == 0:
            return None
        if objname is not None:
            target_row = target_row[target_row['objname'] == objname]
        if len(target_row) == 0:
            return None
        if ra is not None and dec is not None:
            target_coord = SkyCoord(ra = ra * u.deg, dec = dec * u.deg)
            data_coord = SkyCoord(ra = target_row['ra'] * u.deg, dec = target_row['dec'] * u.deg)
            sep = target_coord.separation(data_coord)
            close_idx = np.where(sep < 30 * u.arcsec)[0]
            target_row = target_row[close_idx]
        if len(target_row) == 0:
            return None
        if alert_time is not None:
            alert_time = Time(alert_time)
            db_alert_time = Time(target_row['alert_time'])
            close_idx = np.where(np.abs(db_alert_time - alert_time) < 15)[0]
            target_row = target_row[close_idx]
        if len(target_row) == 0:
            return None

        if len(target_row) > 1:
            print(f"Multiple targets found. Returning the first one")
        target_row = target_row[0]
            
        input_dict = dict(target_row)
        # If isinstance(Time), convert to isot
        input_dict = {key: Time(value).isot if isinstance(value, (Time, datetime.datetime)) else value for key, value in input_dict.items()}
        # If not float, int, str, bool, None, remove
        input_dict = {key: value for key, value in input_dict.items() if (np.isscalar(value)) or (value is None)}
        
        alert_instance = Alert(**input_dict)

        return alert_instance
    
    def _print_progress(self, title, progress_dict, stop=None, verbose: bool = True):
        if verbose:
            print(f"\n--- ALERT {title} Progress ---")
            for step, done in progress_dict.items():
                checkbox = "✅" if done else "⬜"
                print(f"{checkbox} {step}")
            if stop:
                print(f"⚠️ Stopped: {stop}")
            print("-----------------------------\n")


    
#%%
if __name__ == "__main__":
    self = AlertTrigger()
    while True:
        now = Time.now()    
        print("Receiving alerts... at", now.isot)
        self.receive_alerts(utctime = now, verbose= False)
        self.trigger_alerts()
        time.sleep(30)

# %%