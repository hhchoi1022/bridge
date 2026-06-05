#%%
from ezphot.utils import DataBrowser
from ezphot.methods import *
from ezphot.imageobjects import *
from ezphot.dataobjects import Catalog, CatalogSet, LightCurve, PhotometricSpectrum
from bridge.configuration import Configuration
from bridge.connector import *
from bridge.objects import Alert
import time
from astropy.time import Time
from astropy.table import Table
import datetime
import numpy as np
from numpy.ma import is_masked
from tqdm import tqdm
from pathlib import Path
from astropy.table import vstack
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u

from ezphot.imageobjects import ImageSet
from bridge.utils.hostgalaxycatalog import HostGalaxyCatalog
from bridge.alertmonitor.alertprocessor import AlertProcessor
from bridge.alertmonitor.alertclassifier import AlertClassifier
import matplotlib
matplotlib.use("Agg")
import os
os.nice(10)
#%%
class AlertChecker:
    def __init__(self):
        self.config = Configuration(config_filenames=['alertchecker.config'])
        self.db_connector = SQLConnector()
        self.gwportal_connector = GWPortalConnector()
        self.ezphot_connector = DataBrowser('scidata')
        self.classifier = AlertClassifier()
        self.hostgalaxycatalog = HostGalaxyCatalog()
    
    def check_alerts(self): 
        process_start_time = time.time()
        db_data = self.db_connector.get_data(tbl_name = 'transient_status', select_key = '*')
        # Update alert information first
        for row in db_data:
            input_dict = dict(row)
            # If isinstance(Time), convert to isot
            input_dict = {key: Time(value).isot if isinstance(value, (Time, datetime.datetime)) else value for key, value in input_dict.items()}
            # If not float, int, str, bool, None, remove
            input_dict = {key: value for key, value in input_dict.items() if (np.isscalar(value)) or (value is None)}
            alert_instance = Alert(**input_dict)
            self.update_alert_db_status(alert_instance)
        db_data_incompleted = db_data[db_data['is_completed'] == False]
        
        alert_instances_to_check = []
        for row in db_data_incompleted:
            input_dict = dict(row)
            # If isinstance(Time), convert to isot
            input_dict = {key: Time(value).isot if isinstance(value, (Time, datetime.datetime)) else value for key, value in input_dict.items()}
            # If not float, int, str, bool, None, remove
            input_dict = {key: value for key, value in input_dict.items() if (np.isscalar(value)) or (value is None)}
            alert_instance = Alert(**input_dict)
            alert_instances_to_check.append(alert_instance)

        for alert_instance in alert_instances_to_check:
            alert_instance = self.check_ezphot_reference(alert_instance)
            if not alert_instance.is_reference_ready:
                alert_instance = self.prepare_reference_img(alert_instance)
            alert_instance = self.check_observation(alert_instance)
            if not alert_instance.is_observed:
                continue
            # # pipeline
            catalog_set_pipeline = self.get_pipeline_photometry(alert_instance)
            if catalog_set_pipeline is not None:
                lc_result = self.draw_lightcurve(alert_instance, catalog_set_pipeline, catalog_type = 'py7dt')
                ps_result = self.draw_photometricspectrum(alert_instance, catalog_set_pipeline, catalog_type = 'py7dt')
                cl_result = self.classify_alert(alert_instance, catalog_set_pipeline, catalog_type = 'py7dt')
            
            # ezphot
            ezphot_result = self.trigger_photometry(alert_instance)

            alert_instance = self.check_observation(alert_instance)
            if alert_instance.is_observed:
                alert_instance.is_processed = True
                alert_instance.is_completed = True
            catalog_set_ezphot_auto = self.get_ezphot_photometry(alert_instance, pattern = 'coadd_scaled*com.fits.cat', catalog_type = 'ezphot')
            if catalog_set_ezphot_auto is not None:
                lc_result = self.draw_lightcurve(alert_instance, catalog_set_ezphot_auto, catalog_type = 'ezphot')
                ps_result = self.draw_photometricspectrum(alert_instance, catalog_set_ezphot_auto, catalog_type = 'ezphot')
                cl_result = self.classify_alert(alert_instance, catalog_set_ezphot_auto, catalog_type = 'ezphot')
            catalog_set_ezphot_forced = self.get_ezphot_photometry(alert_instance, pattern = 'coadd_scaled*com.fits.circ.cat', catalog_type = 'ezphot_forced')
            if catalog_set_ezphot_forced is not None:
                lc_result = self.draw_lightcurve(alert_instance, catalog_set_ezphot_forced, catalog_type = 'ezphot_forced')
                ps_result = self.draw_photometricspectrum(alert_instance, catalog_set_ezphot_forced, catalog_type = 'ezphot_forced')
                cl_result = self.classify_alert(alert_instance, catalog_set_ezphot_forced, catalog_type = 'ezphot_forced')
            catalog_set_ezphot_DIA = self.get_ezphot_photometry(alert_instance, pattern = 'sub*fits.cat', catalog_type = 'ezphot_DIA')
            if catalog_set_ezphot_DIA is not None:
                lc_result = self.draw_lightcurve(alert_instance, catalog_set_ezphot_DIA, catalog_type = 'ezphot_DIA')
                ps_result = self.draw_photometricspectrum(alert_instance, catalog_set_ezphot_DIA, catalog_type = 'ezphot_DIA')
                cl_result = self.classify_alert(alert_instance, catalog_set_ezphot_DIA, catalog_type = 'ezphot_DIA')
                alert_instance.is_subtracted = True
            catalog_set_ezphot_DIA_forced = self.get_ezphot_photometry(alert_instance, pattern = 'sub*fits.circ.cat', catalog_type = 'ezphot_DIA_forced')
            if catalog_set_ezphot_DIA_forced is not None:
                lc_result = self.draw_lightcurve(alert_instance, catalog_set_ezphot_DIA_forced, catalog_type = 'ezphot_DIA_forced')
                ps_result = self.draw_photometricspectrum(alert_instance, catalog_set_ezphot_DIA_forced, catalog_type = 'ezphot_DIA_forced')
                cl_result = self.classify_alert(alert_instance, catalog_set_ezphot_DIA_forced, catalog_type = 'ezphot_DIA_forced')
                alert_instance.is_subtracted = True
            # tractor
            tractor_result = self.trigger_tractor_photometry(alert_instance)
            catalog_set_tractor = self.get_tractor_photometry(alert_instance, pattern = '*com.fits.tract7dtcat', catalog_type = 'tract7dt')
            if catalog_set_tractor is not None:
                lc_result = self.draw_lightcurve(alert_instance, catalog_set_tractor, catalog_type = 'tract7dt')
                ps_result = self.draw_photometricspectrum(alert_instance, catalog_set_tractor, catalog_type = 'tract7dt')
                cl_result = self.classify_alert(alert_instance, catalog_set_tractor, catalog_type = 'tract7dt')
            
            alert_instance = self.check_ezphot_processed(alert_instance)
            self.update_alert_db_status(alert_instance)
        
    def check_observation(self, alert_instance: Alert):
        self.gwportal_connector.query_type = 'raw'
        tbl_observation_objname = self.gwportal_connector.query(
            obs_start_date = Time(alert_instance.trigger_time).datetime.strftime("%Y-%m-%d"),
            object_name = alert_instance.objname,
            verbose = False
        )
        tbl_observation = tbl_observation_objname
        if ('tile' in tbl_observation_objname.colnames) & ('target' in tbl_observation_objname.colnames):
            tbl_observation_objname.remove_columns(['tile', 'target'])
        if alert_instance.tile_id is not None:
            tbl_observation_tile = self.gwportal_connector.query(
                obs_start_date = Time(alert_instance.trigger_time).datetime.strftime("%Y-%m-%d"),
                tile_name = alert_instance.tile_id,
                verbose = False
            )
            if ('tile' in tbl_observation_tile.colnames) & ('target' in tbl_observation_tile.colnames):
                tbl_observation_tile.remove_columns(['tile', 'target'])
            tbl_observation = vstack([tbl_observation, tbl_observation_tile])
            
        if len(tbl_observation) > 0:
            alert_instance.is_observed = True     
            alert_instance.last_observed_time = Time(np.max(Time(tbl_observation['mjd'], format = 'mjd')), format = 'mjd').isot
        else:
            alert_instance.is_observed = False
            alert_instance.last_observed_time = None
        return alert_instance
    
    def check_ezphot_processed(self, alert_instance: Alert):
        # Try with objname first
        self.ezphot_connector.objname = np.atleast_1d(alert_instance.objname)[0]
        imginfo_objname = self.ezphot_connector.search(
            pattern = '*.fits',
            return_type = 'imginfo'
        )
        imginfo_tile = Table()
        if alert_instance.tile_id is not None:
            self.ezphot_connector.objname = np.atleast_1d(alert_instance.tile_id)[0]
            imginfo_tile = self.ezphot_connector.search(
            pattern = '*.fits',
            return_type = 'imginfo'
            )
            
        imginfo = Table()
        for info_tbl in (imginfo_tile, imginfo_objname):
            if info_tbl is not None:
                imginfo = vstack([imginfo, info_tbl])        
            
        if len(imginfo) > 0:
            imginfo_valid = imginfo[Time(imginfo['obsdate'], format = 'isot').mjd>= Time(alert_instance.trigger_time).mjd]
        else:
            imginfo_valid = []
            
        if len(imginfo_valid) > 0:
            alert_instance.is_processed = True
            alert_instance.last_processed_time = np.max(Time(imginfo_valid['obsdate'], format = 'isot')).isot
        else:
            alert_instance.is_processed = False
            alert_instance.last_processed_time = None
        # alert_instance.update_time = Time.now().isot
        # self.update_alert_db_status(alert_instance)
        return alert_instance

    def check_ezphot_reference(self, alert_instance: Alert, days_before = 5, days_after = 150):
        databrowser = DataBrowser('refdata')
        databrowser.objname = np.atleast_1d(alert_instance.objname)[0]
        imginfo_objname = databrowser.search(
            pattern = '*.fits',
            return_type = 'imginfo'
        )
        imginfo_tile = Table()
        if alert_instance.tile_id is not None:
            databrowser.objname = np.atleast_1d(alert_instance.tile_id)[0]
            imginfo_tile = databrowser.search(
            pattern = '*.fits',
            return_type = 'imginfo'
            )
                
        imginfo = Table()
        for info_tbl in (imginfo_tile, imginfo_objname):
            if info_tbl is not None:
                imginfo = vstack([imginfo, info_tbl])   
                        
        if len(imginfo) > 0:
            mjd_mask = (Time(imginfo['obsdate'], format = 'isot').mjd < Time(alert_instance.trigger_time).mjd-days_before) | (Time(imginfo['obsdate'], format = 'isot').mjd > Time(alert_instance.trigger_time).mjd+days_after)
            imginfo_valid = imginfo[mjd_mask]
        else:
            imginfo_valid = []
            
        if len(imginfo_valid) > 0:
            alert_instance.is_reference_ready = True
        else:
            alert_instance.is_reference_ready = False
        # self.update_alert_db_status(alert_instance)
        return alert_instance
    
    def prepare_reference_img(self, alert_instance: Alert, remove_single_images: bool = True):
        if alert_instance.is_reference_ready:
            return alert_instance
        processor = AlertProcessor()
        processor.load_images_db(alert_instance, 'raw', obs_start_time = None, obs_end_time = alert_instance.trigger_time-5*u.day)
        if len(processor.target_images) == 0:
            return alert_instance
        # Prepare images for stacking
        processor.pipeline_before_stacking(alert_instance)
        single_images = processor.target_images
        imgset = ImageSet(processor.target_images)
        imgsetlist = imgset.divide_images(by_filter = True,
                                          by_exptime = False,
                                          by_objname = False,
                                          by_telname = False,
                                          by_observatory = False,
                                          by_obsdate = False)
        target_images_to_stack_all = []
        for imgset_filter in imgsetlist:
            target_images_to_stack = imgset_filter.select_quality_images(
                seeing_key = 'SEEING',
                depth_key = 'UL5SKY_APER_2',
                ellipticity_key = 'ELLIP',
                obsdate_key = 'DATE-OBS',
                weight_ellipticity = 3.0,
                weight_seeing = 2.0,
                weight_depth = 1.0,
                max_numbers = None,
                seeing_limit = 6.0,
                depth_limit = 15.0,
                ellipticity_limit = 0.5,
                visualize = True,
                verbose = True,
            )
            target_images_to_stack_all.extend(target_images_to_stack)
        processor.target_images = target_images_to_stack_all
            
        # Stack images
        processor.stacking(by_filter = True,
                           by_exptime = False,
                           by_objname = False,
                           by_telname = False,
                           by_observatory = True,
                           by_obsdate = False)
        processor.config.do_DIA = False
        processor.config.do_stack = False
        # Process the stacked image
        processor.pipeline_after_stacking(alert_instance)
        target_imglist = processor.target_images
        
        for target_img in target_imglist:
            reference_img = target_img.to_referenceimage()
            reference_img.register(verbose = False)
            dest_dir = reference_img.path.parent 
            target_img.remove(remove_main = True, remove_connected_files = True, skip_patterns = ['*cat*'], verbose = False)
            connected_files = target_img.connected_files
            for connected_file in connected_files:
                connected_file = Path(connected_file)
                connected_file.rename(dest_dir / connected_file.name)
        
        if remove_single_images:
            for single_img in single_images:
                single_img.remove(remove_main = True, remove_connected_files = True, verbose = False)
        alert_instance.is_reference_ready = True
        # alert_instance.update_time = Time.now().isot
        # self.update_alert_db_status(alert_instance)
        return alert_instance
    
    def trigger_photometry(self, 
                           alert_instance: Alert, 
                           remove_single_images: bool = True):
        processor = AlertProcessor()

        last_processed_time = alert_instance.last_processed_time
        if last_processed_time is not None:
            last_processed_time = Time(last_processed_time) - 1 * u.day
            processor.load_images_db(alert_instance, 'raw', obs_start_time = last_processed_time)
        else:
            processor.load_images_db(alert_instance, 'raw')
        if len(processor.target_images) == 0:
            return False
        # Prepare images for stacking
        processor.pipeline_before_stacking(alert_instance)
        single_images = processor.target_images
        # Stack images
        processor.stacking(by_filter = True,
                           by_exptime = False,
                           by_objname = False,
                           by_telname = False,
                           by_observatory = True,
                           by_obsdate = True,
                           obsdate_delta = 0.5,
                           obsdate_key = 'obsdate')
        # Process the stacked image
        processor.pipeline_after_stacking(alert_instance)
        
        if remove_single_images:
            for single_img in single_images:
                single_img.remove(remove_main = True, remove_connected_files = True, verbose = False)
        return True
    
    def trigger_tractor_photometry(self, alert_instance: Alert):
        alert_instance.match_host(self.hostgalaxycatalog,
                                  search_radius_arcsec = 600,
                                  max_dell = 2.5,
                                  return_all = False,
                                  plot = False,
                                  save_path = None)
        processor = AlertProcessor()
        
        all_folders = list(Path(processor.config.tractor_photometry['base_dir']).glob(f'{alert_instance.objname}*'))
        # if len(all_folders) > 0:
        #     all_obsdates = [Time(datetime.datetime.strptime(Path(folder).name.split(alert_instance.objname + "_")[1], '%Y%m%d_%H%M%S')) for folder in all_folders]
        #     last_obsdate = np.max(all_obsdates)
        #     last_processed_time = last_obsdate + 1 * u.day
        # else:
        #     last_processed_time = None
        last_processed_time = alert_instance.last_processed_time
        processor.load_images_ezphot(alert_instance, 'coadd_scaled*com.fits', obs_start_time = last_processed_time)       
        if len(processor.target_images) == 0:
            return False
        processor.tractor_photometry(alert_instance)
        return True
        
    def update_alert_db_status(self, alert_instance: Alert):
        input_dict = alert_instance.__dict__
        # If isinstance(Time), convert to isot
        input_dict = {key: Time(value).isot if isinstance(value, (Time, datetime.datetime)) else value for key, value in input_dict.items()}
        # If isintance of list, tuple, np.ndarray, convert to string
        input_dict = {key: str(value[0]) if isinstance(value, (list, tuple, np.ndarray)) else value for key, value in input_dict.items()}        
        # If not float, int, str, bool, None, remove
        input_dict = {key: value for key, value in input_dict.items() if (np.isscalar(value)) or (value is None)}
        keys = input_dict.keys()
        values = input_dict.values()
        result = self.db_connector.update_row(tbl_name = 'transient_status', update_value = values, update_key = keys, id_value = alert_instance.status_id, id_key = 'status_id')
        return result
    
    def get_pipeline_photometry(self, alert_instance: Alert):
        self.gwportal_connector.query_type = 'combined'
        # First try with coordinate
        tbl_combined_coord = self.gwportal_connector.query(
            obs_start_date = Time(alert_instance.trigger_time).datetime.strftime("%Y-%m-%d"),
            ra = alert_instance.ra,
            dec = alert_instance.dec,
            verbose = False
            )
        # Does not work with the objname now (251215)
        tbl_combined = tbl_combined_coord
        
        if len(tbl_combined) == 0:
            return None
        
        # py7dt
        all_catalog_paths = [Path(path).parent /  Path(path).name.replace('.fits', '_cat.fits') for path in tbl_combined['filepath']]
        path_exists = [path.exists() for path in all_catalog_paths]
        # gppy
        if not all(path_exists):
            all_catalog_paths = [Path(path).parent / 'phot' /  Path(path).name.replace('.fits', '.phot.cat') for path in tbl_combined['filepath']]
            path_exists = [path.exists() for path in all_catalog_paths]
            if not all(path_exists):
                return None
        
        catlist = []
        for path in tqdm(all_catalog_paths, desc = 'Loading catalogs...'):
            catlist.append(Catalog(path = path))
            
        # Sort objname and tile_id
        catlist_objname = []
        catlist_tile_id = []
        for cat in catlist:
            if cat.info.objname == alert_instance.objname:
                catlist_objname.append(cat)
            elif cat.info.objname == alert_instance.tile_id[0]:
                catlist_tile_id.append(cat)
        catlist = catlist_objname + catlist_tile_id

        for cat in tqdm(catlist, desc = 'Formatting...'):
            filter_ = cat.info.filter
            all_colnames_with_filter = [col for col in cat.data.colnames if col.endswith(f'_{filter_}')]
            all_colnames_without_filter = [col.replace(f'_{filter_}', '') for col in cat.data.colnames if col.endswith(f'_{filter_}')]
            all_colnames_without_filter_formatted = [col.replace('MAG_', 'MAGSKY_') for col in all_colnames_without_filter]

            cols_to_remove = [col for col in all_colnames_without_filter_formatted if col in cat.data.colnames]
            if cols_to_remove:
                cat.data.remove_columns(cols_to_remove)
            cat.data.rename_columns(all_colnames_with_filter, all_colnames_without_filter_formatted)
            cat.data.rename_columns(['ALPHA_J2000', 'DELTA_J2000'], ['X_WORLD', 'Y_WORLD'])
            
            target_img = cat.target_img
            if target_img is not None:
                header = target_img.header
                zp_colnames = [col for col in header.keys() if col.startswith('ZP_')]
                zp_colnames_formatted = []
                zperr_colnames = []
                zperr_colnames_formatted = []
                ul_colnames = []
                ul_colnames_formatted = []
                for zp_colname in zp_colnames:
                    zperr_colname = zp_colname.replace('ZP_', 'EZP_')
                    ul_colname = zp_colname.replace('ZP_', 'UL5_')
                    
                    if not zp_colname.endswith('_AUTO'):
                        if zp_colname.endswith('_0'):
                            zp_colname_formatted = zp_colname.replace('_0', '_APER')
                        else:
                            zp_colname_formatted = zp_colname.replace('ZP_', 'ZP_APER_')
                        
                        zperr_colname_formatted = zp_colname_formatted.replace('ZP_', 'ZPERR_')
                        ul_colname_formatted = zp_colname_formatted.replace('ZP_', 'UL5SKY_')
                    else:
                        zp_colname_formatted = zp_colname
                        zperr_colname_formatted = zp_colname_formatted.replace('ZP_', 'ZPERR_')
                        ul_colname_formatted = zp_colname_formatted.replace('ZP_', 'UL5SKY_')
                    
                    zp_colnames_formatted.append(zp_colname_formatted)
                    zperr_colnames.append(zperr_colname)
                    zperr_colnames_formatted.append(zperr_colname_formatted)
                    ul_colnames.append(ul_colname)
                    ul_colnames_formatted.append(ul_colname_formatted)
                    
                for col_in_header, col_in_catalog in zip(zp_colnames, zp_colnames_formatted):
                    cat.data[col_in_catalog] = header[col_in_header]
                for col_in_header, col_in_catalog in zip(zperr_colnames, zperr_colnames_formatted):
                    cat.data[col_in_catalog] = header[col_in_header]
                for col_in_header, col_in_catalog in zip(ul_colnames, ul_colnames_formatted):
                    cat.data[col_in_catalog] = header[col_in_header]
        
        catalog_set = CatalogSet(catlist)
        if self.config.save_catalog:
            save_catalog_dir = f'gppy_v1_{Time(alert_instance.update_time).datetime.strftime("%Y%m%d_%H%M%S")}'
            for catalog in tqdm(catalog_set.target_catalogs, desc = 'Saving catalogs...'):
                catalog.savedir = Path(self.config.save_dir) / alert_instance.objname / save_catalog_dir
                catalog.write(verbose = False)
        return catalog_set
        
    def get_ezphot_photometry(self, alert_instance: Alert, pattern: str = 'coadd_scaled*com.fits.cat', catalog_type: str = 'ezphot'):
        # 1st try with objname
        self.ezphot_connector.objname = np.atleast_1d(alert_instance.objname)[0]
        catalog_set_objname = self.ezphot_connector.search(pattern = pattern, return_type = 'catalog')
        all_target_catalogs = catalog_set_objname.target_catalogs
        
        # 2nd try with tile_id
        if alert_instance.tile_id is not None:
            self.ezphot_connector.objname = np.atleast_1d(alert_instance.tile_id)[0]
            catalog_set_tile = self.ezphot_connector.search(pattern = pattern, return_type = 'catalog')
            all_target_catalogs += catalog_set_tile.target_catalogs
    
        if len(all_target_catalogs) == 0:
            return None
        catalog_set = CatalogSet(all_target_catalogs)
        catalog_set.select_catalogs(
            obs_start = Time(alert_instance.trigger_time).datetime.strftime("%Y-%m-%d"),
        )
        
        if self.config.save_catalog:
            save_catalog_dir = f'{Time(alert_instance.update_time).datetime.strftime("%Y%m%d_%H%M%S")}_{catalog_type}'
            for catalog in tqdm(catalog_set.target_catalogs, desc = 'Saving catalogs...'):
                catalog.savedir = Path(self.config.save_dir) / alert_instance.objname / save_catalog_dir
                catalog.write(verbose = False)
        return catalog_set
    
    def get_tractor_photometry(self, alert_instance: Alert, pattern: str = '*com.fits.tract7dtcat', catalog_type = 'tract7dt'):
        self.ezphot_connector.objname = np.atleast_1d(alert_instance.objname)[0]
        catalog_set_objname = self.ezphot_connector.search(pattern = pattern, return_type = 'catalog')
        all_target_catalogs = catalog_set_objname.target_catalogs
        
        # 2nd try with tile_id
        if alert_instance.tile_id is not None:
            self.ezphot_connector.objname = np.atleast_1d(alert_instance.tile_id)[0]
            catalog_set_tile = self.ezphot_connector.search(pattern = pattern, return_type = 'catalog')
            all_target_catalogs += catalog_set_tile.target_catalogs
    
        if len(all_target_catalogs) == 0:
            return None
        catalog_set = CatalogSet(all_target_catalogs)
        catalog_set.select_catalogs(
            obs_start = Time(alert_instance.trigger_time).datetime.strftime("%Y-%m-%d"),
        )
        
        if self.config.save_catalog:
            save_catalog_dir = f'{Time(alert_instance.update_time).datetime.strftime("%Y%m%d_%H%M%S")}_{catalog_type}'
            for catalog in tqdm(catalog_set.target_catalogs, desc = 'Saving catalogs...'):
                catalog.savedir = Path(self.config.save_dir) / alert_instance.objname / save_catalog_dir
                catalog.write(verbose = False)
        return catalog_set
        
    def draw_lightcurve(self, alert_instance: Alert, catalog_set: CatalogSet, save_path : str = None, catalog_type: str = 'ezphot'):
        try:            
            if not alert_instance.is_coordinate_given:
                return False
            
            all_mjds = [Time(cat.info.obsdate, format = 'isot').mjd for cat in catalog_set.target_catalogs]
            mid_mjd = np.mean(all_mjds)
            dateime_string = Time(mid_mjd, format = 'mjd').datetime.strftime("%Y%m%d_%H%M%S")
            if save_path is None:
                filename = f'{alert_instance.objname}_{dateime_string}_{catalog_type}_LC.png'
                save_path = Path(self.config.save_dir) / alert_instance.objname /  filename
            save_path.parent.mkdir(parents = True, exist_ok = True)

            # if save_path.exists():
            #     return False
            matching_radius_arcsec = self.config.matching_radius_arcsec
            if 'py7dt' in catalog_type:
                flux_key = 'MAGSKY_APER_5'
                fluxerr_key = 'MAGERR_APER_5'
                zperr_key = 'ZPERR_APER_5'
                depth_key = 'UL5SKY_APER_5'
            elif 'tract' in catalog_type:
                flux_key = 'MAG_TRACT7DT'
                fluxerr_key = 'MAGERR_TRACT7DT'
                zperr_key = 'ZPERR_TRACT7DT'
                depth_key = 'UL5SKY_TRACT7DT'
                matching_radius_arcsec = 1
            else:
                flux_key = self.config.flux_key
                fluxerr_key = self.config.fluxerr_key
                zperr_key = self.config.zperr_key
                depth_key = self.config.depth_key
            
            lc = LightCurve(catalog_set)
            lc.extract_source_info(
                ra = alert_instance.ra,
                dec = alert_instance.dec,
                ra_key = self.config.ra_key,
                dec_key = self.config.dec_key,
                flux_key = flux_key,
                fluxerr_key = fluxerr_key,
                zperr_key = zperr_key,
                depth_key = depth_key,
                matching_radius_arcsec=matching_radius_arcsec)

            if lc.data is None:
                return False
            
            for key, value in self.config.lightcurve.items():
                if value is not None:
                    setattr(lc.plt_params, key, value)
            
            lightcurve_all_at_once = lc.plot(
                ra = alert_instance.ra, 
                dec = alert_instance.dec,
                matching_radius_arcsec = matching_radius_arcsec,
                ra_key = self.config.ra_key, 
                dec_key = self.config.dec_key,
                flux_key = flux_key,
                fluxerr_key = fluxerr_key,
                zperr_key = zperr_key,
                depth_key = depth_key,
                plot_all_in_one_figure = True,
                apply_offset = True,
                overplot_stamp = False,
                title = alert_instance.objname + f' ({catalog_type})',
                verbose = False)

            lightcurve_single = lc.plot(
                ra = alert_instance.ra, 
                dec = alert_instance.dec,
                matching_radius_arcsec = matching_radius_arcsec,
                ra_key = self.config.ra_key, 
                dec_key = self.config.dec_key,
                flux_key = flux_key,
                fluxerr_key = fluxerr_key,
                zperr_key = zperr_key,
                depth_key = depth_key,
                plot_all_in_one_figure = False,
                apply_offset = True,
                overplot_stamp = True,
                title = alert_instance.objname + f' ({catalog_type})',
                verbose = False)
            
            if self.config.save_lightcurve:
                # Save lightcurve data
                lc.data.write(save_path.parent / (str(save_path.stem) + '.dat'), format = 'ascii.fixed_width')
                # Save all at once
                for key, fig in lightcurve_all_at_once[0].items():
                    try:
                        fig.savefig(save_path, dpi = 300, bbox_inches = 'tight')
                        plt.close(fig)
                    except Exception as e:
                        print(f"Error saving lightcurve: {e}")
                # Save single
                for key, fig in lightcurve_single[0].items():
                    try:
                        save_path_single = save_path.parent / (str(save_path.stem) + f'_{key}.png')
                        fig.savefig(save_path_single, dpi = 300, bbox_inches = 'tight')
                        plt.close(fig)
                    except Exception as e:
                        print(f"Error saving lightcurve: {e}")
            
            plt.close("all")
            return True
        
        except Exception as e:
            print(f"Error drawing lightcurve: {e}")
            alert_instance.lc_error = str(e)
            return False

    def draw_photometricspectrum(self, alert_instance: Alert, catalog_set: CatalogSet, save_path : str = None, catalog_type: str = 'gppy_v1'):
        try:
            if not alert_instance.is_coordinate_given:
                return False
            
            all_mjds = [Time(cat.info.obsdate, format = 'isot').mjd for cat in catalog_set.target_catalogs]
            mid_mjd = np.mean(all_mjds)
            dateime_string = Time(mid_mjd, format = 'mjd').datetime.strftime("%Y%m%d_%H%M%S")
            if save_path is None:
                filename = f'{alert_instance.objname}_{dateime_string}_{catalog_type}_PS.png'
                save_path = Path(self.config.save_dir) / alert_instance.objname /  filename
            save_path.parent.mkdir(parents = True, exist_ok = True)

            # if save_path.exists():
            #     return False
            matching_radius_arcsec = self.config.matching_radius_arcsec
            if 'py7dt' in catalog_type:
                flux_key = 'MAGSKY_APER_5'
                fluxerr_key = 'MAGERR_APER_5'
                zperr_key = 'ZPERR_APER_5'
                depth_key = 'UL5SKY_APER_5'
            elif 'tract' in catalog_type:
                flux_key = 'MAG_TRACT7DT'
                fluxerr_key = 'MAGERR_TRACT7DT'
                zperr_key = 'ZPERR_TRACT7DT'
                depth_key = 'UL5SKY_TRACT7DT'
                matching_radius_arcsec = 1
            else:
                flux_key = self.config.flux_key
                fluxerr_key = self.config.fluxerr_key
                zperr_key = self.config.zperr_key
                depth_key = self.config.depth_key
            
            photspec = PhotometricSpectrum(catalog_set)
            photspec.extract_source_info(
                ra = alert_instance.ra,
                dec = alert_instance.dec,
                ra_key = self.config.ra_key,
                dec_key = self.config.dec_key,
                flux_key=flux_key,
                fluxerr_key=fluxerr_key,
                zperr_key = zperr_key,
                depth_key = depth_key,
                matching_radius_arcsec=matching_radius_arcsec)

            if photspec.data is None:
                return False
            
            for key, value in self.config.photometricspectrum.items():
                if value is not None:
                    setattr(photspec.plt_params, key, value)
            
            photspec_all_at_once = photspec.plot(
                ra = alert_instance.ra,
                dec = alert_instance.dec,
                matching_radius_arcsec = matching_radius_arcsec,
                ra_key = self.config.ra_key,
                dec_key = self.config.dec_key,
                flux_key = flux_key,
                fluxerr_key = fluxerr_key,
                zperr_key = zperr_key,
                depth_key = depth_key,
                plot_all_in_one_figure = True,
                overplot_gaiaxp = False,
                overplot_sdss = False,
                overplot_ps1 = False,
                overplot_stamp = False,
                verbose = False,
                title = alert_instance.objname + f' ({catalog_type})'
            )

            photspec_single = photspec.plot(
                ra = alert_instance.ra,
                dec = alert_instance.dec,
                matching_radius_arcsec = matching_radius_arcsec,
                ra_key = self.config.ra_key,
                dec_key = self.config.dec_key,
                flux_key = flux_key,
                fluxerr_key = fluxerr_key,
                zperr_key = zperr_key,
                depth_key = depth_key,
                plot_all_in_one_figure = False,
                overplot_gaiaxp = False,
                overplot_sdss = False,
                overplot_ps1 = False,
                overplot_stamp = True,
                verbose = False,
                title = alert_instance.objname + f' ({catalog_type})'
            )

            num_spectrum = len(set(photspec_single[3]['obsdate_group']))

            if self.config.save_photometricspectrum:
                # Save photometricspectrum data
                photspec.data.write(save_path.parent / (str(save_path.stem) + '.dat'), format = 'ascii.fixed_width')
                if num_spectrum == 1:
                    for key, fig in photspec_single[0].items():
                        try:
                            fig.savefig(save_path, dpi = 300, bbox_inches = 'tight')
                            plt.close(fig)
                        except Exception as e:
                            print(f"Error saving photometricspectrum: {e}")
                else:
                    for key, fig in photspec_all_at_once[0].items():
                        try:
                            fig.savefig(save_path, dpi = 300, bbox_inches = 'tight')
                            plt.close(fig)
                        except Exception as e:
                            print(f"Error saving photometricspectrum: {e}")
                    for key, fig in photspec_single[0].items():
                        try:
                            key_str = key.replace(' ', '_')
                            save_path_single = save_path.parent / (str(save_path.stem) + f'_{key_str}.png')
                            fig.savefig(save_path_single, dpi = 300, bbox_inches = 'tight')
                            plt.close(fig)
                        except Exception as e:
                            print(f"Error saving photometricspectrum: {e}")
            plt.close("all")
            return  True
        
        except Exception as e:
            print(f"Error drawing photometricspectrum: {e}")
            alert_instance.ps_error = str(e)
            return False
        
    def classify_alert(self, alert_instance: Alert, catalog_set: CatalogSet, catalog_type = 'ezphot'):
        
        def _safe_value(v):
            if isinstance(v, np.generic):
                v = v.item()          # np.float32 -> float
            if isinstance(v, str):
                v = v.strip()
            return v
        
        if 'py7dt' in catalog_type:
            flux_key = 'MAGSKY_APER_5'
            fluxerr_key = 'MAGERR_APER_5'
            zperr_key = 'ZPERR_APER_5'
            depth_key = 'UL5SKY_APER_5'
        elif 'tract' in catalog_type:
            flux_key = 'MAG_TRACT7DT'
            fluxerr_key = 'MAGERR_TRACT7DT'
            zperr_key = 'ZPERR_TRACT7DT'
            depth_key = 'UL5SKY_TRACT7DT'
        else:
            flux_key = self.config.flux_key
            fluxerr_key = self.config.fluxerr_key
            zperr_key = self.config.zperr_key
            depth_key = self.config.depth_key
        

        catalogset_list = catalog_set.divide_catalogs()
        all_rows = []

        for catalogset_single in catalogset_list:

            all_mjds = [Time(cat.info.obsdate, format = 'isot').mjd for cat in catalogset_single.target_catalogs]
            mid_mjd = np.mean(all_mjds)
            dateime_string = Time(mid_mjd, format = 'mjd').datetime.strftime("%Y%m%d_%H%M%S")
            filename = f'{alert_instance.objname}_{dateime_string}_{catalog_type}_CL_summary.ascii_fixed_width'
            save_dir = Path(self.config.save_dir) / alert_instance.objname
            save_path = save_dir /  filename
                
            # if save_path.exists():
            #     continue

            photspec = PhotometricSpectrum(catalogset_single)

            photspec.extract_source_info(
                ra=alert_instance.ra,
                dec=alert_instance.dec,
                ra_key=self.config.ra_key,
                dec_key=self.config.dec_key,
                flux_key=flux_key,
                fluxerr_key=fluxerr_key,
                zperr_key=zperr_key,
                depth_key=depth_key,
                matching_radius_arcsec=self.config.matching_radius_arcsec,
            )
            
            if photspec.data is None or len(photspec.data) == 0 or is_masked(photspec.data[flux_key]):
                print(f"No valid photometry data for {alert_instance.objname}")
                continue

            obsdate_str = str(photspec.data[0]["obsdate_group"]).replace(' ', '_')

            _, file_path = self.classifier.ngsf_formatter(
                photspec,
                objname=alert_instance.objname,
                mag_key=flux_key,
                magerr_key=fluxerr_key,
                filter_key=self.config.filter_key,
                objname_key=self.config.objname_key,
                visualize=False,
                save=True,
                save_path=save_dir / f"{alert_instance.objname}_{dateime_string}_{catalog_type}_CL_spec.csv",
            )

            superfit = self.classifier.fit(file_path, redshift=None)

            if not hasattr(superfit, "results"):
                continue

            alert_instance.is_classified = True
            
            results = superfit.results[:5]

            for i,row in enumerate(results):
                # Plt result
                fig, ax = superfit.plot_fit_result(i)
                save_fig_path = save_dir / f"{alert_instance.objname}_{dateime_string}_{catalog_type}_CL_fit{i}.png"
                fig.savefig(save_fig_path, dpi = 300, bbox_inches = 'tight')
                plt.close(fig)
                row_to_input = {
                    "objname": alert_instance.objname,
                    "obsdate": obsdate_str,
                    "classification": row["SN_TYPE"],
                    "redshift": row["Z"],
                    "extinction": row["A_v"],
                    "phase": row["Phase"],
                    "frac_sn": row["Frac(SN)"],
                    "frac_host": row["Frac(gal)"],
                    "chisq": row["CHI2/dof"],
                    "file": file_path
                }

                row_to_input = {k: _safe_value(v) for k, v in row_to_input.items()}
                all_rows.append(row_to_input)


        df = pd.DataFrame(all_rows)
        
        if len(df) == 0:
            return alert_instance

        cls_series = (
            df["classification"]
            .astype(str)
            .str.strip()
        )
        cls_series = cls_series[(cls_series != "") & (cls_series.str.lower() != "nan")]

        if len(cls_series) == 0:
            classification_result = None
            classification_counts = {}
        else:
            vc = cls_series.value_counts(dropna=True)
            classification_counts = vc.to_dict()
            classification_result = vc.index[0]

        result_tbl = Table.from_pandas(df)
        result_tbl.write(save_path, format="ascii.fixed_width", overwrite=True)
        
        alert_instance.is_classified = True
        alert_instance.classification_result = classification_result

        return True
# %%
if __name__ == "__main__":
    self = AlertChecker()
    while True:
        print("Checking alerts... at", Time.now().isot)
        self.check_alerts()
        time.sleep(300)
    

# %%