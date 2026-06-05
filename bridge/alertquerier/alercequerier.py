#%%
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.table import Table
import time
from astropy.visualization import ZScaleInterval
import numpy as np

from alerce.core import Alerce
from bridge.configuration import Configuration

#%%
class ALERCEQuerier:
      
   def __init__(self,
                survey_type: str = 'ztf'):
      self.survey_type = survey_type
      self.config = Configuration(config_filenames=['alercequerier.config'])
      self.broker = Alerce()
      self.max_retries = 3
      self.retry_delay = 5

   def query_alerts(self,
                    firstmjd: float = Time.now().mjd - 10,
                    lastmjd: float = None,#Time.now().mjd,
                    verbose: bool = True):
      # Update alert date
      alerce_kwargs = self.config._dict[self.survey_type]['queryconfig']
      alerce_kwargs.update({'firstmjd': firstmjd})
      if lastmjd is not None:
         alerce_kwargs.update({'lastmjd': lastmjd})
      
      if verbose:
         print('Querying alerts from ALERCEBroker')
         print('===== Configuration =====')
         for key, value in alerce_kwargs.items():
            print(key, ': ', value)
         print('=========================')
      start_time = time.time()
      tbl = []
      for i in range(self.max_retries):
         try:
            tbl = Table().from_pandas(self.broker.query_objects(**alerce_kwargs))
            break
         except Exception as e:
            if verbose:
               print(f"Retry {i+1} of {self.max_retries}: Error querying objects: {e}")
            time.sleep(self.retry_delay)
      if verbose:
         print('Query finished: elapsed time= ', time.time() - start_time, 'seconds')
         print('Number of alerts: ', len(tbl))
      return tbl   
   
   def query_object(self, 
                    object_id: str,
                    verbose: bool = True):
      if verbose:
         print('Querying object from ALERCEBroker')
         print('Object ID: ', object_id)
      for i in range(self.max_retries):
         try:
            object = self.broker.query_object(object_id, survey = self.survey_type)
            break
         except Exception as e:
            if verbose:
               print(f"Retry {i+1} of {self.max_retries}: Error querying object: {e}")
            time.sleep(self.retry_delay)
      return object

   def query_stamp(self,
                   object_id: str,
                   measurement_id: str = None,
                   verbose: bool = True):
      if verbose:
         print('Querying stamp from ALERCEBroker')
         print('Object ID: ', object_id)
      stamp_dict = {}
      for i in range(self.max_retries):
         try:
            stamp = self.broker.get_stamps(object_id, survey = self.survey_type, measurement_id = measurement_id)
            if self.survey_type == 'ztf':
               stamp_dict['cutoutScience'] = stamp[0]
               stamp_dict['cutoutTemplate'] = stamp[1]
               stamp_dict['cutoutDifference'] = stamp[2]
            elif self.survey_type == 'lsst':
               stamp_dict = stamp
            else:
               raise ValueError(f"Survey type {self.survey_type} not supported")
            break
               
         except Exception as e:
            if verbose:
               print(f"Retry {i+1} of {self.max_retries}: Error querying stamp: {e}")
            time.sleep(self.retry_delay)
      return stamp_dict

   def query_detections(self,
                        object_id: str,
                        verbose: bool = True):
      if verbose:
         print('Querying detections from ALERCEBroker')
         print('Object ID: ', object_id)
      detection_queried = False
      for i in range(self.max_retries):
         try:
            detections = self.broker.query_detections(object_id, survey = self.survey_type)
            detection_queried = True
            break
         except Exception as e:
            if verbose:
               print(f"Retry {i+1} of {self.max_retries}: Error querying detections: {e}")
            time.sleep(self.retry_delay)
      if detection_queried:
         detection_tbl = Table(detections)
      else:
         detection_tbl = Table()
      return detection_tbl
   
   def query_nondetections(self,
                           object_id: str,
                           verbose: bool = True):
      if verbose:
         print('Querying non-detections from ALERCEBroker')
         print('Object ID: ', object_id)
      non_detection_queried = False
      for i in range(self.max_retries):
         try:
            non_detections = self.broker.query_non_detections(object_id, survey = self.survey_type)
            non_detection_queried = True
            break
         except Exception as e:
            if verbose:
               print(f"Retry {i+1} of {self.max_retries}: Error querying non-detections: {e}")
            time.sleep(self.retry_delay)
      if non_detection_queried:
         non_detection_tbl = Table(non_detections)
      else:
         non_detection_tbl = Table()
      return non_detection_tbl
   
   def query_forced_photometry(self,
                               object_id: str,
                               verbose: bool = True):
      if verbose:
         print('Querying forced photometry from ALERCEBroker')
         print('Object ID: ', object_id)
      for i in range(self.max_retries):
         try:
            forced_photometry = self.broker.query_forced_photometry(object_id, survey = self.survey_type)
            break
         except Exception as e:
            if verbose:
               print(f"Retry {i+1} of {self.max_retries}: Error querying forced photometry: {e}")
            time.sleep(self.retry_delay)
      forced_photometry_tbl = Table(forced_photometry)
      return forced_photometry_tbl
   
   def query_lightcurve(self,
                        object_id: str,
                        verbose: bool = True):
      if verbose:
         print('Querying lightcurve from ALERCEBroker')
         print('Object ID: ', object_id)
      for i in range(self.max_retries):
         try:
            lightcurve = self.broker.query_lightcurve(object_id, survey = self.survey_type)
            break
         except Exception as e:
            if verbose:
               print(f"Retry {i+1} of {self.max_retries}: Error querying lightcurve: {e}")
            time.sleep(self.retry_delay)
      detection_tbl = Table(lightcurve['detections'])
      non_detection_tbl = Table(lightcurve['non_detections'])
      return detection_tbl, non_detection_tbl

   def plot_stamp(self, 
                  object_id: str, 
                  measurement_idx: int = 0,
                  show: bool = True, 
                  save_path: str = None, 
                  dpi: int = 300, 
                  show_magnitude: bool = False,
                  verbose: bool = True):
      # Apply ZSCALE 
      # Get FITS HDUs
      if verbose:
         print('Plotting stamp from ALERCEBroker')
         print('Object ID: ', object_id)
         print('Measurement index: ', measurement_idx)
         
      measurement_id = None
      title_suffix = ''
      if show_magnitude or measurement_idx != 0:
         detection_tbl = self.query_detections(object_id, verbose = False)
         if len(detection_tbl) < measurement_idx + 1:
            raise ValueError(f"Measurement index {measurement_idx} is out of range for object {object_id}. Number of detections: {len(detection_tbl)}")
         if self.survey_type == 'ztf':
            if 'magpsf_corr' in detection_tbl.columns:
               measurement_id = detection_tbl['candid'][measurement_idx]
               magnitude = detection_tbl['magpsf_corr'][measurement_idx] 
               magerr = detection_tbl['sigmapsf_corr'][measurement_idx]
               title_suffix = f'\nMagnitude[PSF_corrected]: {magnitude:.2f} ± {magerr:.2f}'
            else:
               measurement_id = detection_tbl['candid'][measurement_idx]
               magnitude = detection_tbl['magpsf'][measurement_idx] 
               magerr = detection_tbl['sigmapsf'][measurement_idx]
               title_suffix = f'\nMagnitude[PSF_uncorrected]: {magnitude:.2f} ± {magerr:.2f}'
         elif self.survey_type == 'lsst':
            measurement_id = detection_tbl['measurement_id'][measurement_idx]
            flux = detection_tbl['psfFlux'][measurement_idx] 
            fluxerr = detection_tbl['psfFluxErr'][measurement_idx]
            magnitude = -2.5 * np.log10(flux) + 31.4
            magerr = 2.5 / np.log(10) * fluxerr / flux
            title_suffix = f'\nMagnitude[PSF]: {magnitude:.2f} ± {magerr:.2f}'

      hdulist_dict = self.query_stamp(object_id, measurement_id = measurement_id, verbose = False)
         
      # One row per HDU (except primary if empty)
      n = len(hdulist_dict)
      # Set supertitle
      fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
      fig.suptitle(str(object_id) + title_suffix)

      # Handle case of single HDU (axes not iterable)
      if n == 1:
         axes = [axes]
      
      zscale = ZScaleInterval()
      for ax, (key, hdu) in zip(axes, hdulist_dict.items()):
         if hdu is None:
               continue
         
         vmin, vmax = zscale.get_limits(hdu.data)
         im = ax.imshow(hdu.data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
         fig.colorbar(im, ax=ax, shrink=0.7)
         ax.set_title(key)

      plt.tight_layout()

      # Save if requested
      if save_path is not None:
         fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

      if show:
         plt.show()
      plt.close(fig)

      return fig
      
   def show_classifiers(self):
      classifiers = self.broker.query_classifiers(survey = self.survey_type)
      for classifier in classifiers:
         for key, value in classifier.items():
            print(key, ': ', value)
         print('--------------------------------')
      return classifiers         

   @property
   def query_kwargs(self):
      kwargs = {}
      for key, value in self.config._dict.items():
         if value is not None:
            kwargs[key] = value
      return kwargs
# %%
if __name__ == '__main__':
   self = ALERCEQuerier('ztf')
# %%