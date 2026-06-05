#%%

import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.table import Table
import time
from astropy.visualization import ZScaleInterval
import io
import requests
import pandas as pd
import io
import requests
from astropy.io import fits
from bridge.configuration import Configuration
import numpy as np
#%%
class FINKQuerier:
   
   def __init__(self, 
                survey_type: str = 'ztf'):
      self.survey_type = survey_type
      self.config = Configuration(config_filenames=['finkquerier.config'])
      self.max_retries = 3
      self.retry_delay = 5
   
   def query_alerts(self,
                     firstmjd: float = Time.now().mjd - 5,
                     lastmjd: float = None,#Time.now().mjd,
                     verbose: bool = True):
      """
      Query FINK alerts with a given trend, probability, and probability type.

      Parameters
      ----------
      trend: str
         Trend of the alert. rising or fading
      probability: float
         Probability of the alert.
      probability_type: str
         Probability type of the alert. Probability types can be found in the FINK documentataion: https://fink-broker.readthedocs.io/en/latest/broker/science_modules/#machine-and-deep-learning
      firstmjd: float
         First MJD of the alert.
      lastmjd: float
         Last MJD of the alert.
      n: int
         Number of alerts to query.
      """
      
      if self.survey_type == 'lsst':
         raise RuntimeError('LSST is not supported for querying alerts')
      
      startdate = Time(firstmjd, format='mjd').iso
      stopdate = Time(lastmjd, format='mjd').iso if lastmjd is not None else Time.now().iso
      fink_kwargs = self.config._dict[self.survey_type]['queryconfig']
      fink_kwargs.update({'startdate': startdate, 'stopdate': stopdate})
      fink_kwargs_copy = fink_kwargs.copy()
      for key, value in fink_kwargs.items():
         if value is None:
            fink_kwargs_copy.pop(key)
      fink_kwargs = fink_kwargs_copy
      
      if verbose:
         print('Querying alerts from FINKBroker')
         print('===== Configuration =====')
         for key, value in fink_kwargs.items():
            print(key, ': ', value)
         print('=========================')
         
      start_time = time.time()
      tbl = Table()
      for i in range(self.max_retries):
         try:
            tbl = self.query_with_config(**fink_kwargs)
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
      url_detections = f'{self.config._dict[self.survey_type]["base_url"]}/{self.config._dict[self.survey_type]["url_key"]["objects"]}'
      if verbose:
         print('Querying detections from FINKBroker')
         print('Object ID: ', object_id)
      query_kwargs = {}
      if self.survey_type == 'ztf':
         query_kwargs['objectId'] = object_id
         query_kwargs['output-format'] = 'json'
      elif self.survey_type == 'lsst':
         query_kwargs['diaObjectId'] = str(object_id)
         query_kwargs['output-format'] = 'json'
      def get_detections(object_id: str):
         r = requests.post(
            url_detections,
            json=query_kwargs,
            timeout = 300
         )
         pdf = pd.read_json(io.BytesIO(r.content))
         result = Table.from_pandas(pdf)
         return result
      detections = Table()
      for i in range(self.max_retries):
         try:
            detections = get_detections(object_id)
            break
         except Exception as e:
            if verbose:
               print(f"Retry {i+1} of {self.max_retries}: Error querying detections: {e}")
            time.sleep(self.retry_delay)
      return detections   
   
   def query_stamp(self,
                   object_id: str,
                   measurement_id: str = None,
                   verbose: bool = True):
      url_stamp = f'{self.config._dict[self.survey_type]["base_url"]}/{self.config._dict[self.survey_type]["url_key"]["cutouts"]}'
      if verbose:
         print('Querying stamp from FINKBroker')
         print('Object ID: ', object_id)
      
      if self.survey_type == 'ztf':
         query_kwargs = {
            'objectId': str(object_id),
            'kind': 'All',
            'output-format': 'array'
         }
         if measurement_id is not None:
            query_kwargs['candid'] = str(measurement_id)
      elif self.survey_type == 'lsst':
         query_kwargs = {
            'diaSourceId': str(object_id),
            'kind': 'All',
            'output-format': 'array'
         }
         if measurement_id is not None:
            query_kwargs['diaSourceId'] = str(measurement_id)

      def get_stamp():
         # get data for ZTF21aaxtctv
         r = requests.post(
            url_stamp,
            json=query_kwargs,
            timeout = 300
         )
         data = r.json()
         
         if self.survey_type == 'ztf':
            sci = data['b:cutoutScience_stampData']
            ref = data['b:cutoutTemplate_stampData']
            sub = data['b:cutoutDifference_stampData']
         elif self.survey_type == 'lsst':
            sci = data['b:cutoutScience']
            ref = data['b:cutoutTemplate']
            sub = data['b:cutoutDifference']
         
         hdu_sci = fits.PrimaryHDU(sci)
         hdu_ref = fits.PrimaryHDU(ref)
         hdu_sub = fits.PrimaryHDU(sub)
         
         hdulist = fits.HDUList([hdu_sci, hdu_ref, hdu_sub])
         return hdulist
         
      for i in range(self.max_retries):
         try:
            stamplist = get_stamp()
            break
         except Exception as e:
            if verbose:
               print(f"Retry {i+1} of {self.max_retries}: Error querying stamp: {e}")
            time.sleep(self.retry_delay)
      
      stamp_dict = dict(cutoutScience=stamplist[0], cutoutTemplate=stamplist[1], cutoutDifference=stamplist[2])
      return stamp_dict

   def query_detections(self,
                        object_id: str,
                        verbose: bool = True):
      url_detections = f'{self.config._dict[self.survey_type]["base_url"]}/{self.config._dict[self.survey_type]["url_key"]["detections"]}'
      if verbose:
         print('Querying detections from FINKBroker')
         print('Object ID: ', object_id)
      query_kwargs = {}
      if self.survey_type == 'ztf':
         query_kwargs['objectId'] = object_id
         query_kwargs['output-format'] = 'json'
      elif self.survey_type == 'lsst':
         query_kwargs['diaObjectId'] = str(object_id)
         query_kwargs['output-format'] = 'json'
      def get_detections(object_id: str):
         r = requests.post(
            url_detections,
            json=query_kwargs,
            timeout = 300
         )
         pdf = pd.read_json(io.BytesIO(r.content))
         result = Table.from_pandas(pdf)
         return result
      detections = Table()
      for i in range(self.max_retries):
         try:
            detections = get_detections(object_id)
            break
         except Exception as e:
            if verbose:
               print(f"Retry {i+1} of {self.max_retries}: Error querying detections: {e}")
            time.sleep(self.retry_delay)
      return detections
   
   def query_schema(self, 
                    endpoint: str = '/api/v1/objects'):
      url_schema = f'{self.config._dict[self.survey_type]["base_url"]}/{self.config._dict[self.survey_type]["url_key"]["schema"]}'
      if self.survey_type == 'ztf':
         query_kwargs = {}
      else:
         query_kwargs = {'endpoint': endpoint,
                         'major_version': 10,
                         'minor_version': 0}
      r = requests.get(url_schema, json=query_kwargs)
      pdf = pd.read_json(io.BytesIO(r.content))
      tbl = Table.from_pandas(pdf)
      return tbl


   def plot_stamp(self, 
                  object_id: str, 
                  measurement_idx: int = 0,
                  show: bool = True, 
                  save_path: str = None, 
                  dpi: int = 300, 
                  show_magnitude: bool = True,
                  verbose: bool = True):
      if verbose:
         print('Plotting stamp from FINKBroker')
         print('Object ID: ', object_id)
         print('Measurement index: ', measurement_idx)
         
      measurement_id = None
      title_suffix = ''
      if show_magnitude or measurement_idx != 0:
         detection_tbl = self.query_detections(object_id, verbose = False)
         if len(detection_tbl) < measurement_idx + 1:
            raise ValueError(f"Measurement index {measurement_idx} is out of range for object {object_id}. Number of detections: {len(detection_tbl)}")
         if self.survey_type == 'ztf':
            measurement_id = detection_tbl['i:candid'][measurement_idx]
            magnitude = detection_tbl['i:magpsf'][measurement_idx] 
            magerr = detection_tbl['i:sigmapsf'][measurement_idx]
            obsdate = Time(detection_tbl['i:jd'][measurement_idx], format='jd').datetime.strftime('%Y-%m-%dT%H:%M:%S')
            title_suffix = f'\nMagnitude[PSF]: {magnitude:.2f} ± {magerr:.2f} ({obsdate})'
         elif self.survey_type == 'lsst':
            measurement_id = detection_tbl['r:diaSourceId'][measurement_idx]
            flux = detection_tbl['r:psfFlux'][measurement_idx] 
            fluxerr = detection_tbl['r:psfFluxErr'][measurement_idx]
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
   
   def query_with_config(self, **kwargs):
      url_alerts = f'{self.config._dict[self.survey_type]["base_url"]}/{self.config._dict[self.survey_type]["url_key"]["alerts"]}'
      # Get all classified SN Ia from TNS between March 1st 2021 and March 5th 2021
      r = requests.post(
      url_alerts,
      json=kwargs,
      timeout = 300
      )
      pdf = pd.read_json(io.BytesIO(r.content))
      tbl = Table.from_pandas(pdf)
      return tbl

   def show_classifiers(self):
      import requests

      url = f'{self.config._dict[self.survey_type]["base_url"]}/classes'

      response = requests.get(url)
      response.raise_for_status()  # raise error if request failed

      classes = response.json()       # parsed JSON (dict or list)
      return classes#['Fink classifiers']
