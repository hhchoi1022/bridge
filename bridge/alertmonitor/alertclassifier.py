#%%

from typing import Union, Optional
import numpy as np
from pathlib import Path
import pandas as pd
from astropy.time import Time


from ezphot.dataobjects import Spectrum
from ezphot.dataobjects import PhotometricSpectrum

from bridge.objects import Alert
from bridge.configuration import Configuration

from NGSF_7DT.sf_class import Superfit
#%%
class AlertClassifier:
    def __init__(self):
        self.config = Configuration(config_filenames=['alertclassifier.config'])
    
    def ngsf_formatter(self, 
                       photspec: PhotometricSpectrum,
                       objname: str = None,
                       mag_key = 'MAGSKY_APER_2',
                       magerr_key = 'MAGERR_APER_2',
                       filter_key = 'filter',
                       objname_key = 'objname',
                       visualize = False,
                       save = True,
                       save_path: str = None
                       ):
        """
        Load photometry and spectrum data from a PhotometrySpectrum object.
        
        Parameters:
            photspec (PhotometrySpectrum): A PhotometrySpectrum object containing photometry and spectrum data.
        
        Returns:
            pd.DataFrame: A DataFrame containing the photometry and spectrum data.
        """
        def _is_valid_mag(m):
            if m is None:
                return False
            try:
                mv = float(m)
            except Exception:
                return False
            if np.isnan(mv) or mv in (99.0, 999.0) or np.isinf(mv):
                return False
            return True
        
        def _abmag_to_flambda(wavelength: Union[float, np.ndarray],   # Å
            mag_ab: Union[float, np.ndarray],
            magerr_ab: Optional[Union[float, np.ndarray]] = None):
            """
            Convert AB magnitude to f_lambda [erg s^-1 cm^-2 Å^-1]

            Parameters
            ----------
            wavelength : float or array
                Effective wavelength in Å
            mag_ab : float or array
                AB magnitude
            magerr_ab : float or array, optional
                AB magnitude error

            Returns
            -------
            f_lambda : float or array
            f_lambda_err : float or array (if magerr_ab is given)
            """
            c_ang_per_s = 2.99792458e18  # Å / s

            mag_ab = np.asarray(mag_ab, dtype=float)
            wavelength = np.asarray(wavelength, dtype=float)

            f_nu = 10.0 ** (-0.4 * (mag_ab + 48.6))   # erg s^-1 cm^-2 Hz^-1
            f_lambda = f_nu * c_ang_per_s / wavelength**2

            if magerr_ab is None:
                return f_lambda

            magerr_ab = np.asarray(magerr_ab, dtype=float)
            frac = (np.log(10.0) / 2.5) * magerr_ab
            f_lambda_err = f_lambda * frac

            return f_lambda, f_lambda_err
            
        data_tbl = photspec.data
        data_df = data_tbl.to_pandas()
        
        if not mag_key in data_df.columns:
            raise ValueError(f"mag_key {mag_key} not found in data_df")
        if not magerr_key in data_df.columns:
            raise ValueError(f"magerr_key {magerr_key} not found in data_df")
        if not filter_key in data_df.columns:
            raise ValueError(f"filter_key {filter_key} not found in data_df")
        
        all_mags = data_df[mag_key].values
        all_magerrs = data_df[magerr_key].values
        valid_mask = ([_is_valid_mag(m) for m in all_mags] or [_is_valid_mag(m) for m in all_magerrs])
        valid_df = data_df[valid_mask]
        valid_df['wl'] = valid_df[filter_key].map(photspec.FILTER_PIVOT_WAVELENGTH_NM)*10
        if objname is None and objname_key in valid_df.columns:
            objname = valid_df[objname_key].values[0]
        

        valid_df = valid_df.sort_values(by = 'wl')
        mag = valid_df[mag_key].values
        magerr = valid_df[magerr_key].values
        wl = valid_df['wl'].values
        flux, fluxerr = _abmag_to_flambda(wl, mag, magerr)
        spec = Spectrum(wavelength = wl, flux = flux, fluxerr = fluxerr, flux_unit = 'flamb', wavelength_unit = 'AA')
        sed_df = pd.DataFrame({'wl': wl, 'flux': flux, 'fluxerr': fluxerr})
        if save:
            if save_path is None:
                save_dir = Path(self.config.save_dir) / objname
                save_path = save_dir / f'{objname}_{Time.now().isot}.txt'
            else:
                save_dir = save_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            sed_df.to_csv(save_path, index = False, sep = " ", header = False)
        if visualize:
            fig_ab, ax_ab = spec.show(show_flux_unit = 'ab')
            fig_nu, ax_nu = spec.show(show_flux_unit = 'fnu')
            fig_flamb, ax_flamb = spec.show(show_flux_unit = 'flamb')
            if save:
                fig_ab.savefig(save_path.replace('.txt', '_ab.png'))
                fig_nu.savefig(save_path.replace('.txt', '_nu.png'))
                fig_flamb.savefig(save_path.replace('.txt', '_flamb.png'))
        return sed_df, save_path
    
    def fit(self, 
            file_path: str,
            redshift: float = None):
        superfit = Superfit()
        
        superfit.parameters.object_to_fit = file_path
        
        spec_data = self._read_file(file_path)
        error_exists = False
        if spec_data.shape[1] > 2:
            error_exists = True
        
        if redshift is not None:
            superfit.parameters.use_exact_z = 1
            superfit.parameters.z_exact = redshift
        else:
            superfit.parameters.use_exact_z = 0
            superfit.parameters.z_exact = 0.0
        if error_exists:
            superfit.parameters.kind = 'included'
        else:
            superfit.parameters.kind = 'fixed'
            
        superfit.superfit()
        return superfit
    
    def _read_file(self, file_path: str):


        '''
        This function removes all entries beginning with '#' from a file with a header, keeping only the column
        
        data and saving it into a file.
        
        
        parameters
        ----------
        
        It takes one path (in the form of "/home/user/Dropbox/something") to pull and eliminate its header
        
        
        returns
        -------
        
        File without header.
        
        '''

        lines = [] 
        
        file = open(file_path,'r')
        
        lines = file.readlines()

        lines = [i for i in lines if i]

        lines = [i for i in lines if i[0].isalpha() == False and i[0] != '#' and i[0] != '%' and i[0] != '@']

        lines = [i for i in lines if i[0] != '\n']
        
        lines = [s.strip('\n') for s in lines] # remove empty lines
        
        lines = [s.replace('\n', '') for s in lines]  #replace with nothing

        columns = [] 
        
        for line in lines:
            ii = line.split()
            columns.append(ii)
            
        columns = np.array(columns)
        
        lam_floats  = [float(i) for i in columns[:,0]]
        flux_floats = [float(i) for i in columns[:,1]]
        if columns.shape[1] > 2:
            fluxerr_floats = [float(i) for i in columns[:,2]]
        else:
            fluxerr_floats = None

        #Check if lambda is in ascending order, reverse it if this is not the case

        if lam_floats[0] > lam_floats[-1]:
            lam_floats = list(reversed(lam_floats))
            flux_floats = list(reversed(flux_floats))

            if fluxerr_floats is not None:
                fluxerr_floats = list(reversed(fluxerr_floats))

        if fluxerr_floats is not None:
            spectrum = np.array([lam_floats, flux_floats, fluxerr_floats]).T
        else:
            spectrum = np.array([lam_floats, flux_floats]).T
            
        return spectrum




# %%
if __name__ == "__main__":
    import glob
    from astropy.io import ascii
    self = AlertClassifier()
    file_path = '/home/hhchoi1022/snal/data/2013fc/SN2013fc_2013-11-04_05-32-02_ESO-NTT_EFOSC2-NTT_PESSTO_SSDR1-4.csv'
#%%
if __name__ == "__main__":
    superfit = self.fit(file_path)
    print(superfit.parameters)
# %%
if __name__ == "__main__":
    ra = 260.034283333
    dec = -60.1566944444
    photspec.plot()
# %%
if __name__ == "__main__":
    tbl, file_path = self.ngsf_formatter(photspec, objname = objname, mag_key = mag_key, magerr_key = magerr_key, filter_key = filter_key, objname_key = objname_key, visualize = visualize, save = save)

# %%
if __name__ == "__main__":
    superfit = self.fit(file_path)
    print(superfit.parameters)
# %%
