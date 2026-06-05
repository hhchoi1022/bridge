#%%
from pathlib import Path
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp
import numpy as np
from collections import defaultdict
import math
from s2sphere import Cap, LatLng, CellUnion, RegionCoverer, Angle
import matplotlib.pyplot as plt
import json
from astropy.time import Time
#%%
class HostGalaxyCatalog:
    def __init__(self, catalog_path: str = f'{Path(__file__).parent}/hostgalaxycatalog.fits'):
        if Path(catalog_path).exists():
            self.hostgalaxycatalog_path = catalog_path
        else:
            raise FileNotFoundError(f"Host galaxy catalog file not found at {catalog_path}")
        self.catalog = None
        self.load_catalog()
        self.NSIDE = 16384  # ~13 arcsec pixel size
        if 'healpix' not in self.catalog.columns:
            self._add_healpix_index()
        self.target_catalog = self.catalog
        self._build_hp_index()
    
    def cut_catalog(self, **kwargs: dict):
        print(f'Start cutting catalog... with kwargs = {kwargs}')
        raw_catalog = self.catalog.copy()
        for cut_key, cut_value in kwargs.items():

            if cut_key not in raw_catalog.columns:
                print(f"WARNING: {cut_key} not in catalog columns. Skipping...")
                continue
            
            print(f'Cutting {cut_key} with {cut_value}...')

            if '<' in cut_value:
                cut_value = cut_value.replace('<', '')
                raw_catalog = raw_catalog[raw_catalog[cut_key] < float(cut_value)]
            elif '>' in cut_value:
                cut_value = cut_value.replace('>', '')
                raw_catalog = raw_catalog[raw_catalog[cut_key] > float(cut_value)]
            elif '=<' in cut_value:
                cut_value = cut_value.replace('=<', '')
                raw_catalog = raw_catalog[raw_catalog[cut_key] <= float(cut_value)]
            elif '>=' in cut_value:
                cut_value = cut_value.replace('>=', '')
                raw_catalog = raw_catalog[raw_catalog[cut_key] >= float(cut_value)]
            elif '=' in cut_value:
                cut_value = cut_value.replace('=', '')
                raw_catalog = raw_catalog[raw_catalog[cut_key] == str(cut_value)]
            else:
                print(f"WARNING: {cut_key} not in catalog columns. Skipping...")
                continue
            
        self.target_catalog = raw_catalog
        print(f'Nearby catalog cut with {len(self.target_catalog)} sources.')
        self._build_hp_index()

    def search_catalog(self, ra: float, dec: float, radius_deg: float):
        radius_deg = radius_deg * u.deg
        theta = np.deg2rad(90 - dec)
        phi = np.deg2rad(ra)
        vec = hp.ang2vec(theta, phi)
        pix = hp.query_disc(self.NSIDE, vec, radius_deg.to(u.rad).value, nest=True)

        indices = [i for hpix in pix for i in self.hp_index.get(hpix, [])]
        subset = self.target_catalog[indices]
        
        # precise separation filter
        from astropy.coordinates import SkyCoord
        coords = SkyCoord(subset['RA'], subset['Dec'], unit='deg')
        target = SkyCoord(ra, dec, unit='deg')
        sep = coords.separation(target)
        mask = sep < radius_deg
        result = subset[mask].copy()
        result['separation'] = sep[mask].to(u.arcsec)
        return result

    def load_catalog(self):
        catalog_path = self.hostgalaxycatalog_path
        print(f'Loading GLADE catalog from {catalog_path} ...')
        self.catalog = Table.read(catalog_path, format='fits')
        #self.catalog = Table.read('./glade_plus_updated_NED.fits', format='fits')
        print(f'GLADE catalog loaded with {len(self.catalog)} sources.')
        self.coords = SkyCoord(self.catalog['RA'], self.catalog['Dec'], unit=(u.deg, u.deg))
        
    def _add_healpix_index(self):
        print('Adding healpix index to GLADE catalog...')
        theta = np.deg2rad(90 - self.catalog['Dec'])
        phi = np.deg2rad(self.catalog['RA'])
        self.catalog['healpix'] = hp.ang2pix(self.NSIDE, theta, phi, nest=True)
        print('Healpix index added to GLADE catalog.')
        
    def _build_hp_index(self):
        print("Building healpix index dictionary...")
        self.hp_index = defaultdict(list)
        for i, hpix in enumerate(self.target_catalog['healpix']):
            self.hp_index[hpix].append(i)
        print("Healpix index built.")
        
    def show_coverage(self):

        print("Checing sky coverage... It will take a while...")
        if len(self.target_catalog) > 10000000:
            print("Too many targets (>10000000). Skipping... Please cut the catalog first with self.cut_catalog()")
            return
        # 예: ra, dec (deg), 반지름 theta_arcsec
        ra_list = self.target_catalog['RA']  # degrees
        dec_list = self.target_catalog['Dec']   # degrees
        theta_list = self.target_catalog['diameter_final']  # arcsec



        coverer = RegionCoverer()
        coverer.min_level = 10     # 분해능 (0 coarse → 30 finest)
        coverer.max_level = 14     # 더 정밀하게 하려면 ↑
        coverer.max_cells = 1000   # 한 region을 덮는 최대 cell 수

        all_cells = []
        from tqdm import tqdm
        for ra, dec, th_arcsec in tqdm(zip(ra_list, dec_list, theta_list), total=len(ra_list)):
            if not np.isfinite(th_arcsec) or th_arcsec <= 0:
                continue
            th_arcsec = abs(th_arcsec)
            if th_arcsec > 3600:
                continue
            try:
                center = LatLng.from_degrees(dec, ra).to_point()
                radius = Angle.from_degrees(th_arcsec / 3600.0)  # arcsec→deg
                cap = Cap.from_axis_angle(center, radius)
                cells = coverer.get_covering(cap)
                all_cells.extend(cells)
            except Exception as e:
                print(f"Error: {e}")
                continue

        # 합집합

        union = CellUnion(all_cells)
        union.normalize()

        # 면적 (단위: steradian)
        from s2sphere import Cell, CellId

        area_sr = 0.0
        for cid in union.cell_ids():
            cell = Cell(cid)
            area_sr += cell.exact_area()   # 또는 approx_area()

        sky_fraction = area_sr / (4 * math.pi)
        area_deg2 = area_sr * (180.0 / math.pi)**2

        print(f"Sky fraction ≈ {sky_fraction:.6f}")
        print("Plotting sky coverage...")

        nside = 4096
        npix = hp.nside2npix(nside)
        coverage = np.zeros(npix, dtype=float)

        for ra, dec, radius_arcsec in zip(self.target_catalog['RA'], self.target_catalog['Dec'], self.target_catalog['diameter_final']):
            if not np.isfinite(radius_arcsec) or radius_arcsec <= 0:
                continue  # skip bad values
            
            radius_arcsec = abs(radius_arcsec)  # enforce positive
            if radius_arcsec > 3600:  # skip crazy >1° radii
                continue

            theta = np.radians(90.0 - dec)   # colatitude
            phi   = np.radians(ra)           # longitude
            radius_rad = np.radians(radius_arcsec / 3600.0)

            vec = hp.ang2vec(theta, phi)
            ipix = hp.query_disc(nside, vec, radius_rad)
            coverage[ipix] = 1

        hp.mollview(coverage, title = rf"Sky coverage: {area_deg2:.2f} $deg^2$",
                    unit="coverage", cmap="viridis", rot=180)
        hp.graticule()
        plt.show()


# %% Usage
if __name__ == "__main__":   
    hostgalaxycatalog = HostGalaxyCatalog()
    catalog_tbl = hostgalaxycatalog.search_catalog(ra=180.0, dec=0.0, radius_deg=1.0)
    print(catalog_tbl)

#%% Check aperture for each galaxy near ra = 100, dec = -20 with radius = 1deg
if __name__ == "__main__":
    catalog_tbl = hostgalaxycatalog.search_catalog(ra=100.0, dec=-20.0, radius_deg=1.0)
    catalog_tbl.sort('diameter_final', reverse=True)
    print(catalog_tbl['diameter_final']) # Diameter in arcsec
    print(catalog_tbl['distance_final']) # Distance in Mpc
    
# %% Cut the catalog with the distance < 200Mpc
if __name__ == "__main__":
    hostgalaxycatalog.cut_catalog(distance_final = '<200')
    catalog_tbl_nearby = hostgalaxycatalog.search_catalog(ra=100.0, dec=-20.0, radius_deg=1.0)
    print(catalog_tbl_nearby['diameter_final']) # Diameter in arcsec
    print(catalog_tbl_nearby['distance_final']) # Distance in Mpc
# %% Show coverage 
if __name__ == "__main__":
    hostgalaxycatalog.cut_catalog(distance_final = '<50')
    hostgalaxycatalog.show_coverage()
# %%
