#%%
from bridge.utils import HostGalaxyCatalog
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from astropy.coordinates import SkyCoord
from ezphot.helper import Helper
from astropy.table import Table
from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm

helper = Helper()
# %%
gcatalog = HostGalaxyCatalog(catalog_path = './glade_plus_updated.fits')
gcatalog_all = gcatalog.target_catalog
#%%
gcatalog_all = gcatalog_all[gcatalog_all['d_L'] < 400]

#%% Update redshift with NED LVS catalog
nedlvs_path = './NEDLVS_20250602.fits'
nedlvs_catalog = Table.read(nedlvs_path, format='fits')
nedlvs_catalog = nedlvs_catalog[nedlvs_catalog['DistMpc'] < 400]

gcatalog_coord = SkyCoord(gcatalog_all['RA'], gcatalog_all['Dec'], unit=(u.deg, u.deg))
nedcatalog_coord = SkyCoord(nedlvs_catalog['ra'], nedlvs_catalog['dec'], unit=(u.deg, u.deg))
gcatalog_idx, nedcatalog_idx, unmatched_idx = helper.cross_match(gcatalog_coord, nedcatalog_coord, max_distance_second = 30)

# Initialize new columns with None
gcatalog_all['z_NED'] = None
gcatalog_all['objname_NED'] = None
gcatalog_all['distance_NED'] = None
gcatalog_all['distance_unc_NED'] = None
gcatalog_all['distance_method_NED'] = None

# Assign only for matched indices
gcatalog_all['z_NED'][gcatalog_idx] = nedlvs_catalog['z'][nedcatalog_idx]
gcatalog_all['objname_NED'][gcatalog_idx] = nedlvs_catalog['objname'][nedcatalog_idx]
gcatalog_all['distance_NED'][gcatalog_idx] = nedlvs_catalog['DistMpc'][nedcatalog_idx]
gcatalog_all['distance_unc_NED'][gcatalog_idx] = nedlvs_catalog['DistMpc_unc'][nedcatalog_idx]
gcatalog_all['distance_method_NED'][gcatalog_idx] = nedlvs_catalog['DistMpc_method'][nedcatalog_idx]
#%%
gcatalog_nearby = gcatalog_all[gcatalog_all['d_L'] < 400]
#%%
import numpy as np


import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
x = np.asarray(gcatalog_nearby['distance_NED'], dtype=float)
y = np.asarray(gcatalog_nearby['d_L'], dtype=float)

mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]

# ------------------------------------------------------------
# Fit y = a x (intercept = 0)
# ------------------------------------------------------------
a = np.sum(x * y) / np.sum(x**2)

x_fit = np.linspace(0, 400, 500)
y_fit = a * x_fit

# residuals = y - a * x
residuals = y-x
rms = np.sqrt(np.mean(residuals**2))

# ------------------------------------------------------------
# Distance bins
# ------------------------------------------------------------
bins = np.arange(0, 400, 50)

from astropy.stats import sigma_clipped_stats
x_med = []
y_err = []
for lo, hi in zip(bins[:-1], bins[1:]):
    med_x = np.mean([lo, hi])
    m = (x >= lo) & (x < hi)
    if np.sum(m) >= 5:
        median, mean, std = sigma_clipped_stats(residuals[m], sigma=3.0, maxiters=3)
        x_med.append(med_x)
        y_err.append(std)
x_med = np.array(x_med)
y_fit_bins = a * x_med
y_err = np.array(y_err)
# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(6, 6))

# Raw scatter
plt.scatter(x, y, alpha=0.02, s=10)

# Binned median ± std
plt.errorbar(
    x_med, y_fit_bins,
    yerr=y_err,
    fmt='o', color='r', ecolor='r',
    elinewidth=1.5, capsize=4,
    label='Median ± std (binned)',
    alpha=0.5
)

# ---- TEXT ANNOTATION (FIXED) ----
for xm, ym, ye in zip(x_med, y_fit_bins, y_err):
    plt.text(
        xm, ym + 0.04 * ym,          # small vertical offset
        rf'$\pm$ {ye:.1f} Mpc',
        fontsize=12,
        ha='center',
        va='bottom',
        color='r'
    )

# Best-fit line
plt.plot(
    x_fit, y_fit, 'b-', lw=2,
    label=f'Fit: y = {a:.4f} x'
)

plt.xlim(0, 400)
plt.ylim(0, 400)
plt.xlabel('distance_NED [Mpc]', fontsize = 14)
plt.ylabel('distance_GLADE [Mpc]', fontsize = 14)
plt.legend()

plt.tight_layout()
plt.show()

# %% Cut nearby and bright galaxies
gcatalog.cut_catalog(z_cmb = '<0.1')
catalog_nearby = gcatalog.target_catalog
from astroquery.ned import Ned

#%% Run this code with multiprocess
def process_galaxy(source):
    if source['objname_NED'] is not None:
        return source["GLADE_no"], source['objname_NED']
    ra = source["RA"]
    dec = source["Dec"]
    coord = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame="icrs")

    try:
        result = Ned.query_region(coord, radius=120/3600 * u.deg)

        result.sort('Separation')

        result_pd = result.to_pandas()

        # Keep only galaxies
        result_galaxy = result_pd#result_pd[result_pd["Type"] == "G"].copy()

        # Extract only the numeric part of "Magnitude and Filter"
        result_galaxy["mag_value"] = (
            result_galaxy["Magnitude and Filter"]
            .astype(str)
            .str.extract(r"([-+]?\d*\.?\d+)")
        )

        # Convert to float, coercing errors to NaN
        result_galaxy["mag_value"] = pd.to_numeric(
            result_galaxy["mag_value"], errors="coerce"
        )

        # Drop rows where mag_value could not be parsed
        #result_galaxy = result_galaxy.dropna(subset=["mag_value"])

        # Sort by magnitude
        result_galaxy_bright = result_galaxy#result_galaxy.sort_values(by="mag_value")

        # Take the brightest (lowest mag) if available
        if not result_galaxy_bright.empty:
            gal_name = result_galaxy_bright.iloc[0]["Object Name"]
        else:
            gal_name = None

        return source["GLADE_no"], gal_name

    except Exception as e:
        return source["GLADE_no"], None
    
#%%
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

MAX_THREADS = 256   # 8~20 is usually optimal for astroquery

with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    results = list(tqdm(
        executor.map(process_galaxy, catalog_nearby),
        total=len(catalog_nearby)
    ))
#%%
all_names = [name[1] for name in results]
catalog_nearby['objname_NED_query'] = all_names
catalog_nearby['z_NED'] = catalog_nearby['z_NED'].astype(float)
catalog_nearby['objname_NED'] = catalog_nearby['objname_NED'].astype(str)
catalog_nearby['distance_NED'] = catalog_nearby['distance_NED'].astype(float)
catalog_nearby['distance_unc_NED'] = catalog_nearby['distance_unc_NED'].astype(float)
catalog_nearby['distance_method_NED'] = catalog_nearby['distance_method_NED'].astype(str)
catalog_nearby['objname_NED_query'] = catalog_nearby['objname_NED_query'].astype(str)
# catalog_nearby_bkg = catalog_nearby.copy()
# catalog_nearby.remove_column('z_NED')
# catalog_nearby.write('glade_plus_updated_NED_query_temp.fits', format='fits')
#%%
catalog_nearby = Table.read('glade_plus_updated_NED_query_temp.fits', format='fits')
for row in catalog_nearby:
    if row['objname_NED'] != row['objname_NED_query']:
        print(row['objname_NED'], row['objname_NED_query'], row['GLADE_no'])
#%%
from concurrent.futures import ThreadPoolExecutor, as_completed
from astropy.coordinates import SkyCoord
from astroquery.ned import Ned
from astropy import units as u
from tqdm import tqdm
#%%

def fetch_diameter(source):
    """
    Worker: Query NED for a single source and extract diameter information.
    Returns a dictionary of results.
    """
    out = dict()
    out['GLADE_no'] = source['GLADE_no']
    out['name_from_query'] = source['objname_NED_query']
    out['name_from_ned'] = source['objname_NED']

    major_axis = None
    minor_axis = None

    # Choose name to search
    name_from_query = out['name_from_query']
    name_from_ned = out['name_from_ned']

    if (str(name_from_ned) == 'None') and (str(name_from_query) == 'None'):
        return out  # nothing to do

    # Prefer name_from_ned if exists
    name_to_search = str(name_from_ned)
    if name_to_search == 'None':
        name_to_search = str(name_from_query)

    try:
        tbl = Ned.get_table(name_to_search, table='diameters')

        # Use only arcsec or arcmin
        tbl = tbl[(tbl['Major Axis Unit'] == 'arcmin') | (tbl['Major Axis Unit'] == 'arcsec')]

        # Prefer "Outer" measurement
        tbl_outer = tbl[['Outer' in val for val in tbl['Measured Quantity']]]

        if len(tbl_outer) > 0:
            row = tbl_outer[0]
        elif len(tbl) > 0:
            row = tbl[0]
        else:
            return out

        # Major axis
        major_axis = row['Major Axis']
        if row['Major Axis Unit'] == 'arcmin':
            major_axis = major_axis * u.arcmin
        else:
            major_axis = major_axis * u.arcsec

        # Minor axis
        minor_axis = row['Minor Axis']
        if row['Minor Axis Unit'] == 'arcmin':
            minor_axis = minor_axis * u.arcmin
        else:
            minor_axis = minor_axis * u.arcsec

    except Exception as e:
        out['error'] = str(e)

    out['major_axis'] = major_axis
    out['minor_axis'] = minor_axis
    
    return out

#%%
# ===========================================================
# RUN MULTITHREADED
# ===========================================================
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
catalog_nearby_to_trigger = catalog_nearby
MAX_THREADS = 32   # 8~20 is usually optimal for astroquery
print('Expected finish time: ', len(catalog_nearby_to_trigger)/1000*50/3600, 'hours')
with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    results = list(tqdm(
        executor.map(fetch_diameter, catalog_nearby_to_trigger),
        total=len(catalog_nearby_to_trigger)
    ))

#%%
#%%
all_diameter = []
all_major_axis = []
all_minor_axis = []
for value in tqdm(results):
    major_axis = None
    minor_axis = None
    if 'major_axis' in value:
        major_axis = value['major_axis']
    if 'minor_axis' in value:
        minor_axis = value['minor_axis']
    if major_axis is None and minor_axis is None:
        all_diameter.append(None)
        all_major_axis.append(None)
        all_minor_axis.append(None)
        continue
    try:
        major_axis = major_axis.to('arcsec').value
    except:
        major_axis = np.nan
    try:
        minor_axis = minor_axis.to('arcsec').value
    except:
        minor_axis = np.nan
        
    all_major_axis.append(major_axis)
    all_minor_axis.append(minor_axis)
    all_diameter.append(np.nanmax([major_axis, minor_axis]))
#%%
catalog_nearby['diameter'] = all_diameter
catalog_nearby['major_axis'] = all_major_axis
catalog_nearby['minor_axis'] = all_minor_axis
#%% Join with the catalog_all
from astropy.table import join
mask_missing = gcatalog_all['GLADE_no'].mask if hasattr(gcatalog_all['GLADE_no'], 'mask') else pd.isnull(gcatalog_all['GLADE_no'])
gcatalog_all_clean = gcatalog_all[~mask_missing]
catalog_all = join(gcatalog_all_clean, catalog_nearby, keys=['GLADE_no'], join_type='left')
# Example: drop all columns ending with "_2"
for col in catalog_all.colnames:
    if col.endswith("_2"):
        catalog_all.remove_column(col)
    if col.endswith('_1'):
        catalog_all.rename_column(col, col.replace('_1', ''))

#%%
catalog_all.rename_column('diameter', 'diameter_NED')
catalog_all.rename_column('major_axis', 'major_axis_NED')
catalog_all.rename_column('minor_axis', 'minor_axis_NED')
#%%
import numpy as np
from astropy.table import MaskedColumn

def clean_to_float(col):
    # Convert entries to float, replacing invalid with NaN
    arr = []
    for v in col:
        try:
            arr.append(float(v))
        except (ValueError, TypeError):
            arr.append(np.nan)
    arr = np.array(arr, dtype=float)
    # Return as MaskedColumn with a proper float fill value
    return MaskedColumn(arr, dtype=float, fill_value=np.nan)

for col in ['z_NED', 'distance_NED', 'distance_unc_NED',
            'diameter_NED', 'major_axis_NED', 'minor_axis_NED']:
    catalog_all[col] = clean_to_float(catalog_all[col])

# Text columns: force fixed-length unicode
for col in ['objname_NED', 'distance_method_NED', 'objname_NED_query']:
    catalog_all[col] = catalog_all[col].astype('U50')
#%%
catalog_all.write('glade_plus_updated_NED.fits', format='fits')
#%%
catalog_all['z_final'] = catalog_all['z_cmb']
z_mask = ~np.isnan(catalog_all['z_NED'])
catalog_all['z_final'][z_mask] = catalog_all['z_NED'][z_mask]
catalog_all['z_flag_NED'] = z_mask
#%%
redshift_glade = catalog_all['z_cmb']
distance_glade = cosmo.angular_diameter_distance(redshift_glade)
catalog_all['distance_final'] = distance_glade
distance_mask = ~np.isnan(catalog_all['distance_NED'])
catalog_all['distance_final'][distance_mask] = catalog_all['distance_NED'][distance_mask]
catalog_all['distance_flag_NED'] = distance_mask
#%%
default_size_galaxy = 10 * u.kpc                 # 예: 물리적 크기 15 kpc
diameter_glade = ((default_size_galaxy / distance_glade) * u.rad).to(u.arcsec)
catalog_all['diameter_final'] = diameter_glade
diameter_mask = ~np.isnan(catalog_all['diameter_NED'])
catalog_all['diameter_final'][diameter_mask] = catalog_all['diameter_NED'][diameter_mask] 
catalog_all['diameter_flag_NED'] = diameter_mask

#%%
catalog_all.write('HostGalaxyCatalog_d400M_s10K_f1.fits', format='fits', overwrite=True)



#%%

catalog_all = Table.read('/home/hhchoi1022/.bridge/HostGalaxyCatalog_d400M_s10K_f1.fits', format='fits')
# %%
catalog_all['diameter_cut_flag'] = False
catalog_all['diameter_cut_flag'][(catalog_all['diameter_flag_NED'] == False) & (catalog_all['diameter_final'] > 60)] = True
catalog_all['diameter_final'][(catalog_all['diameter_flag_NED'] == False) & (catalog_all['diameter_final'] > 60)] = 60

#%%
catalog_all.write('/home/hhchoi1022/.bridge/HostGalaxyCatalog_d400M_s10K_f1_cut60.fits', format = 'fits')
# %%
