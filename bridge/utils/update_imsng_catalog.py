#%%
from bridge.alertquerier import TNSQuerier
from astropy.io import ascii
#%%
# Example usag
# if len(sys.argv) != 2:
#     print("Usage: python tnsqurier.py <configfile>")
#     sys.exit(1)

# config_file_path = sys.argv[1]
self = TNSQuerier()
glade_tbl = ascii.read('../utils/close_catalog.ascii_fixed_width', format = 'fixed_width')
# all_transients = []
# for row in close_tbl:
#     url_parameters = {"ra": str(row['RA']), "decl": str(row['Dec']), "radius": str(row['diameter_final'])}
#     file_ = self.search_tns(url_parameters = url_parameters, alert_save_dir = self.config['alert_save_dir'])
#     if file_ is not None:
#         data = ascii.read(file_, format ='csv')
#         all_transients.append(data['Name'].tolist())
#     else:
#         all_transients.append([])

# len_all = [len(transient) for transient in all_transients]
# close_tbl['num_transients'] = len_all
# close_tbl['transients'] = all_transients

#%%
imsng_tbl = ascii.read('../utils/alltarget_prior2.dat', format = 'fixed_width')
# all_transients = []
# for row in close_tbl:
#     url_parameters = {"ra": str(row['ra']), "decl": str(row['dec']), "radius": str(600)}
#     file_ = self.search_tns(url_parameters = url_parameters, alert_save_dir = self.config['alert_save_dir'])
#     if file_ is not None:
#         data = ascii.read(file_, format ='csv')
#         all_transients.append(data['Name'].tolist())
#     else:
#         all_transients.append([])

# len_all = [len(transient) for transient in all_transients]
# close_tbl['num_transients'] = len_all
# close_tbl['transients'] = all_transients
# %%
import numpy as np
glade_tbl_with_transients = glade_tbl[glade_tbl['num_transients'] > 0]
np.mean(glade_tbl_with_transients['num_transients'])
# %%
imsng_tbl_with_transients = imsng_tbl[imsng_tbl['num_transients'] > 0]
np.mean(imsng_tbl_with_transients['num_transients'])
#%%
from ezphot.helper import Helper
helper = Helper()
#%%
from astropy.coordinates import SkyCoord
import astropy.units as u
glade_coord = SkyCoord(glade_tbl_with_transients['RA'], glade_tbl_with_transients['Dec'], unit=(u.deg, u.deg))
imsng_coord = SkyCoord(imsng_tbl_with_transients['ra'], imsng_tbl_with_transients['dec'], unit=(u.hourangle, u.deg))
glade_idx, imsng_idx, unmatched_idx = helper.cross_match(glade_coord, imsng_coord, max_distance_second = 600)
#%%


from astropy.table import Table
updated_imsng_tbl = Table()
updated_imsng_tbl['objname'] = imsng_tbl_with_transients['obj']
coord = SkyCoord(imsng_tbl_with_transients['ra'], imsng_tbl_with_transients['dec'], unit=(u.hourangle, u.deg))
coord_str_ra  = coord.ra.to_string(unit=u.hour, sep=":", precision=3, pad=True, alwayssign=False)
coord_str_dec = coord.dec.to_string(unit=u.deg, sep=":", precision=2, pad=True, alwayssign=True)
updated_imsng_tbl['ra'] = coord_str_ra
updated_imsng_tbl['dec'] = coord_str_dec
updated_imsng_tbl['ra_deg'] = np.array(coord.ra.value).round(4)
updated_imsng_tbl['dec_deg'] = np.array(coord.dec.value).round(4)
updated_imsng_tbl['distance'] = np.array(imsng_tbl_with_transients['dist']).round(1)
updated_imsng_tbl['maxaxis'] = np.array(imsng_tbl_with_transients['maxaxis']).round(1)
updated_imsng_tbl['minaxis'] = np.array(imsng_tbl_with_transients['minaxis']).round(1)
updated_imsng_tbl['num_transients'] = imsng_tbl_with_transients['num_transients']
updated_imsng_tbl['transients'] = imsng_tbl_with_transients['transients']
#%% Attach glade catalog to updated imsng catalog
glade_tbl_with_transients[unmatched_idx]
updated_glade_tbl = Table()

updated_glade_tbl['objname'] = glade_tbl_with_transients['objname_NED'][unmatched_idx]
coord = SkyCoord(glade_tbl_with_transients['RA'][unmatched_idx], glade_tbl_with_transients['Dec'][unmatched_idx], unit=(u.deg, u.deg))
coord_str_ra  = coord.ra.to_string(unit=u.hour, sep=":", precision=3, pad=True, alwayssign=False)
coord_str_dec = coord.dec.to_string(unit=u.deg, sep=":", precision=2, pad=True, alwayssign=True)
updated_glade_tbl['ra'] = coord_str_ra
updated_glade_tbl['dec'] = coord_str_dec
updated_glade_tbl['ra_deg'] = np.array(coord.ra.value).round(4)
updated_glade_tbl['dec_deg'] = np.array(coord.dec.value).round(4)
updated_glade_tbl['distance'] = np.array(glade_tbl_with_transients['distance_final'][unmatched_idx]).round(1)
updated_glade_tbl['maxaxis'] = np.array(glade_tbl_with_transients['major_axis_NED'][unmatched_idx]).round(1)
updated_glade_tbl['minaxis'] = np.array(glade_tbl_with_transients['minor_axis_NED'][unmatched_idx]).round(1)
updated_glade_tbl['num_transients'] = glade_tbl_with_transients['num_transients'][unmatched_idx]
updated_glade_tbl['transients'] = glade_tbl_with_transients['transients'][unmatched_idx]
#%%
from astropy.table import vstack
updated_tbl = vstack([updated_imsng_tbl, updated_glade_tbl])
updated_tbl.sort('ra')
updated_tbl.write('../utils/updated_imsng_catalog.dat', format = 'ascii.fixed_width', overwrite = True)
# %%
tbl = ascii.read('../utils/updated_imsng_catalog.dat', format = 'fixed_width')
# %%
tbl.sort('num_transients', reverse = True)
# %%
import matplotlib.pyplot as plt
plt.hist(tbl['num_transients'], bins = 20)

# %%
tbl['priority'] = np.zeros(len(tbl))
high_priority_idx = tbl['num_transients'] > 2
medium_priority_idx = tbl['num_transients'] == 2
low_priority_idx = tbl['num_transients'] == 1
tbl['priority'][high_priority_idx] = 1
tbl['priority'][medium_priority_idx] = 2
tbl['priority'][low_priority_idx] = 3
transient_list = tbl['transients']
tbl.remove_column('transients')
tbl.add_column(transient_list, name = 'transients')
# %%
tbl.write('../utils/updated_imsng_catalog_with_priority.dat', format = 'ascii.fixed_width', overwrite = True)
# %%
