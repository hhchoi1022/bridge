
#%%
import matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt
# plt.ion()
#%%
from bridge.connector import SQLConnector
from bridge.alertquerier import ALERCEQuerier
from bridge.alertquerier import FINKQuerier
from astropy.time import Time
# %%
connector = SQLConnector()
lsst_fink_tbl = connector.get_data(tbl_name = 'lsst_fink', select_key = '*')
lsst_alerce_tbl = connector.get_data(tbl_name = 'lsst_alerce', select_key = '*')
#%% Check coverage of the DDFs 
import numpy as np
from bridge.connector import SDSConnector

ra_list = np.array([9.45, 35.57, 52.98, 150.11, 58.9, 63.6, 187.4])
dec_list = np.array([-44.02, -4.82, -28.12, 2.23, -49.32, -47.6, 8.0])
name_list = np.array(['ELAISS1', 'XMM_LSS', 'ECDFS', 'COSMOS', 'EDFS_a', 'EDFS_b', 'M49'])
sds_connector = SDSConnector()
all_tiles_dict = dict()
for ra, dec, name in zip(ra_list, dec_list, name_list):
    tile_tbl, matched_indices, _ = sds_connector.find_overlapping_tiles(list_ra = [ra], list_dec = [dec], list_name = [name], list_aperture = 3.5, visualize = True, save_fig = True, match_tolerance_minutes = 3, fraction_overlap_lower = 0.2)
    all_tiles_dict[name] = tile_tbl['id'].tolist()

# %% 1. Check the statistics of the alerts

# LSST_FINK: Receiving topics: ['fink_sn_near_galaxy_candidate_lsst', 'fink_extragalactic_lt20mag_candidate_lsst', 'fink_early_snia_candidate_lsst', 'fink_in_tns_lsst', 'fink_extragalactic_new_candidate_lsst']
lsst_fink_df = lsst_fink_tbl.to_pandas()
import pandas as pd
from astropy.time import Time
# MJD → datetime
lsst_fink_df['midpoint_datetime'] = Time(lsst_fink_df['diaSource_midpointMjdTai'], format='mjd').to_datetime()
# ISOT → datetime
lsst_fink_df['broker_datetime'] = pd.to_datetime(lsst_fink_df['brokerEndProcessTimestamp'])

lsst_fink_df['midpoint_date'] = lsst_fink_df['midpoint_datetime'].dt.date
lsst_fink_df['broker_date'] = lsst_fink_df['broker_datetime'].dt.date
latency = (
    lsst_fink_df['broker_datetime'] - lsst_fink_df['midpoint_datetime']
).dt.total_seconds()

lsst_fink_df['latency_sec'] = latency

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

plt.style.use('seaborn-v0_8-whitegrid')

# 전체 font size 설정
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

df = lsst_fink_df.copy()

# -----------------------------
# Daily statistics
# -----------------------------
midpoint_daily = df.groupby('midpoint_date').size()
broker_daily = df.groupby('broker_date').size()

# -----------------------------
# latency filtering
# -----------------------------
latency_mask = df['broker_datetime'] >= pd.Timestamp("2026-03-01")
latency = df.loc[latency_mask, 'latency_sec'].dropna()

# -----------------------------
# Plot
# -----------------------------
fig = plt.figure(figsize=(14,10))
gs = fig.add_gridspec(3,1, hspace=0.8)

# midpoint
ax1 = fig.add_subplot(gs[0])
ax1.plot(midpoint_daily.index, midpoint_daily.values, marker='o', lw=2)
ax1.set_yscale('log')
ax1.set_ylabel("Number of Alerts")
ax1.set_title("Daily Alerts (diaSource_midpointMjdTai)")
ax1.tick_params(axis='x', rotation=45)

# broker
ax2 = fig.add_subplot(gs[1])
ax2.plot(broker_daily.index, broker_daily.values, marker='o', lw=2, color='tab:orange')
ax2.set_yscale('log')
ax2.set_ylabel("Processed Alerts")
ax2.set_title("Daily Alerts Processed by Broker")
ax2.tick_params(axis='x', rotation=45)

# latency
ax3 = fig.add_subplot(gs[2])
ax3.hist(latency, bins=50, alpha=0.8)
ax3.set_yscale('log')

median_latency = np.median(latency)

ax3.axvline(
    median_latency,
    linestyle='--',
    lw=2,
    label=f"median = {median_latency:.1f} s"
)

ax3.set_xlabel("Latency (seconds)")
ax3.set_ylabel("Counts")
ax3.set_title("Broker Processing Latency (after 2026-03-01)")
ax3.legend()

plt.show()

# %% 2. Check variable sources in the alerts
lsst_fink_df = lsst_fink_tbl.to_pandas()
#%%
from astropy.table import Table
lsst_fink_df = lsst_fink_df.sort_values("diaSource_midpointMjdTai", ascending=False)
lsst_fink_df_object = (lsst_fink_df.drop_duplicates(subset="diaObject_diaObjectId"))
lsst_fink_tbl_object = Table.from_pandas(lsst_fink_df_object)
#%%
lsst_fink_tbl_object['diaSource_apMag'] = -2.5* np.log10(lsst_fink_tbl_object['diaSource_apFlux']) + 31.4
lsst_fink_tbl_object['diaSource_apMagErr'] = 2.5 / np.log(10) * lsst_fink_tbl_object['diaSource_apFluxErr'] / lsst_fink_tbl_object['diaSource_apFlux']
lsst_fink_tbl_object['diaSource_psfMag'] = -2.5* np.log10(lsst_fink_tbl_object['diaSource_psfFlux']) + 31.4
lsst_fink_tbl_object['diaSource_psfMagErr'] = 2.5 / np.log(10) * lsst_fink_tbl_object['diaSource_psfFluxErr'] / lsst_fink_tbl_object['diaSource_psfFlux']
lsst_fink_tbl_object.sort('diaSource_psfMag')
#%%

#%%
from bridge.alertfilter import LSST_Fink_Filter
filter = LSST_Fink_Filter()
#%%
lsst_fink_tbl_object_filtered = filter.apply_catalog_filter(lsst_fink_tbl_object)
#%%
lsst_fink_tbl_object_hg_filtered = filter.apply_hostgalaxy_filter(lsst_fink_tbl_object_filtered)
#%%
# import time
# ras = lsst_fink_tbl_object_hg_filtered_2['diaSource_ra']
# decs = lsst_fink_tbl_object_hg_filtered_2['diaSource_dec']
# objname = lsst_fink_tbl_object_hg_filtered_2['diaObject_diaObjectId']
# ra = ras[-6]
# dec = decs[-6]
# for ra, dec in zip(ras, decs):
#     tbl_hg = filter.hostgalaxycatalog.match_host(ra, dec, max_dell = 2.5, plot = True, search_radius_arcsec = 900)
#     time.sleep(0.5)
# %%
plt.figure(figsize=(10,5))

plt.hist(
    lsst_fink_tbl_object['diaObject_nDiaSources'],
    bins=100,
    histtype='step',
    linewidth=2,
    label='Original'
)

plt.hist(
    lsst_fink_tbl_object_filtered['diaObject_nDiaSources'],
    bins=100,
    histtype='step',
    linewidth=2,
    color='red',
    label='Filtered'
)

# plt.yscale('log')

plt.xlabel("Number of diaSources per diaObject")
plt.ylabel("Counts")
plt.title("Effect of Catalog Filtering")

plt.legend()
plt.grid(alpha=0.3)

plt.show()
# %%

import matplotlib.pyplot as plt

alercequerier = ALERCEQuerier('lsst')

band_color = {
    'u': 'purple',
    'g': 'green',
    'r': 'red',
    'i': 'orange',
    'z': 'brown',
    'y': 'black'
}

def plot_lightcurve(objid, DM=None, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))

    detections = alercequerier.query_detections(object_id=objid)

    if len(detections) == 0:
        return

    detections.sort("mjd")

    detections['magpsf'] = -2.5 * np.log10(detections['psfFlux']) + 31.4
    detections['sigmapsf'] = 2.5 / np.log(10) * detections['psfFluxErr'] / detections['psfFlux']
    detections['sigmapsf'] = np.where(detections['sigmapsf'] < 0, 0, detections['sigmapsf'])

    detections_groups = detections.group_by("band_name").groups

    for det_band in detections_groups:

        band = det_band["band_name"][0]

        if len(det_band) == 0:
            continue

        ax.errorbar(
            det_band["mjd"],
            det_band["magpsf"],
            yerr=det_band["sigmapsf"],
            fmt='o',
            color=band_color.get(band, "gray"),
            label=band
        )

    ax.axvline(Time.now().mjd, color='black', linestyle='--')

    ax.invert_yaxis()

    # ax.set_xlabel("MJD")
    ax.set_ylabel("Apparent magnitude")
    ax.tick_params(axis = 'both', labelsize=11)

    ax.set_title(str(objid), fontsize=12)

    # -------------------------
    # Absolute magnitude axis
    # -------------------------
    if DM is not None:

        ax2 = ax.twinx()

        y1, y2 = ax.get_ylim()

        ax2.set_ylim(y1 - DM, y2 - DM)

        # ax2.set_ylabel("Absolute magnitude")
        ax2.tick_params(axis = 'both', labelsize=11)

    if ax is None:
        plt.tight_layout()
        plt.show()
#%%
objname = 170094456357257301 # Type Ia
# objname = 170068070078349344 # Type Ia?
# objname = 170068070008094735 # Bright, but too slowly rising?
# objname = 170059315245744213
# objname = 170059283425656877
# objname = 170094456349392906 # Fading
# objname = 170107660764446767 # Fading
# objname = 170028488668479507 # mag 22?
# objname = 170028486213238822 # mag 21.5?
# objname = 170019716456251517 # mag 21.3?
# objname = 170019696475111620 # Fading
# objname = 314002971135836309 # Fading
# objname = 313994145051443228 # Fading
# objname = 313985346353234012 # Fading
objname = 313879819574051272

N = len(lsst_fink_tbl_object_hg_filtered)

ncol = 4
nrow = int(np.ceil(N / ncol))

fig, axes = plt.subplots(
    nrow, ncol,
    figsize=(4.5*ncol, 3*nrow)
)

axes = axes.flatten()

for i, row in enumerate(lsst_fink_tbl_object_hg_filtered):

    objid = row['diaObject_diaObjectId']
    DM = row['host_galaxy_distance_modulus']

    ax = axes[i]

    plot_lightcurve(objid, DM=DM, ax=ax)

    # 각 subplot ylabel 제거
    ax.set_ylabel("")

# 남는 subplot 제거
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# 전체 ylabel 하나만
fig.supylabel("Magnitude", x=0.04, fontsize = 15)
fig.supxlabel("MJD", y=0.04, fontsize = 15)

# subplot 간격 조절
fig.subplots_adjust(
    left=0.12,
    right=0.98,
    bottom=0.07,
    top=0.95,
    wspace=0.35,
    hspace=0.45
)

plt.show()
#%%
tbl_selected = lsst_fink_tbl_object[lsst_fink_tbl_object['diaObject_diaObjectId'] == objname]
print(tbl_selected)
# %%
for key, value in filter.config.catalog_constraints.items():
    print(key, '->', tbl_selected[key][0])
    print('--------------------------------')
#%%
filter.config.catalog_constraints
# %%
lsst_fink_tbl_object_filtered[lsst_fink_tbl_object_filtered['diaObject_diaObjectId'] == objname]
# %%
