
#%%
from astropy.table import Table
from tqdm import tqdm
import numpy as np
from bridge.utils import HostGalaxyCatalog
from bridge.alertfilter import BaseFilter
#%%
class TNS_Filter(BaseFilter):
    def __init__(self, hostgalaxycatalog=None):
        super().__init__('tns_filter.config', hostgalaxycatalog)

    def apply_hostgalaxy_filter(self, tbl: Table, return_all: bool = False, verbose = True) -> Table:
        """
        Keep only rows that have at least one host galaxy within radius_arcsec.
        Uses diaSource_ra, diaSource_dec for position.
        """
        for col in ('ra_deg', 'dec_deg'):
            if col not in tbl.colnames:
                print(f"WARNING: '{col}' not in table. Skipping host galaxy filter.")
                return tbl

        total_mask = tbl['mask_all'].copy() if 'mask_all' in tbl.colnames else np.ones(len(tbl), dtype=bool)
        min_dist = self.config.hostgalaxy_constraints.get('min_distance', -np.inf)
        max_dist = self.config.hostgalaxy_constraints.get('max_distance', np.inf)
        tbl['mask_hostgalaxy'] = False
        tbl['hostgalaxy_distance_modulus'] = np.full(len(tbl), np.nan)
        
        for i, row in tqdm(enumerate(tbl), total=len(tbl), desc='Applying host galaxy filter...'):

            ra = float(row['ra_deg'])
            dec = float(row['dec_deg'])

            host_galaxy = self.hostgalaxycatalog.match_host(
                ra, dec,
                self.config.hostgalaxy_constraints['search_radius_arcsec'],
                max_dell=self.config.hostgalaxy_constraints['max_dell'],
                plot=False,
                return_all=False
            )

            if host_galaxy is None:
                continue

            for key, val in dict(host_galaxy).items():
                colname = f'hostgalaxy_{key.lower()}'
                
                if colname not in tbl.colnames:
                    if isinstance(val, str):
                        tbl[colname] = [None] * len(tbl)
                    else:
                        tbl[colname] = np.full(len(tbl), np.nan)
                        
            for key, val in dict(host_galaxy).items():
                colname = f'hostgalaxy_{key.lower()}'
                tbl[colname][i] = val
                
            if (min_dist <= host_galaxy['Dist'] <= max_dist):
                tbl['mask_hostgalaxy'][i] = True
            else:
                tbl['mask_hostgalaxy'][i] = False         
            
            tbl['hostgalaxy_distance_modulus'][i] = 5 * np.log10(host_galaxy['Dist']) + 25

        prev_count = np.sum(total_mask)
        total_mask &= np.asarray(tbl['mask_hostgalaxy'], dtype=bool)
        new_count = np.sum(total_mask)

        if verbose:
            print(f'Filter hostgalaxy: {prev_count} -> {new_count}')

        tbl = self._update_mask_all(tbl)

        if not return_all:
            tbl = tbl[tbl['mask_all']]

        return tbl

    def apply_magnitude_filter(self, tbl, return_all: bool = False, verbose=True):
        tbl['absmag'] = tbl['Discovery Mag/Flux'] - tbl['hostgalaxy_distance_modulus']

        total_mask = tbl['mask_all'].copy() if 'mask_all' in tbl.colnames else np.ones(len(tbl), dtype=bool)

        for key, constraint in self.config.magnitude_constraints.items():

            if key not in tbl.colnames:
                print(f"WARNING: {key} not in table. Skipping...")
                continue

            mask_col = f'mask_{key}'

            # constraint가 list일 수도 있음
            if isinstance(constraint, (list, tuple)):
                mask = np.ones(len(tbl), dtype=bool)
                for c in constraint:
                    mask &= self._apply_single_constraint(tbl, key, c)
            else:
                mask = self._apply_single_constraint(tbl, key, constraint)

            tbl[mask_col] = mask

            prev_count = np.sum(total_mask)
            total_mask &= mask
            new_count = np.sum(total_mask)

            if verbose:
                print(f'Filter {key}: {prev_count} -> {new_count}')

        tbl = self._update_mask_all(tbl)

        if not return_all:
            tbl = tbl[tbl['mask_all']]

        return tbl
# %%