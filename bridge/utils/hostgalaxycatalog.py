#%%
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import numpy as np

from bridge.configuration import Configuration
#%%
class HostGalaxyCatalog:
    def __init__(self, catalog_path: str = None, nside: int = 8192):
        
        self.config = Configuration(config_filenames=['hostgalaxycatalog.config'])
        if catalog_path is not None:
            self.config.hostgalaxycatalog_path = catalog_path

        self.catalog = None
        self.target_catalog = None
        self.coords = None
        self.NSIDE = nside

        self._load_catalog()
        self._prepare_catalog()
        self.target_catalog = self.catalog

        self._build_sorted_hp_index()

    def __repr__(self):
        return f'HostGalaxyCatalog(n_selected/n_all = {len(self.target_catalog)}/{len(self.catalog)})'

    def search_catalog(self, ra: float, dec: float, radius_arcsec: float = 300.0):
        """
        Search the catalog for galaxies within the given radius.
        ==========
        Parameters:
        - ra: float, transient RA in degrees
        - dec: float, transient Dec in degrees
        - radius_arcsec: float, search radius in arcseconds
        =======
        Returns:
        - candidates: Table, candidates table
        =======
        Examples:
        >>> candidates = self.search_catalog(ra=100.0, dec=20.0, radius_arcsec=600.0)
        >>> print(candidates)
        """

        radius = (radius_arcsec * u.arcsec).to(u.rad).value

        theta = np.deg2rad(90.0 - dec)
        phi = np.deg2rad(ra)
        vec = hp.ang2vec(theta, phi)

        pix = hp.query_disc(self.NSIDE, vec, radius, nest=True)
        indices = self._query_hpix(pix)
        if len(indices) == 0:
            return Table()

        subset = self.target_catalog[indices]

        coords = self.coords[indices]
        target = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

        sep = coords.separation(target)
        mask = sep <= (radius_arcsec * u.arcsec)

        result = subset[mask].copy()
        result['separation_arcsec'] = sep[mask].arcsec

        return result

    # def cut_catalog(self, **kwargs):
    #     """
    #     Cut the catalog with the given kwargs.
    #     =======
    #     Parameters:
    #     - kwargs: dict, keywords and values to cut the catalog keywords must be in the catalog columns
    #     =======
    #     Returns:
    #     - None
    #     =======
    #     Examples:
    #     >>> self.cut_catalog(z_cmb   = '<0.1')
    #     >>> self.cut_catalog(distance = '<200')
    #     """
    #     print(f'Start cutting catalog... kwargs = {kwargs}')
    #     raw_catalog = self.catalog.copy()

    #     for cut_key, cut_value in kwargs.items():
    #         if cut_key not in raw_catalog.colnames:
    #             print(f"WARNING: {cut_key} not in catalog columns. Skipping...")
    #             continue

    #         print(f'Cutting {cut_key} with {cut_value}...')

    #         if isinstance(cut_value, str):
    #             if cut_value.startswith('<='):
    #                 raw_catalog = raw_catalog[raw_catalog[cut_key] <= float(cut_value[2:])]
    #             elif cut_value.startswith('>='):
    #                 raw_catalog = raw_catalog[raw_catalog[cut_key] >= float(cut_value[2:])]
    #             elif cut_value.startswith('<'):
    #                 raw_catalog = raw_catalog[raw_catalog[cut_key] < float(cut_value[1:])]
    #             elif cut_value.startswith('>'):
    #                 raw_catalog = raw_catalog[raw_catalog[cut_key] > float(cut_value[1:])]
    #             elif cut_value.startswith('='):
    #                 raw_catalog = raw_catalog[raw_catalog[cut_key] == cut_value[1:]]
    #         else:
    #             raw_catalog = raw_catalog[raw_catalog[cut_key] == cut_value]

    #     self.target_catalog = raw_catalog
    #     print(f'Catalog cut finished: {len(self.target_catalog)} sources.')

    #     self._build_sorted_hp_index()

    def match_host(self,
                   ra: float,
                   dec: float,
                   search_radius_arcsec: float = 600.0,
                   max_dell: float = 1.5,
                   return_all: bool = False, 
                   plot: bool = True,
                   save_path: str = None):
        """
        Match host galaxy candidates for a given transient coordinate.
        ==========
        Parameters:
        - ra: float, transient RA in degrees
        - dec: float, transient Dec in degrees
        - search_radius_arcsec: float, search radius in arcseconds
        - max_dell: float, maximum distance in arcseconds
        - return_all: bool, return all candidates if True
        - plot: bool, plot the candidates if True
        - save_path: str, save path for the plot
        =======
        Returns:
        - candidates: Table, candidates table
        - best: Table, best candidate table
        =======
        Examples:
        >>> candidates = self.match_host(ra=100.0, dec=20.0, search_radius_arcsec=600.0, max_dell=10, return_all=True, plot=False, save_path=None)
        >>> print(candidates)
        >>> best = self.match_host(ra=100.0, dec=20.0, search_radius_arcsec=600.0, max_dell=10, return_all=False, plot=False, save_path=None)
        >>> print(best)
        """
        candidates = self.search_catalog(ra, dec, radius_arcsec=search_radius_arcsec)

        if len(candidates) == 0:
            return None if not return_all else Table()

        d_ell_list = []
        dx_list = []
        dy_list = []

        for row in candidates:
            d_ell, dx, dy = self._compute_ellipse_distance(
                ra_t=ra,
                dec_t=dec,
                ra_g=float(row['RAdeg']),
                dec_g=float(row['DEdeg']),
                r1_arcsec=float(row['R1']),
                r2_arcsec=float(row['R2']),
                pa_deg=float(row['PA']),
            )
            d_ell_list.append(d_ell)
            dx_list.append(dx)
            dy_list.append(dy)
        candidates['d_ell'] = np.array(d_ell_list, dtype=float)
        candidates['dx_arcsec'] = np.array(dx_list, dtype=float)
        candidates['dy_arcsec'] = np.array(dy_list, dtype=float)

        # optional quality score
        # smaller is better
        score = np.array(candidates['d_ell'], dtype=float)

        # optionally downweight crowded / uncertain systems
        if 'fracNearby' in candidates.colnames:
            frac = np.asarray(candidates['fracNearby'], dtype=float)
            frac[~np.isfinite(frac)] = 0.0
            score = score * (1.0 + 0.5 * frac)

        candidates['host_score'] = score
        candidates.sort('host_score')

        # optional cut
        best_within_dell = None
        candidates_within_dell = candidates.copy()   
        if max_dell is not None:
            candidates_within_dell = candidates_within_dell[candidates_within_dell['d_ell'] <= max_dell]
            if not len(candidates_within_dell) == 0:
                best_within_dell = candidates_within_dell[candidates_within_dell['host_score'].argmin()]

        if plot:
            self.plot_host_candidates(ra, dec, search_radius_arcsec, candidates, best=best_within_dell, save_path=save_path)

        if return_all:
            return candidates_within_dell
        else:
            return best_within_dell

    def plot_host_candidates(self, ra_t, dec_t, search_radius_arcsec, candidates, best=None, save_path=None):
        """
        Plot the host galaxy candidates.
        ==========
        Parameters:
        - ra_t: float, transient RA in degrees
        - dec_t: float, transient Dec in degrees
        - search_radius_arcsec: float, search radius in arcseconds
        - candidates: Table, candidates table
        - best: Table, best candidate table
        - save_path: str, save path for the plot
        =======
        Returns:
        - None
        =======
        Examples:
        >>> self.plot_host_candidates(ra_t=100.0, dec_t=20.0, search_radius_arcsec=600.0, candidates=candidates, best=best, save_path=None)
        """
        fig, ax = plt.subplots(figsize=(7,7))

        for i, row in enumerate(candidates):

            ra_g = row["RAdeg"]
            dec_g = row["DEdeg"]

            R1 = row["R1"]/3600
            R2 = row["R2"]/3600
            PA = row["PA"]

            theta = np.deg2rad(90-PA)

            t = np.linspace(0,2*np.pi,200)

            x = R1*np.cos(t)
            y = R2*np.sin(t)

            xr = x*np.cos(theta) - y*np.sin(theta)
            yr = x*np.sin(theta) + y*np.cos(theta)

            color = "gray"
            lw = 1
            alpha = 0.5
            
            ax.plot(ra_g + xr, dec_g + yr,
                    color=color,
                    lw=lw,
                    alpha=alpha)

            ax.scatter(ra_g, dec_g,
                    color=color,
                    s=20)
        
        if best is not None:
            ra_g = best["RAdeg"]
            dec_g = best["DEdeg"]

            R1 = best["R1"]/3600
            R2 = best["R2"]/3600
            PA = best["PA"]

            theta = np.deg2rad(90-PA)

            t = np.linspace(0,2*np.pi,200)

            x = R1*np.cos(t)
            y = R2*np.sin(t)

            xr = x*np.cos(theta) - y*np.sin(theta)
            yr = x*np.sin(theta) + y*np.cos(theta)

            color = "red"
            lw = 1.5
            alpha = 0.5
            
            ax.plot(ra_g + xr, dec_g + yr,
                    color=color,
                    lw=lw,
                    alpha=alpha)

            ax.scatter(ra_g, dec_g,
                    color=color,
                    s=20)

        # transient
        ax.scatter(ra_t, dec_t, marker="*", s=200, color="blue", label="Transient")

        # ---------------------------
        # Main figure limits
        # ---------------------------

        r_deg = search_radius_arcsec / 3600

        ax.set_xlim(ra_t - r_deg, ra_t + r_deg)
        ax.set_ylim(dec_t - r_deg, dec_t + r_deg)

        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
        ax.set_title("Candidate Host Galaxies")

        ax.grid(alpha=0.3)
        ax.invert_xaxis()

        # ---------------------------
        # inset (best host)
        # ---------------------------

        if best is not None:

            ra_g = best["RAdeg"]
            dec_g = best["DEdeg"]

            R1 = best["R1"]
            R2 = best["R2"]
            PA = best["PA"]

            theta = np.deg2rad(90-PA)

            axins = inset_axes(ax, width="35%", height="35%", loc="upper right", bbox_to_anchor=(0.5, -0.05, 1, 1), bbox_transform=ax.transAxes)

            t = np.linspace(0,2*np.pi,400)

            x = (R1/3600)*np.cos(t)
            y = (R2/3600)*np.sin(t)

            xr = x*np.cos(theta) - y*np.sin(theta)
            yr = x*np.sin(theta) + y*np.cos(theta)

            axins.plot(ra_g + xr, dec_g + yr, color="red", lw=2)

            axins.scatter(ra_g, dec_g, color="black", s=40)

            axins.scatter(ra_t, dec_t, marker="*", s=150, color="blue")

            # ---------------------------
            # Zoom limits = ellipse size
            # ---------------------------
            
            margin = 1.5 * max(R1, R2) / 3600

            axins.set_xlim(ra_g - margin, ra_g + margin)
            axins.set_ylim(dec_g - margin, dec_g + margin)
            from matplotlib.ticker import FormatStrFormatter

            axins.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            axins.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            axins.set_xticks(np.linspace(ra_g - margin, ra_g + margin, 3))
            axins.set_yticks(np.linspace(dec_g - margin, dec_g + margin, 3))

            axins.set_title(f"Best Host: {best['Name']}", fontsize=10)

            axins.grid(alpha=0.3)
            axins.invert_xaxis()

            # ---------------------------
            # Best host properties textbox
            # ---------------------------

            props = [
                'Name',
                'Dist',
                'z',
                'e_Dist',
                'R1',
                'R2',
                'PA',
                'Gmag',
                'BPmag',
                'gmag',
                'rmag',
                'imag',
                'zmag',
                'W1mag',
                'W2mag',
                'E(B-V)',
                'logM'
            ]

            text_lines = []
            for key in props:
                if key in best.colnames:
                    val = best[key]
                    if isinstance(val, float):
                        text_lines.append(f"{key}: {val:.3f}")
                    else:
                        text_lines.append(f"{key}: {val}")

            textstr = "\n".join(text_lines)
            axins.text(
                0.5, -0.35,
                textstr,
                transform=axins.transAxes,
                fontsize=9,
                ha='center',
                va='top',
                multialignment='left',
                linespacing=1.4,
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=1,
                    edgecolor="black",
                    pad=0.6
                )
            )

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def _compute_ellipse_distance(self, ra_t, dec_t, ra_g, dec_g, r1_arcsec, r2_arcsec, pa_deg):
        """
        Compute ellipse-normalized distance:
            d_ell = sqrt((x'/R1)^2 + (y'/R2)^2)
        """

        # local tangent-plane offsets in arcsec
        dx = (ra_t - ra_g) * np.cos(np.deg2rad(dec_g)) * 3600.0
        dy = (dec_t - dec_g) * 3600.0

        # rotate into major/minor axis frame
        # PA convention may need sign check depending on catalog definition
        theta = np.deg2rad(pa_deg)

        x_prime =  dx * np.sin(theta) + dy * np.cos(theta)
        y_prime =  dx * np.cos(theta) - dy * np.sin(theta)

        d_ell = np.sqrt((x_prime / r1_arcsec)**2 + (y_prime / r2_arcsec)**2)
        return d_ell, dx, dy

    def _load_catalog(self):
        catalog_path = self.config.hostgalaxycatalog_path
        print(f'Loading host galaxy catalog from {catalog_path} ...')

        self.catalog = Table.read(catalog_path, format='fits', memmap=True)

        print(f'Catalog loaded with {len(self.catalog)} sources.')

    def _prepare_catalog(self):
        cat = self.catalog

        ra = np.asarray(cat['RAdeg'], dtype=float)
        dec = np.asarray(cat['DEdeg'], dtype=float)

        self.coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

        colname = f'healpix_nside{self.NSIDE}'

        if colname not in cat.colnames:
            print(f'Adding {colname} (slow, do once and save!)')
            theta = np.deg2rad(90.0 - dec)
            phi = np.deg2rad(ra)

            cat[colname] = hp.ang2pix(self.NSIDE, theta, phi, nest=True)

    def _build_sorted_hp_index(self):
        print("Building sorted healpix index ...")

        hpix = np.asarray(self.target_catalog[f'healpix_nside{self.NSIDE}'], dtype=np.int64)

        order = np.argsort(hpix)

        self.sorted_hpix = hpix[order]
        self.sorted_index = order

        print("Sorted healpix index built.")

    def _query_hpix(self, hpix_list):

        hpix_list = np.asarray(hpix_list, dtype=np.int64)

        left = np.searchsorted(self.sorted_hpix, hpix_list, side='left')
        right = np.searchsorted(self.sorted_hpix, hpix_list, side='right')

        mask = right > left
        left = left[mask]
        right = right[mask]

        if len(left) == 0:
            return np.array([], dtype=int)

        return np.concatenate([
            self.sorted_index[l:r] for l, r in zip(left, right)
        ])

# %%
if __name__ == "__main__":
    ra = 185.01687190191404
    dec = 8.644788357165169
    hostgalaxycatalog = HostGalaxyCatalog()
    hostgalaxycatalog.match_host(ra, dec, search_radius_arcsec = 300, max_dell = 2.0, return_all = True, plot = True, save_path = None)
# %%
