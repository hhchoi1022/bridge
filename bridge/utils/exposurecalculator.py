
# Calculate required exposure time to reach the certain limiting magnitude 
# This code must be empirical from the observation.
#%%
from astropy.table import Table
from astropy.time import Time
from sklearn.linear_model import LinearRegression
from pathlib import Path
import glob
import json

#%%
from bridge.connector import GWPortalConnector
from bridge.objects import Targets
from bridge.objects import mainObserver
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clip
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import re
from sklearn.linear_model import LinearRegression
from astropy.stats import sigma_clip

#%%
def calculate_moonsep_worker(ra, dec, mjd):
    target = Targets(ra_list = ra, dec_list = dec)
    moon_sep = target.moon_sep(utctime = Time(mjd, format = 'mjd'))
    return moon_sep

def calculate_moonphase_worker(mjd):
    observer = mainObserver()
    moon_phase = observer.moon_phase(utctimes = Time(mjd, format = 'mjd'))
    return moon_phase

def calculate_hour_since_sunset_worker(mjd):
    observer = mainObserver()
    sunrise = observer.sun_risetime(utctimes = Time(mjd, format = 'mjd'), mode = 'next', horizon = 10)
    sunset = observer.sun_settime(sunrise, mode = 'previous', horizon = -18)
    hour_since_sunset = (mjd - sunset.mjd) * 24
    if hour_since_sunset < 0:
        hour_since_sunset += 24
    return hour_since_sunset
    
#%%

class ExposureCalculator:
    def __init__(self):
        self.db = GWPortalConnector()
        self.db_tbl = Table()
        self.model = None
        
    BROAD_BAND_COLORS = {
        "g": "green",
        "r": "red",
        "i": "gold",
        "z": "k"
    }
    
    BROAD_ORDER = ["g", "r", "i", "z"]

    @staticmethod
    def _filter_wavelength(filt):
        """
        Extract wavelength from filter name.
        Examples:
          m400 -> 400
          m875 -> 875
          g/r/i -> None
        """
        m = re.match(r"m(\d+)", filt)
        return int(m.group(1)) if m else None
    
    def get_filter_colors(self, filters):
        """
        Return dict: {filter_name: color}
        """
        filters = list(filters)

        colors = {}

        # -------------------------
        # Broad bands
        # -------------------------
        for f in filters:
            if f in self.BROAD_BAND_COLORS:
                colors[f] = self.BROAD_BAND_COLORS[f]

        # -------------------------
        # Medium bands
        # -------------------------
        mfilters = [f for f in filters if f.startswith("m")]

        if len(mfilters) > 0:
            waves = np.array([self._filter_wavelength(f) for f in mfilters])

            norm = mcolors.Normalize(vmin=waves.min(), vmax=waves.max())
            cmap = cm.get_cmap("jet")

            for f, w in zip(mfilters, waves):
                colors[f] = cmap(norm(w))

        return colors
    
    def sort_filters_broad_then_medium(self, filters):
        broad_rank = {f: i for i, f in enumerate(self.BROAD_ORDER)}

        def key(f):
            f = str(f)
            # 0) broad
            if f in broad_rank:
                return (0, broad_rank[f], -1, f)

            # 1) medium
            w = self._filter_wavelength(f)
            if w is not None:
                return (1, 0, w, f)

            # 2) others
            return (2, 0, 10**9, f)

        return sorted(list(filters), key=key)

    def update_depth_model(self,
                           obs_start_date: str = '2025-01-01',
                           obs_end_date: str = '2026-01-01',
                           depth_table: Table = None,
                           model_save_path: str = None,
                           depth_table_save_path: str = None,
                           n_processes: int = 32
                           ):
        date_end = Time.now()
        current_file = Path(__file__).resolve()
        current_dir = current_file.parent
        if model_save_path is None:
            model_save_path = f'{current_dir}/depth_model_{date_end.datetime.strftime("%Y%m%d%H%M%S")}.json'
        if depth_table_save_path is None:
            depth_table_save_path = f'{current_dir}/depth_data_{date_end.datetime.strftime("%Y%m%d%H%M%S")}.csv'
            
        
        if depth_table is None:
            self.db_tbl = self.db.query(obs_start_date = obs_start_date, obs_end_date = obs_end_date)
            
            # Calculate required parameters for the model        
            with ProcessPoolExecutor(max_workers = n_processes) as pool:
                results = list(tqdm(pool.map(calculate_moonsep_worker, self.db_tbl['ra_center'], self.db_tbl['dec_center'], self.db_tbl['mjd']), desc = 'Calculating moon separation...', total = len(self.db_tbl)))
                all_moonsep = results
            with ProcessPoolExecutor(max_workers = n_processes) as pool:
                results = list(tqdm(pool.map(calculate_moonphase_worker, self.db_tbl['mjd']), desc = 'Calculating moon phase...', total = len(self.db_tbl)))
                all_moonphase = results
            with ProcessPoolExecutor(max_workers = n_processes) as pool:
                results = list(tqdm(pool.map(calculate_hour_since_sunset_worker, self.db_tbl['mjd']), desc = 'Calculating hour since sunset...', total = len(self.db_tbl)))
                all_hour_since_sunset = results
                
            self.db_tbl['moon_sep'] = all_moonsep
            self.db_tbl['moon_phase'] = all_moonphase
            self.db_tbl['hour_since_sunset'] = all_hour_since_sunset
        else:
            self.db_tbl = depth_table
        
        filter_names = sorted(list(set(self.db_tbl['filter'])))
        model_dict = {}
        for filter_name in filter_names:
            fit_result = self.fit_depth_model(self.db_tbl, filter_name)
            model_dict[str(filter_name)] = fit_result
            
        with open(model_save_path, 'w') as f:
            json.dump(model_dict, f, indent = 4)
            self.model = model_dict
        
        self.db_tbl.remove_columns(['poly', 'poly_galactic'])
        self.db_tbl.write(depth_table_save_path, format = 'csv', overwrite = True)

    def fit_depth_model(
        self,
        tbl,
        filter_name,
        exptime=100,
        sigma=3.0,
        maxiters=3,
        verbose=True,
        visualize=True
        ):
        """
        Fit limiting magnitude (UL5) as a linear function of observing conditions
        at a fixed exposure time, with sigma clipping on residuals.

        Model:
            UL5 = b0
                + b1 * moon_phase
                + b2 * moon_sep
                + b3 * hour_since_sunset
                + b4 * seeing
        """
        def _to_float_array(col):
            """Safely convert Astropy Column / MaskedColumn to float ndarray."""
            if hasattr(col, 'filled'):          # MaskedColumn
                col = col.filled(np.nan)
            return np.asarray(col, dtype=float)

        # ------------------------------------------------------------
        # 1. Select data
        # ------------------------------------------------------------
        m = (tbl['filter'] == filter_name) & (tbl['exptime'] == exptime)
        tbl_group = tbl[m]

        if len(tbl_group) < 10:
            raise RuntimeError("Not enough data points for fitting")
        
        moon_phase = _to_float_array(tbl_group['moon_phase'])
        moon_sep   = _to_float_array(tbl_group['moon_sep'])
        hour       = _to_float_array(tbl_group['hour_since_sunset'])
        seeing     = _to_float_array(tbl_group['seeing'])
        ul5        = _to_float_array(tbl_group['ul5'])

        finite = (
            np.isfinite(moon_phase) &
            np.isfinite(moon_sep) &
            np.isfinite(hour) &
            np.isfinite(seeing) &
            np.isfinite(ul5)
        )

        tbl_group = tbl_group[finite]

        # NOW it is safe
        X = np.column_stack([
            tbl_group['moon_phase'],
            tbl_group['moon_sep'],
            tbl_group['hour_since_sunset'],
            tbl_group['seeing'],
        ]).astype(float)

        y = np.asarray(tbl_group['ul5'], dtype=float)

        # ------------------------------------------------------------
        # 2. Initial fit
        # ------------------------------------------------------------
        model_init = LinearRegression()
        model_init.fit(X, y)
        y_pred_init = model_init.predict(X)

        # ------------------------------------------------------------
        # 3. Sigma clipping on residuals
        # ------------------------------------------------------------
        resid = y - y_pred_init
        finite = np.isfinite(resid)

        clipped = sigma_clip(
            resid[finite],
            sigma=sigma,
            maxiters=maxiters,
            cenfunc='median',
            stdfunc='mad_std',
        )

        mask_good = np.zeros_like(y, dtype=bool)
        mask_good[finite] = ~clipped.mask

        if verbose:
            print(
                f"[{filter_name}, {exptime}s] "
                f"Rejected {np.sum(~mask_good)}/{len(y)} points "
                f"({100*np.sum(~mask_good)/len(y):.1f}%)"
            )

        # ------------------------------------------------------------
        # 4. Final fit
        # ------------------------------------------------------------
        model = LinearRegression()
        model.fit(X[mask_good], y[mask_good])

        y_model_all = model.predict(X)
        scatter = np.std(y[mask_good] - model.predict(X[mask_good]))
        median = [np.nanmedian(moon_phase), np.nanmedian(moon_sep), np.nanmedian(hour), np.nanmedian(seeing)]

        result = {
            "intercept": model.intercept_,
            "coefficients": list(model.coef_),
            "medians": median,
            "scatter": scatter,
            "n_used": int(np.sum(mask_good)),
            "n_total": int(len(y)),
            "exptime": exptime,
            "filter": str(filter_name)}

        # ------------------------------------------------------------
        # 5. Visualization (PARTIAL DEPENDENCE — 2x2)
        # ------------------------------------------------------------
        if visualize:
            features = [
                ("moon_phase", tbl_group['moon_phase']),
                ("moon_sep", tbl_group['moon_sep']),
                ("hour_since_sunset", tbl_group['hour_since_sunset']),
                ("seeing", tbl_group['seeing']),
            ]

            medians = {
                "moon_phase": np.median(tbl_group['moon_phase']),
                "moon_sep": np.median(tbl_group['moon_sep']),
                "hour_since_sunset": np.median(tbl_group['hour_since_sunset']),
                "seeing": np.median(tbl_group['seeing']),
            }

            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.flatten()  # <-- KEY FIX

            for i, (name, values) in enumerate(features):
                ax = axes[i]

                x = np.asarray(values, dtype=float)
                x_grid = np.linspace(np.nanmin(x), np.nanmax(x), 200)

                # Partial-dependence design matrix
                X_grid = np.zeros((len(x_grid), X.shape[1]))
                X_grid[:, 0] = medians["moon_phase"]
                X_grid[:, 1] = medians["moon_sep"]
                X_grid[:, 2] = medians["hour_since_sunset"]
                X_grid[:, 3] = medians["seeing"]
                X_grid[:, i] = x_grid

                y_grid = model.predict(X_grid)

                # Used points
                ax.scatter(
                    x[mask_good],
                    y[mask_good],
                    s=20,
                    facecolor = 'none',
                    edgecolor = 'black',
                    alpha=0.6,
                    label="Used in fit",
                )

                # Rejected points
                ax.scatter(
                    x[~mask_good],
                    y[~mask_good],
                    s=30,
                    facecolor = 'none',
                    edgecolor = 'red',
                    alpha=0.2,
                    label="Sigma-clipped",
                )

                # Linear model
                ax.plot(
                    x_grid,
                    y_grid,
                    "r--",
                    lw=3,
                    label="Linear model",
                )

                ax.set_xlabel(name.replace("_", " ").title(), fontsize = 16)
                ax.set_ylabel("UL5", fontsize =16)
                ax.grid(alpha=0.3)
                ax.set_title(name.replace("_", " ").title(), fontsize = 16)

            # One legend for all panels
            handles, labels = axes[0].get_legend_handles_labels()

            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=3,
                bbox_to_anchor=(0.5, 0.92),   # <--- controls vertical position
                frameon=True,
                fontsize = 16
            )

            fig.suptitle(f"{filter_name} band", fontsize=25, y=0.96)

            fig.tight_layout(rect=[0, 0, 1, 0.88])  # <--- reserve space for legend+title
            plt.show()

            # ------------------------------------------------------------
            # 6. Residual diagnostic (UL5 - model prediction, all components)
            # ------------------------------------------------------------
            y_pred_all = model.predict(X)
            residual_all = y - y_pred_all

            resid_used = residual_all[mask_good]
            resid_rej  = residual_all[~mask_good]

            mean_used = np.mean(resid_used)
            std_used  = np.std(resid_used)

            fig_r, axes_r = plt.subplots(1, 2, figsize=(12, 5))

            # --- Left: residual vs predicted UL5 ---
            ax0 = axes_r[0]
            ax0.scatter(
                y_pred_all[mask_good],
                resid_used,
                s=20,
                facecolor='none',
                edgecolor='black',
                alpha=0.6,
                label="Used in fit",
            )
            ax0.scatter(
                y_pred_all[~mask_good],
                resid_rej,
                s=30,
                facecolor='none',
                edgecolor='red',
                alpha=0.3,
                label="Sigma-clipped",
            )
            ax0.axhline(0.0, color='red', ls='--', lw=2)
            ax0.axhline(mean_used + std_used, color='gray', ls=':', lw=1)
            ax0.axhline(mean_used - std_used, color='gray', ls=':', lw=1)
            ax0.set_xlabel("Predicted UL5", fontsize=16)
            ax0.set_ylabel("Residual (UL5$_{obs}$ - UL5$_{model}$)", fontsize=16)
            ax0.set_title("Residual vs Predicted UL5", fontsize=16)
            ax0.grid(alpha=0.3)
            ax0.legend(fontsize=12, loc='best')

            # --- Right: residual histogram ---
            ax1 = axes_r[1]
            bins = np.linspace(
                np.nanmin(residual_all[np.isfinite(residual_all)]),
                np.nanmax(residual_all[np.isfinite(residual_all)]),
                40,
            )
            ax1.hist(
                resid_used, bins=bins,
                color='black', alpha=0.5, label="Used in fit",
            )
            ax1.hist(
                resid_rej, bins=bins,
                color='red', alpha=0.4, label="Sigma-clipped",
            )
            ax1.axvline(0.0, color='red', ls='--', lw=2)
            ax1.axvline(mean_used, color='blue', ls='-', lw=1.5,
                        label=f"mean = {mean_used:+.3f}")
            ax1.axvline(mean_used + std_used, color='gray', ls=':', lw=1)
            ax1.axvline(mean_used - std_used, color='gray', ls=':', lw=1,
                        label=f"std = {std_used:.3f}")
            ax1.set_xlabel("Residual (UL5$_{obs}$ - UL5$_{model}$)", fontsize=16)
            ax1.set_ylabel("Count", fontsize=16)
            ax1.set_title("Residual distribution", fontsize=16)
            ax1.grid(alpha=0.3)
            ax1.legend(fontsize=11, loc='best')

            fig_r.suptitle(
                f"{filter_name} band — full-model residual "
                f"(N$_{{used}}$ = {int(np.sum(mask_good))}, scatter = {scatter:.3f})",
                fontsize=18, y=0.99,
            )
            fig_r.tight_layout(rect=[0, 0, 1, 0.93])
            plt.show()

        return result

    def load_model(self, model_path: str = None):
        if model_path is None:
            model_list = glob.glob(f'{Path(__file__).parent}/depth_model_*.json')
            model_path = sorted(model_list)[-1]
            
        with open(model_path, 'r') as f:
            self.model = json.load(f)
        print(f"Loading model from {model_path}")
        
    def load_depth_table(self, depth_table_path: str = None):
        if depth_table_path is None:
            depth_table_path = glob.glob(f'{Path(__file__).parent}/depth_data_*.csv')
            depth_table_path = sorted(depth_table_path)[-1]
        self.db_tbl = Table.read(depth_table_path)
        print(f"Loading depth table from {depth_table_path}")

    def calculate_exptime(self,
                          filter: str or list[str],
                          magnitude: float or list[float],
                          snr: float or list[float] = 5.0,
                          ra: float = None,
                          dec: float = None,
                          obsdate: Time = Time.now(),
                          moon_phase: float = 0.5,
                          moon_separation: float = 90,
                          hour_since_sunset: float = 5,
                          seeing: float = 2.0,
                          verbose: bool = True) -> dict:
        if self.model is None:
            raise ValueError("Model is not loaded. Please load the model first with self.load_model(model_path)")
        
        # ------------------------------
        # 1) If (ra, dec, obsdate) are all given, compute conditions
        # ------------------------------
        if (ra is not None) and (dec is not None) and (obsdate is not None):
            obsdate_mjd = Time(obsdate).mjd
            moon_phase = calculate_moonphase_worker(obsdate_mjd)
            moon_separation = calculate_moonsep_worker(ra, dec, obsdate_mjd)
            hour_since_sunset = calculate_hour_since_sunset_worker(obsdate_mjd)
            if verbose:
                print(f"Calculated from the target information (RA = {ra}, Dec = {dec}, Obsdate = {obsdate})")
                print(f"Moon phase = {moon_phase} \nMoon separation = {moon_separation} \nHour since sunset = {hour_since_sunset}")
    
        # --------------------------------------------------
        # 2. Fallback to model medians if still None
        # --------------------------------------------------
        if filter == 'all':
            filter = list(self.model.keys())
        filter = np.atleast_1d(filter)
        magnitude = np.atleast_1d(magnitude)
        snr = np.atleast_1d(snr)
        len_result = max(len(filter), len(magnitude), len(snr))
        if len(filter) == 1:
            filter = [filter[0]] * len_result
        if len(magnitude) == 1:
            magnitude = [magnitude[0]] * len_result
        if len(snr) == 1:
            snr = [snr[0]] * len_result
        
        if len(filter) != len(magnitude) or len(filter) != len(snr) or len(magnitude) != len(snr):
            raise ValueError("filter, magnitude, and snr must have the same length")
        
        result = dict()
        for filt, mag, snr in zip(filter, magnitude, snr):
            filt = str(filt)
            result[filt] = dict()
            model = self.model[filt]
            # defaults = model["medians"]
            defaults = [0.5, 90, 5, 2.0]
            if defaults is None:
                raise RuntimeError("Model medians not found. Refit model with medians saved.")

            moon_phase = defaults[0] if moon_phase is None else moon_phase
            moon_separation = defaults[1] if moon_separation is None else moon_separation
            hour_since_sunset = (
                defaults[2]
                if hour_since_sunset is None
                else hour_since_sunset
            )
            seeing = defaults[3] if seeing is None else seeing
            
            # ------------------------------
            # 3) Predict UL5 at reference exposure time (e.g. 100s)
            # ------------------------------
            beta0 = float(model["intercept"])
            beta = np.asarray(model["coefficients"], dtype=float)  # [b_phase, b_sep, b_hour, b_seeing]
            t_ref = float(model["exptime"])  # e.g. 100

            ul5_ref = (
                beta0
                + beta[0] * moon_phase
                + beta[1] * moon_separation
                + beta[2] * hour_since_sunset
                + beta[3] * seeing
            )

            # ------------------------------
            # 4) Convert desired magnitude + SNR to exposure time
            # ------------------------------
            # UL5 is defined at SNR=5 in your DB (assumed).
            # magnitude is interpreted as "desired limiting magnitude at SNR=5"
            t_exp = t_ref * 10 ** ((mag - ul5_ref) / 1.25) * (snr / 5.0) ** 2
            result[filt]['t_exp'] = t_exp
            result[filt]['magnitude'] = mag
            result[filt]['snr'] = snr
            result[filt]['ul5_ref'] = ul5_ref
            result[filt]['moon_phase'] = moon_phase
            result[filt]['moon_separation'] = moon_separation
            result[filt]['hour_since_sunset'] = hour_since_sunset
            result[filt]['seeing'] = seeing
        return result

    def calculate_snr(self,
                      filter: str or list[str],
                      magnitude: float or list[float],
                      exptime: float or list[float],
                      ra: float = None,
                      dec: float = None,
                      obsdate: Time = None,
                      moon_phase: float = 0.5,
                      moon_separation: float = 90,
                      hour_since_sunset: float = 5,
                      seeing: float = 2.0,
                      verbose: bool = True) -> dict:
        if self.model is None:
            raise ValueError("Model is not loaded. Please load the model first.")

        # ------------------------------
        # 1) If (ra, dec, obsdate) are all given, compute conditions
        # ------------------------------
        if obsdate is not None:
            moon_phase = calculate_moonphase_worker(Time(obsdate).mjd)
        if (ra is not None) and (dec is not None):
            obsdate_mjd = Time(obsdate).mjd
            moon_phase = calculate_moonphase_worker(obsdate_mjd)
            moon_separation = calculate_moonsep_worker(ra, dec, obsdate_mjd)
            hour_since_sunset = calculate_hour_since_sunset_worker(obsdate_mjd)
            if verbose:
                print(f"Calculated from the target information (RA = {ra}, Dec = {dec}, Obsdate = {obsdate})")
                print(f"Moon phase = {moon_phase} \nMoon separation = {moon_separation} \nHour since sunset = {hour_since_sunset}")

        # ------------------------------
        # 2) Vectorize inputs
        # ------------------------------
        if isinstance(filter, str):
            if filter == 'all':
                filter = list(self.model.keys())

        filter = np.atleast_1d(filter)
        magnitude = np.atleast_1d(magnitude)
        exptime = np.atleast_1d(exptime)

        n = max(len(filter), len(magnitude), len(exptime))

        if len(filter) == 1:
            filter = [filter[0]] * n
        if len(magnitude) == 1:
            magnitude = [magnitude[0]] * n
        if len(exptime) == 1:
            exptime = [exptime[0]] * n

        if not (len(filter) == len(magnitude) == len(exptime)):
            raise ValueError("filter, magnitude, and exptime must have the same length")

        # ------------------------------
        # 3) Loop over filters
        # ------------------------------
        result = {}

        for filt, mag, t in zip(filter, magnitude, exptime):
            filt = str(filt)
            model = self.model[filt]
            # defaults = model["medians"]
            defaults = [0.5, 90, 5, 2.0]

            moon_phase_i = defaults[0] if moon_phase is None else moon_phase
            moon_sep_i = defaults[1] if moon_separation is None else moon_separation
            hour_i = defaults[2] if hour_since_sunset is None else hour_since_sunset
            seeing_i = defaults[3] if seeing is None else seeing

            # ------------------------------
            # 4) Predict UL5 at reference exposure time
            # ------------------------------
            beta0 = float(model["intercept"])
            beta = np.asarray(model["coefficients"], dtype=float)
            t_ref = float(model["exptime"])

            ul5_ref = (
                beta0
                + beta[0] * moon_phase_i
                + beta[1] * moon_sep_i
                + beta[2] * hour_i
                + beta[3] * seeing_i
            )

            # ------------------------------
            # 5) Scale UL5 to requested exposure time
            # ------------------------------
            ul5_t = ul5_ref + 1.25 * np.log10(t / t_ref)

            # ------------------------------
            # 6) Compute SNR
            # ------------------------------
            snr_val = 5.0 * 10 ** (-0.4 * (mag - ul5_t))

            result[filt] = {
                            "snr": float(snr_val.item()),
                            "magnitude": float(mag.item()),
                            "exptime": float(t.item()),
                            "ul5_exptime": float(ul5_t),
                            "filter": filt,
                            "moon_phase": float(moon_phase_i),
                            "moon_separation": float(moon_sep_i),
                            "hour_since_sunset": float(hour_i),
                            "seeing": float(seeing_i)
                            }
        return result
    
    def plot_snr_vs_exptime(self,
                            filter: str or list[str],
                            magnitude=20.0,
                            exptime_range=(1, 1000),
                            exptime_ref = 300,
                            n_points=100,
                            ra=None,
                            dec=None,
                            obsdate=Time.now(),
                            moon_phase=0.5,
                            moon_separation=90,
                            hour_since_sunset=5,
                            seeing=2.0):
        
        exptimes = np.logspace(
            np.log10(exptime_range[0]),
            np.log10(exptime_range[1]),
            n_points,
        )
        exptimes = np.append(exptimes, exptime_ref)
        
        if isinstance(filter, str):
            if filter == 'all':
                filter = list(self.model.keys())
        
        # ----------------------------------
        # Main loop
        # ----------------------------------
        tbl = Table()
        tbl["exptime"] = exptimes
        for filt in filter:
            tbl[filt] = np.nan
        for i, t in enumerate(tqdm(exptimes, desc="Calculating SNR vs Exposure Time")):
            result = self.calculate_snr(
                filter=filter,
                magnitude=magnitude,
                exptime=t,
                ra=ra,
                dec=dec,
                obsdate=obsdate,
                moon_phase=moon_phase,
                moon_separation=moon_separation,
                hour_since_sunset=hour_since_sunset,
                seeing=seeing,
                verbose=False,
            )
            filters = sorted(list(result.keys()))
            
            for filt in filters:
                snr_val = result[filt]["snr"]
                tbl[filt][i] = snr_val
        
        moon_phase_rep = result[filt]["moon_phase"]
        moon_separation_rep = result[filt]["moon_separation"]
        hour_since_sunset_rep = result[filt]["hour_since_sunset"]
        
        plt.figure(figsize=(8, 12))
        xmax = exptimes[-2]
        plt.xlim(exptimes[0], xmax * 2)  
        colors = self.get_filter_colors(filters)
        for filt in filter:
            plt.plot(exptimes, tbl[filt], label=f'{str(filt)} (SNR(t_exp={exptime_ref}s) = {tbl[filt][-1]:.1f})', c = colors[filt])
            plt.text(exptimes[-2] * 1.05, tbl[filt][-2], str(filt), color=colors[filt], fontsize=12, va="center")
        plt.axvline(x=exptime_ref, color='black', linestyle='--', alpha = 0.5)
        lines = []

        if ra is not None:
            lines.append(f"RA = {ra:.2f}")
        if dec is not None:
            lines.append(f"Dec = {dec:.2f}")
        if obsdate is not None:
            lines.append(f"Obsdate = {Time(obsdate).iso}")
        lines += [
            f"Moon phase = {moon_phase_rep:.2f}",
            f"Moon separation = {moon_separation_rep:.2f}",
            f"Hour since sunset = {hour_since_sunset_rep:.2f}",
        ]

        text = "\n".join(lines)

        plt.text(
            0.02, 0.98,
            text,
            transform=plt.gca().transAxes,
            fontsize=12,
            ha="left",
            va="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="gray",
                alpha=0.9,
            ),
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Exposure time (s)", fontsize=16)
        plt.ylabel("SNR", fontsize=16)
        plt.title(f"SNR vs Exposure Time (mag={magnitude})", fontsize=20)
        plt.grid(alpha=0.3)
        plt.legend(ncol=2, loc = 'lower right')
        plt.tight_layout()
        plt.show()


    def plot_exptime_vs_snr(self,
                            filter: str or list[str],
                            magnitude=20.0,
                            snr_range=(1, 50),
                            snr_ref = 5,
                            n_points=100,
                            ra=None,
                            dec=None,
                            obsdate=Time.now(),
                            moon_phase=0.5,
                            moon_separation=90,
                            hour_since_sunset=5,
                            seeing=2.0):
        
        snrs = np.logspace(
            np.log10(snr_range[0]),
            np.log10(snr_range[1]),
            n_points,
        )
        snrs = np.append(snrs, snr_ref)
        
        if isinstance(filter, str):
            if filter == 'all':
                filter = list(self.model.keys())
        
        # ----------------------------------
        # Main loop
        # ----------------------------------
        tbl = Table()
        tbl["snr"] = snrs
        for filt in filter:
            tbl[filt] = np.nan
        for i, s in enumerate(tqdm(snrs, desc="Calculating SNR vs Exposure Time")):
            result = self.calculate_exptime(
                filter=filter,
                magnitude=magnitude,
                snr=s,
                ra=ra,
                dec=dec,
                obsdate=obsdate,
                moon_phase=moon_phase,
                moon_separation=moon_separation,
                hour_since_sunset=hour_since_sunset,
                seeing=seeing,
                verbose = False,
            )
            filters = sorted(list(result.keys()))
            
            for filt in filters:
                t_exp = result[filt]["t_exp"]
                tbl[filt][i] = t_exp
        
        moon_phase_rep = result[filt]["moon_phase"]
        moon_separation_rep = result[filt]["moon_separation"]
        hour_since_sunset_rep = result[filt]["hour_since_sunset"]
        
        plt.figure(figsize=(8, 12))
        xmax = snrs[-2]
        plt.xlim(snrs[0], xmax * 2)  
        colors = self.get_filter_colors(filters)
        for filt in filter:
            plt.plot(snrs, tbl[filt], label=f'{str(filt)} (t_exp(SNR={snr_ref}) = {tbl[filt][-1]:.1f}s)', c = colors[filt])
            plt.text(snrs[-2] * 1.05, tbl[filt][-2], str(filt), color=colors[filt], fontsize=12, va="center")
        plt.axvline(x=snr_ref, color='black', linestyle='--', alpha = 0.5)
        lines = []

        if ra is not None:
            lines.append(f"RA = {ra:.2f}")
        if dec is not None:
            lines.append(f"Dec = {dec:.2f}")
        if obsdate is not None:
            lines.append(f"Obsdate = {Time(obsdate).iso}")
        lines += [
            f"Moon phase = {moon_phase_rep:.2f}",
            f"Moon separation = {moon_separation_rep:.2f}",
            f"Hour since sunset = {hour_since_sunset_rep:.2f}",
        ]

        text = "\n".join(lines)

        plt.text(
            0.02, 0.98,
            text,
            transform=plt.gca().transAxes,
            fontsize=12,
            ha="left",
            va="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="gray",
                alpha=0.9,
            ),
        )
        
        plt.xscale("log")
        plt.yscale("log")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("SNR", fontsize=16)
        plt.ylabel("Exposure time (s)", fontsize=16)
        plt.title(f"Exposure time vs SNR (mag={magnitude})", fontsize=20)
        plt.grid(alpha=0.3)
        plt.legend(ncol=2, loc = 'lower right')
        plt.tight_layout()
        plt.show()

    def plot_depth_model(self, exptime = 100):
        if self.db_tbl is None or len(self.db_tbl) == 0:
            raise RuntimeError("Depth table is empty. Run update_depth_model first.")

        filters = self.sort_filters_broad_then_medium(set(self.db_tbl["filter"]))
        colors = self.get_filter_colors(filters)

        # ------------------------------
        # 1. Collect data
        # ------------------------------
        data = []
        valid_filters = []

        for f in filters:
            m = (self.db_tbl["filter"] == f) & (self.db_tbl["exptime"] == 100)
            ul5 = np.asarray(self.db_tbl["ul5"][m], dtype=float)
            ul5 = ul5[np.isfinite(ul5)]
            ul5 = np.asarray(self.db_tbl["ul5"][m], dtype=float)
            ul5 = ul5[np.isfinite(ul5) & (ul5 > 0)]

            if len(ul5) < 5:
                continue

            data.append(ul5)
            valid_filters.append(f)

        positions = np.arange(len(valid_filters))
        

        # ------------------------------
        # 2. Plot
        # ------------------------------
        fig, ax = plt.subplots(figsize=(9, 6))

        vp = ax.violinplot(
            data,
            positions=positions,
            widths=0.8,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )

        # ------------------------------
        # 3. Color each violin
        # ------------------------------
        for body, f, ul5 in zip(vp["bodies"], valid_filters, data):
            verts = body.get_paths()[0].vertices
            ymin = np.min(ul5)
            ymax = np.max(ul5)
            verts[:, 1] = np.clip(verts[:, 1], ymin, ymax)
            body.set_facecolor(colors[f])
            #body.set_edgecolor("black")
            body.set_alpha(0.7)

        # Median line style
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(2)

        # ------------------------------
        # 4. Optional: model point
        # ------------------------------
        for i, (f, ul5) in enumerate(zip(valid_filters, data)):
            # ------------------
            # model point
            # ------------------
            model = self.model.get(f)
            if model is not None:
                beta0 = model["intercept"]
                beta = np.asarray(model["coefficients"])
                med = model["medians"]

                ul5_model = (
                    beta0
                    + beta[0] * med[0]
                    + beta[1] * med[1]
                    + beta[2] * med[2]
                    + beta[3] * med[3]
                )

                ax.scatter(
                    i,
                    ul5_model,
                    s=70,
                    color="white",
                    edgecolor="black",
                    zorder=5,
                )

            # ------------------
            # upper label (ALL filters)
            # ------------------
            # robust upper edge (avoid tails)
            ul5_top = np.nanpercentile(ul5, 99)
            ul5_median = np.nanpercentile(ul5, 50)

            ax.text(
                i,
                ul5_top + 0.55,            # small vertical offset
                f"{ul5_median:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                rotation=90,
                color=colors[f],
                clip_on=False,
                zorder=6,
            )
        # ------------------------------
        # 5. Formatting
        # ------------------------------
        ax.set_xticks(positions)
        ax.set_xticklabels(valid_filters, fontsize=12, rotation = 45)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylabel("Limiting magnitude (5σ)", fontsize=16)
        ax.set_title(f"Limiting magnitude distribution (exptime={exptime}s)", fontsize=16)

        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()


        

#%%
if __name__ == "__main__":
    since_days: int = 360
    depth_table: Table = None
    model_save_path: str = None
    depth_table_save_path: str = None
    n_processes: int = 32
    
    self = ExposureCalculator()
    ra = 30.96774
    dec = 4.24528
    obsdate = Time.now() - 5 * u.hour
    filter = 'all'
    magnitude = 20
    exptime =100
    snr_ref = 5
    exptime_ref = 300
    n_points = 100
    snr_range = (1, 50)
    exptime_range = (1, 1000)
    moon_phase = 0.5
    moon_separation = None
    hour_since_sunset = None
    seeing = None
    obs_start_date = '2025-01-01'
    obs_end_date = '2026-01-01'
    # self.update_depth_model(obs_start_date = obs_start_date, obs_end_date = obs_end_date, depth_table = depth_table, model_save_path = model_save_path, depth_table_save_path = depth_table_save_path, n_processes = n_processes)
    self.db_tbl = Table.read('/home/hhchoi1022/code/BRIDGE/bridge/utils/depth_data_20260329052547.csv')
    # self.load_model()
    # self.plot_depth_model(exptime = 100)
    self.fit_depth_model(self.db_tbl, 'i', visualize = True)
    # result_snr = self.calculate_snr(filter = 'all', magnitude= 19.5, exptime = 300, obsdate = obsdate)
    # result_exptime = self.calculate_exptime(filter = 'all', magnitude= 19, snr = 10, obsdate = obsdate)
    # self.plot_exptime_vs_snr(filter = 'all', magnitude= 19.5, ra = ra, dec = dec, obsdate = obsdate)
    # self.plot_snr_vs_exptime(filter = 'all', magnitude= 19.5, ra = ra, dec = dec, obsdate = obsdate)
    
# %%