#%%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from astropy.time import Time

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt




Number = Union[int, float, np.number]
#%%

DEFAULT_SOURCE_MAP = {
    "Ia": "salt2",
    "Ib": "nugent",
    "Ic": "nugent",
    "IIb": "nugent",
    "IIP": "nugent",
    "IIL": "nugent",
    "IIn": "nugent",
    "II": "pycoco",
}


@dataclass
class TemplateEstimate:
    last_expected_mjd: float
    last_expected_phase: float
    last_expected_absmag: float
    last_expected_mag: Optional[float] = None
    last_expected_trend: str = None
    last_expected_filter: str = None
    last_expected_transient_source: str = None
    last_expected_transient_type: str = None
    last_expected_is_extrapolated: Optional[bool] = None


class Template:
    """
    Template light-curve handler.

    Parameters
    ----------
    transient_type : str
        Transient class, e.g. 'Ia', 'Ib', 'IIP'.
    observed_mjd : float
        Observed time. This is required.
    observed_absmag : float, optional
        Observed absolute magnitude. If given without observed_phase,
        phase will be estimated from the template using `trend`.
    observed_phase : float, optional
        Observed phase. If given without observed_absmag,
        absmag will be estimated from the template.
    filter_ : str, optional
        Photometric filter name. Default is 'sdssr'.
    trend : str, optional
        'rising' or 'falling'. Used when phase must be inferred
        from magnitude. Default is 'rising'.
    source : str, optional
        Template source name. If None, use DEFAULT_SOURCE_MAP.
    template_dir : str or Path, optional
        Directory containing template csv files.
    """

    def __init__(
        self,
        transient_type: str,
        observed_mjd: Number,
        observed_phase: Optional[Number] = None,
        observed_absmag: Optional[Number] = None,
        observed_mag: Optional[Number] = None,
        filter_: str = "sdssr",
        trend: str = "rising",
        transient_source: Optional[str] = None,
        template_dir: Union[str, Path] = f"{Path(__file__).parent}/templates",
    ):
        self.transient_type = str(transient_type)
        self.observed_mjd = float(observed_mjd)
        self.filter_ = str(filter_)
        self.trend = str(trend).lower()
        self.template_dir = Path(template_dir)

        if self.trend not in {"rising", "falling"}:
            raise ValueError("trend must be 'rising' or 'falling'")

        self.transient_source = self._resolve_transient_source(transient_source)
        self.template_path = self._resolve_template_path()

        self.table = pd.read_csv(self.template_path)
        self._validate_table()
        self._setup_interpolators()

        self.observed_phase = None if observed_phase is None else float(observed_phase)
        self.observed_absmag = None if observed_absmag is None else float(observed_absmag)
        self.observed_mag = None if observed_mag is None else float(observed_mag)

        # Resolve missing quantity
        if self.observed_phase is None and self.observed_absmag is None:
            raise ValueError(
                "At least one of observed_absmag or observed_phase must be provided."
            )

        if self.observed_phase is None:
            self.observed_phase = self.estimate_phase_from_absmag(
                self.observed_absmag,
                trend=self.trend,
            )

        if self.observed_absmag is None:
            self.observed_absmag = float(self._template_absmag(self.observed_phase))
            
    def __repr__(self):
        string = f"LightCurveTemplate(transient_type={self.transient_type}, transient_source={self.transient_source}, filter={self.filter_})" 
        len_title = len(string)
        string += "\n" + "=" * len_title
        string += f"\n  • observed_mjd = {self.observed_mjd}"
        string += f"\n  • observed_phase = {self.observed_phase}"
        string += f"\n  • observed_absmag = {self.observed_absmag}"
        string += f"\n  • observed_mag = {self.observed_mag}"
        string += f"\n  • trend = {self.trend}"
        string += "\n" + "=" * len_title
        return string
    
    def mjd_to_phase(self, mjd):
        """
        Convert observed MJD to template phase using the observed anchor point.

        phase(mjd) = observed_phase + (mjd - observed_mjd)

        Parameters
        ----------
        mjd : float or array-like
            Observed MJD value(s).

        Returns
        -------
        float or np.ndarray
            Template phase(s). Returns a float for scalar input and a numpy array
            for array-like input.
        """
        mjd_arr = np.asarray(mjd, dtype=float)
        phase = self.observed_phase + (mjd_arr - self.observed_mjd)

        if np.ndim(mjd_arr) == 0:
            return float(phase)
        return phase

    def phase_to_mjd(self, phase):
        """
        Convert template phase to observed MJD using the observed anchor point.
        """
        phase_arr = np.asarray(phase, dtype=float)
        mjd = self.observed_mjd + (phase_arr - self.observed_phase)

        if np.ndim(phase_arr) == 0:
            return float(mjd)
        return mjd

    def _resolve_transient_source(self, transient_source: Optional[str]) -> str:
        if transient_source is not None:
            return str(transient_source)

        if self.transient_type not in DEFAULT_SOURCE_MAP:
            raise ValueError(
                f"No default source is defined for transient_type='{self.transient_type}'. "
                f"Please provide transient_source explicitly."
            )
        return DEFAULT_SOURCE_MAP[self.transient_type]

    def _resolve_template_path(self) -> Path:
        filename = f"{self.transient_type}_{self.transient_source}_{self.filter_}.csv"
        path = self.template_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Template file not found: {path}")
        return path

    def _validate_table(self) -> None:
        required = {"phase", "absmag"}
        missing = required - set(self.table.columns)
        if missing:
            raise ValueError(
                f"Template file must contain columns {sorted(required)}. "
                f"Missing: {sorted(missing)}"
            )

        self.table = self.table.sort_values("phase").reset_index(drop=True)

        if len(self.table) < 2:
            raise ValueError("Template file must contain at least two rows.")

    def _setup_interpolators(self) -> None:
        self.phase_grid = self.table["phase"].to_numpy(dtype=float)
        self.absmag_grid = self.table["absmag"].to_numpy(dtype=float)

        if "is_extrapolated" in self.table.columns:
            self.extrapolated_grid = self.table["is_extrapolated"].astype(bool).to_numpy()
        else:
            self.extrapolated_grid = np.zeros_like(self.phase_grid, dtype=bool)

        self._absmag_interp = interp1d(
            self.phase_grid,
            self.absmag_grid,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        self._extra_interp = interp1d(
            self.phase_grid,
            self.extrapolated_grid.astype(float),
            kind="nearest",
            bounds_error=False,
            fill_value=(
                float(self.extrapolated_grid[0]),
                float(self.extrapolated_grid[-1]),
            ),
        )

    def _template_absmag(self, phase: Union[Number, np.ndarray]) -> np.ndarray:
        phase = np.asarray(phase, dtype=float)
        return np.asarray(self._absmag_interp(phase), dtype=float)

    def _is_extrapolated(self, phase: Union[Number, np.ndarray]) -> np.ndarray:
        phase = np.asarray(phase, dtype=float)
        return np.asarray(self._extra_interp(phase) > 0.5, dtype=bool)

    def estimate_phase_from_absmag(
        self,
        absmag: float,
        trend: str = "rising",
    ) -> float:
        """
        Estimate phase from absolute magnitude using the closest template value.

        Rules
        -----
        - rising  -> search only phase <= 0
        - falling -> search only phase >= 0
        - no exact matching; return the closest phase
        - if absmag is out of the allowed magnitude range:
            rising  -> return -np.inf
            falling -> return  np.inf
        """
        absmag = float(absmag)
        trend = trend.lower()

        if trend not in {"rising", "falling"}:
            raise ValueError("trend must be 'rising' or 'falling'")

        phases = self.phase_grid
        mags = self.absmag_grid

        if trend == "rising":
            mask = phases <= 0
            out_of_range_value = -np.inf
        else:
            mask = phases >= 0
            out_of_range_value = np.inf

        phases_sub = phases[mask]
        mags_sub = mags[mask]

        finite = np.isfinite(phases_sub) & np.isfinite(mags_sub)
        phases_sub = phases_sub[finite]
        mags_sub = mags_sub[finite]

        if len(phases_sub) == 0:
            return out_of_range_value

        mag_min = np.min(mags_sub)
        mag_max = np.max(mags_sub)

        # out of range
        if (absmag < mag_min) or (absmag > mag_max):
            return out_of_range_value

        # closest magnitude
        idx = np.argmin(np.abs(mags_sub - absmag))
        return float(phases_sub[idx])

    def estimate_absmag_from_phase(self, phase: Number) -> float:
        return float(self._template_absmag(phase))
    
    def estimate_from_mjd(
        self,
        mjd: float) -> TemplateEstimate:
        """
        Estimate magnitude(s) at given observed time(s).

        Parameters
        ----------
        time : float or array-like
            Observed time(s) in the same system as `observed_mjd`.
        observed_mag : float, optional
            Apparent magnitude at the observed anchor point. If given,
            estimated apparent magnitudes are also returned.

        Returns
        -------
        TemplateEstimate
            phase, absmag, mag, is_extrapolated
        """
        phase = self.mjd_to_phase(mjd)
        return self.estimate_from_phase(phase=phase)
    
    def estimate_from_phase(
        self,
        phase: Union[Number, np.ndarray],
        ) -> TemplateEstimate:
        """
        Estimate absolute magnitude at target phase(s).

        Since this class is normalized only by one of
        (observed_phase, observed_absmag), this method returns the
        template-based representative magnitude.

        If observed_mag is given, the same magnitude difference relative to
        the observed point is propagated to magnitude.
        """
        phase = np.asarray(phase, dtype=float)
        est_absmag = self._template_absmag(phase)
        est_extra = self._is_extrapolated(phase)

        est_mag = None
        if self.observed_mag is not None:
            delta_mag = est_absmag - self.observed_absmag
            est_mag = np.asarray(float(self.observed_mag) + delta_mag, dtype=float)

        return TemplateEstimate(
            last_expected_mjd = self.observed_mjd,
            last_expected_phase=phase,
            last_expected_absmag=est_absmag,
            last_expected_mag=est_mag,
            last_expected_trend=self.trend,
            last_expected_filter=self.filter_,
            last_expected_transient_source=self.transient_source,
            last_expected_transient_type=self.transient_type,
            last_expected_is_extrapolated=est_extra,
        )

    def plot(
        self,
        estimate_mjds: float | list[float] | np.ndarray | None = None,
        phase_min: float | None = None,
        phase_max: float | None = None,
        nphase_grid: int = 1000,
        show_observed: bool = True,
        show_extrapolated: bool = True,
        show_legend: bool = True,
        title: str | None = None,
        ax=None,
        abs_ylim: tuple[float, float] | None = None,
        abs_pad: float = 1.5,
        text_offset_phase: float = 5, 
        text_offset_abs: float = 0
        ):
        import numpy as np
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax_abs = plt.subplots(figsize=(9, 5.5))
        else:
            ax_abs = ax
            fig = ax_abs.figure

        ax_app = ax_abs.twinx()

        if phase_min is None:
            phase_min = float(np.min(self.phase_grid))
        if phase_max is None:
            phase_max = float(np.max(self.phase_grid))

        phase_plot = np.linspace(phase_min, phase_max, nphase_grid)
        absmag_plot = self._template_absmag(phase_plot)

        # constant offset between absolute and apparent magnitude
        offset = None
        if self.observed_mag is not None:
            offset = self.observed_mag - self.observed_absmag

        # template: black
        ax_abs.plot(
            phase_plot,
            absmag_plot,
            color="black",
            lw=2.0,
            label="Template",
        )

        # extrapolated region: grey
        if show_extrapolated:
            extra = self._is_extrapolated(phase_plot)
            if np.any(extra):
                extra_int = extra.astype(int)
                diff = np.diff(np.r_[0, extra_int, 0])
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]

                for i, (s, e) in enumerate(zip(starts, ends)):
                    ax_abs.axvspan(
                        phase_plot[s],
                        phase_plot[e - 1],
                        color="gray",
                        alpha=0.18,
                        label="Extrapolated" if i == 0 else None,
                    )

        # peak marker
        ax_abs.axvline(0.0, ls=":", lw=1.2, color="gray", label="Peak")

        # observed point: blue
        if show_observed:
            obs_label = 'Observed (%.2f (%.2f), Phase=%.2f)'%(self.observed_absmag, self.observed_mag, self.observed_phase)

            if np.isfinite(self.observed_phase) and np.isfinite(self.observed_absmag):
                ax_abs.scatter(
                    [self.observed_phase],
                    [self.observed_absmag],
                    s=70,
                    color="blue",
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=6,
                    label=obs_label,
                )

            if self.observed_mag is not None and np.isfinite(self.observed_phase):
                ax_app.scatter(
                    [self.observed_phase],
                    [self.observed_mag],
                    s=70,
                    marker="o",
                    color="blue",
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=6,
                )

        # estimated points: red + vertical text
        est_absmag = None
        est_appmag = None
        if estimate_mjds is None:
            estimate_mjds = Time.now().mjd
        estimate_mjds = np.atleast_1d(estimate_mjds).astype(float)
        estimate_phases = self.mjd_to_phase(estimate_mjds)
        est_absmag = np.asarray(self._template_absmag(estimate_phases), dtype=float)

        ax_abs.scatter(
            estimate_phases,
            est_absmag,
            s=55,
            marker="o",
            color="red",
            edgecolor="black",
            linewidth=0.5,
            zorder=7,
            label="Estimated",
        )

        if offset is not None:
            est_appmag = est_absmag + offset
            ax_app.scatter(
                estimate_phases,
                est_appmag,
                s=55,
                marker="o",
                color="red",
                edgecolor="black",
                linewidth=0.5,
                zorder=7,
            )

        for i, ph in enumerate(np.atleast_1d(estimate_phases)):
            if est_appmag is not None:
                txt = f"Phase = {ph:.2f}, M = {est_absmag[i]:.2f} ({est_appmag[i]:.2f})"
            else:
                txt = f"Phase = {ph:.2f}, M = {est_absmag[i]:.2f}"

            ax_abs.text(
                ph + text_offset_phase, 
                est_absmag[i] + text_offset_abs,
                txt,
                color="red",
                fontsize=9,
                ha="left",
                va="center",
                rotation=0,
                zorder=8,
            )

        # y-limits based on absolute magnitude
        if abs_ylim is None:
            vals = list(np.ravel(absmag_plot))

            if show_observed and self.observed_absmag is not None and np.isfinite(self.observed_absmag):
                vals.append(self.observed_absmag)

            if est_absmag is not None:
                vals.extend(np.ravel(est_absmag))

            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]

            bright_side = np.min(vals) - abs_pad   # keep as-is / automatic
            faint_side = -5.0                      # fixed

            abs_ylim = (bright_side, faint_side)

        ax_abs.set_ylim(abs_ylim)

        # sync apparent axis from absolute axis
        if offset is not None:
            y0, y1 = ax_abs.get_ylim()
            ax_app.set_ylim(y0 + offset, y1 + offset)

        ax_abs.set_xlabel("Phase [days]")
        ax_abs.set_ylabel("Absolute Magnitude")
        ax_app.set_ylabel("Apparent Magnitude")

        ax_abs.invert_yaxis()
        ax_app.invert_yaxis()

        if title is None:
            title = f"{self.transient_type} | {self.transient_source} | {self.filter_}"
        ax_abs.set_title(title)

        if show_legend:
            h1, l1 = ax_abs.get_legend_handles_labels()
            h2, l2 = ax_app.get_legend_handles_labels()

            seen = set()
            handles = []
            labels = []
            for h, l in zip(h1 + h2, l1 + l2):
                if l and l not in seen:
                    handles.append(h)
                    labels.append(l)
                    seen.add(l)

            ax_abs.legend(handles, labels, loc="best")

        return ax_abs, ax_app, fig
# %%
if __name__ == "__main__":
    template = Template(transient_type = 'Ia', observed_mjd = 0, observed_absmag = -17, observed_mag = 25, filter_ = 'sdssg', trend = 'falling', transient_source = 'salt2')

