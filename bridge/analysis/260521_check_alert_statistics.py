
"""
Code summary

    In this code,
        - We will check the statistics of the alerts in the database over time

    There are 5 types of alert DB.
        - ztf_alerce_alerts
        - ztf_fink_alerts
        - lsst_alerce_alerts
        - lsst_fink_alerts
        - tns_alerts
"""
#%%
from bridge.connector import SQLConnector
sql_connector = SQLConnector()
#%% All sources
all_data_dict = {}
all_data_dict['lsst_fink'] = sql_connector.get_data('lsst_fink')
all_data_dict['lsst_alerce'] = sql_connector.get_data('lsst_alerce')
all_data_dict['ztf_fink'] = sql_connector.get_data('ztf_fink')
all_data_dict['ztf_alerce'] = sql_connector.get_data('ztf_alerce')
all_data_dict['tns'] = sql_connector.get_data('tns')
#%% Removal of duplicated id
from astropy.table import Table
id_key_dict = {}
id_key_dict['lsst_fink'] = 'diaSource_diaObjectId'
id_key_dict['lsst_alerce'] = 'diaObjectId'
id_key_dict['ztf_fink'] = 'objectId'
id_key_dict['ztf_alerce'] = 'oid'
id_key_dict['tns'] = 'ID'
all_object_dict = all_data_dict.copy()
for key, value in all_object_dict.items():
    df = value.to_pandas()
    df = df.drop_duplicates(subset = id_key_dict[key])
    all_object_dict[key] = Table.from_pandas(df)
# %% 1. Plot the number of alerts over time
mjd_key_dict = {}
mjd_key_dict['lsst_fink'] = 'diaSource_midpointMjdTai'
mjd_key_dict['lsst_alerce'] = 'mjd'
mjd_key_dict['ztf_fink'] = 'candidate_jd'
mjd_key_dict['ztf_alerce'] = 'mjd'
mjd_key_dict['tns'] = 'Discovery Date (UT)'
from astropy.time import Time
# Add mjd column to all_object_dict
all_data_dict['lsst_fink']['mjd'] = Time(all_data_dict['lsst_fink'][mjd_key_dict['lsst_fink']], format='mjd').mjd
all_data_dict['lsst_alerce']['mjd'] = Time(all_data_dict['lsst_alerce'][mjd_key_dict['lsst_alerce']], format='mjd').mjd
all_data_dict['ztf_fink']['mjd'] = Time(all_data_dict['ztf_fink'][mjd_key_dict['ztf_fink']], format='jd').mjd
all_data_dict['ztf_alerce']['mjd'] = Time(all_data_dict['ztf_alerce'][mjd_key_dict['ztf_alerce']], format='mjd').mjd
all_data_dict['tns']['mjd'] = Time(all_data_dict['tns'][mjd_key_dict['tns']], format='iso').mjd

all_object_dict['lsst_fink']['mjd'] = Time(all_object_dict['lsst_fink'][mjd_key_dict['lsst_fink']], format='mjd').mjd
all_object_dict['lsst_alerce']['mjd'] = Time(all_object_dict['lsst_alerce'][mjd_key_dict['lsst_alerce']], format='mjd').mjd
all_object_dict['ztf_fink']['mjd'] = Time(all_object_dict['ztf_fink'][mjd_key_dict['ztf_fink']], format='jd').mjd
all_object_dict['ztf_alerce']['mjd'] = Time(all_object_dict['ztf_alerce'][mjd_key_dict['ztf_alerce']], format='mjd').mjd
all_object_dict['tns']['mjd'] = Time(all_object_dict['tns'][mjd_key_dict['tns']], format='iso').mjd

# %% Daily counts and plots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
})

SOURCE_ORDER = ['lsst_fink', 'lsst_alerce', 'ztf_fink', 'ztf_alerce', 'tns']
SOURCE_LABELS = {
    'lsst_fink': 'LSST Fink',
    'lsst_alerce': 'LSST ALeRCE',
    'ztf_fink': 'ZTF Fink',
    'ztf_alerce': 'ZTF ALeRCE',
    'tns': 'TNS',
}
SOURCE_COLORS = {
    'lsst_fink': '#2563eb',
    'lsst_alerce': '#ea580c',
    'ztf_fink': '#16a34a',
    'ztf_alerce': '#dc2626',
    'tns': '#9333ea',
}
ROLLING_WINDOW_DAYS = 7
MAX_GAP_DAYS = 14


def daily_counts_from_mjd(mjd_values):
    """Return a Series indexed by date (midnight UTC) with counts per day."""
    dates = pd.DatetimeIndex(Time(mjd_values, format='mjd').to_datetime()).normalize()
    return dates.value_counts().sort_index()


def smooth_daily_counts(daily, window_days=ROLLING_WINDOW_DAYS):
    """Reindex to calendar days (NaN where no alerts), then centered rolling mean."""
    if daily.empty:
        return daily, daily
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    sparse = daily.reindex(idx).astype(float)
    smoothed = sparse.rolling(window_days, min_periods=1, center=True).mean()
    # Hide smoothed curve on long stretches with no underlying alerts
    had_alert = sparse.notna().astype(float)
    near_data = had_alert.rolling(window_days, min_periods=1, center=True).sum() > 0
    smoothed = smoothed.where(near_data)
    return sparse, smoothed


def iter_line_segments(index, values, max_gap_days=MAX_GAP_DAYS):
    """Yield (x, y) slices; break on large date gaps or non-finite values."""
    index = pd.DatetimeIndex(index)
    values = np.asarray(values, dtype=float)
    if len(index) == 0:
        return
    start = 0
    for i in range(1, len(index)):
        gap = (index[i] - index[i - 1]).days
        invalid = not (np.isfinite(values[i]) and np.isfinite(values[i - 1]))
        if gap > max_gap_days or invalid:
            seg = values[start:i]
            mask = np.isfinite(seg)
            if mask.any():
                yield index[start:i][mask], seg[mask]
            start = i
    seg = values[start:]
    mask = np.isfinite(seg)
    if mask.any():
        yield index[start:][mask], seg[mask]


def plot_gap_aware_lines(ax, index, values, *, color, label, max_gap_days=MAX_GAP_DAYS, **kwargs):
    first = True
    for x, y in iter_line_segments(index, values, max_gap_days=max_gap_days):
        ax.plot(
            x, y,
            color=color,
            label=label if first else '_nolegend_',
            **kwargs,
        )
        first = False


def daily_object_alert_ratio(object_dict, alert_dict, key):
    """Daily ratio unique objects / alerts for one source (NaN when no alerts)."""
    obj_daily = daily_counts_from_mjd(object_dict[key]['mjd'])
    alert_daily = daily_counts_from_mjd(alert_dict[key]['mjd'])
    if alert_daily.empty:
        return obj_daily.iloc[0:0].astype(float)
    idx = obj_daily.index.union(alert_daily.index)
    obj_daily = obj_daily.reindex(idx, fill_value=0).astype(float)
    alert_daily = alert_daily.reindex(idx, fill_value=0).astype(float)
    ratio = obj_daily / alert_daily.replace(0, np.nan)
    return ratio.dropna()


def smooth_daily_ratio(ratio, window_days=ROLLING_WINDOW_DAYS):
    """Rolling mean of daily ratio on calendar grid; NaN outside active periods."""
    if ratio.empty:
        return ratio, ratio
    idx = pd.date_range(ratio.index.min(), ratio.index.max(), freq='D')
    sparse = ratio.reindex(idx).astype(float)
    smoothed = sparse.rolling(window_days, min_periods=1, center=True).mean()
    had_ratio = sparse.notna().astype(float)
    near_data = had_ratio.rolling(window_days, min_periods=1, center=True).sum() > 0
    smoothed = smoothed.where(near_data)
    return sparse, smoothed


def _style_time_axis(ax):
    ax.set_xlabel('Date')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.grid(True, which='major', alpha=0.35)
    ax.grid(True, which='minor', alpha=0.15)


def _style_ratio_axis(ax):
    _style_time_axis(ax)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='0.35', ls='--', lw=1, zorder=0)


def plot_daily_counts(data_dict, title, ylabel, save_path, log_y=True, show_daily=False):
    """
    Overlay plot: faint daily counts + bold 7-day rolling mean; gap-aware lines.
    Log scale by default so low-volume sources stay visible next to spikes.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    for key in SOURCE_ORDER:
        daily = daily_counts_from_mjd(data_dict[key]['mjd'])
        if daily.empty:
            continue
        color = SOURCE_COLORS[key]
        label = SOURCE_LABELS[key]
        sparse, smoothed = smooth_daily_counts(daily)

        if show_daily:
            plot_gap_aware_lines(
                ax, daily.index, daily.values,
                color=color, label='_nolegend_',
                lw=0.8, alpha=0.3, zorder=1,
            )

        plot_gap_aware_lines(
            ax,
            smoothed.index,
            smoothed.values,
            color=color, label=label,
            lw=2.2, alpha=0.95, zorder=3,
            max_gap_days=MAX_GAP_DAYS,
        )

    if log_y:
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title}\n({ROLLING_WINDOW_DAYS}-day rolling mean; gaps > {MAX_GAP_DAYS} d not connected)')
    _style_time_axis(ax)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.14),
        ncol=len(SOURCE_ORDER),
        frameon=True,
        columnspacing=1.2,
    )
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    return fig


def plot_daily_counts_faceted(data_dict, title, ylabel, save_path, log_y=True):
    """One panel per source — easiest to read when scales differ widely."""
    n = len(SOURCE_ORDER)
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.6 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, SOURCE_ORDER):
        daily = daily_counts_from_mjd(data_dict[key]['mjd'])
        color = SOURCE_COLORS[key]
        if daily.empty:
            ax.set_ylabel(SOURCE_LABELS[key], rotation=0, ha='right', va='center')
            ax.text(0.02, 0.5, 'no data', transform=ax.transAxes, color='0.45')
            continue

        sparse, smoothed = smooth_daily_counts(daily)
        plot_gap_aware_lines(
            ax, daily.index, daily.values,
            color=color, label='daily',
            lw=0.9, alpha=0.35, zorder=1,
        )
        plot_gap_aware_lines(
            ax, smoothed.index, smoothed.values,
            color=color, label=f'{ROLLING_WINDOW_DAYS}d mean',
            lw=2.0, alpha=1.0, zorder=3,
        )
        if log_y:
            ax.set_yscale('log')
            ax.set_ylim(bottom=0.8)
        ax.set_ylabel(SOURCE_LABELS[key], rotation=0, ha='right', va='center', fontsize=10)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date')
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    return fig


def plot_daily_ratio(object_dict, alert_dict, title, save_path, show_daily=False):
    """Overlay: unique objects / alerts per day (≤ 1)."""
    fig, ax = plt.subplots(figsize=(8, 4))

    for key in SOURCE_ORDER:
        ratio = daily_object_alert_ratio(object_dict, alert_dict, key)
        if ratio.empty:
            continue
        color = SOURCE_COLORS[key]
        label = SOURCE_LABELS[key]
        sparse, smoothed = smooth_daily_ratio(ratio)

        if show_daily:
            plot_gap_aware_lines(
                ax, ratio.index, ratio.values,
                color=color, label='_nolegend_',
                lw=0.8, alpha=0.3, zorder=1,
            )
        plot_gap_aware_lines(
            ax, smoothed.index, smoothed.values,
            color=color, label=label,
            lw=2.2, alpha=0.95, zorder=3,
        )

    ax.set_ylabel('Objects / alerts')
    ax.set_title(
        f'{title}\n({ROLLING_WINDOW_DAYS}-day rolling mean; gaps > {MAX_GAP_DAYS} d not connected)'
    )
    _style_ratio_axis(ax)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.14),
        ncol=len(SOURCE_ORDER),
        frameon=True,
        columnspacing=1.2,
    )
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    return fig


def _plot_smoothed_count(ax, daily, *, color, label, linestyle='-', lw=2.0, zorder=3):
    """Plot gap-aware 7-day rolling mean for a daily count series."""
    if daily.empty:
        return
    _, smoothed = smooth_daily_counts(daily)
    plot_gap_aware_lines(
        ax, smoothed.index, smoothed.values,
        color=color, label=label, linestyle=linestyle,
        lw=lw, alpha=0.95, zorder=zorder,
    )


def plot_alerts_and_objects_together(
    alert_dict,
    object_dict,
    title,
    save_path,
    log_y=True,
    faceted=True,
):
    """
    Plot alerts and unique objects on the same axes.
    Solid = alerts, dashed = objects; color encodes source.
    """
    subtitle = (
        f'({ROLLING_WINDOW_DAYS}-day rolling mean; solid=alerts, dashed=objects; '
        f'gaps > {MAX_GAP_DAYS} d not connected)'
    )

    if faceted:
        n = len(SOURCE_ORDER)
        fig, axes = plt.subplots(n, 1, figsize=(8, 1.6 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, key in zip(axes, SOURCE_ORDER):
            color = SOURCE_COLORS[key]
            alert_daily = daily_counts_from_mjd(alert_dict[key]['mjd'])
            obj_daily = daily_counts_from_mjd(object_dict[key]['mjd'])
            if alert_daily.empty and obj_daily.empty:
                ax.set_ylabel(SOURCE_LABELS[key], rotation=0, ha='right', va='center')
                ax.text(0.02, 0.5, 'no data', transform=ax.transAxes, color='0.45')
                continue
            _plot_smoothed_count(ax, alert_daily, color=color, label='alerts', linestyle='-')
            _plot_smoothed_count(ax, obj_daily, color=color, label='objects', linestyle='--')
            if log_y:
                ax.set_yscale('log')
                ax.set_ylim(bottom=0.8)
            ax.set_ylabel(SOURCE_LABELS[key], rotation=0, ha='right', va='center', fontsize=10)
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('Date')
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.suptitle(f'{title}\n{subtitle}', fontsize=14, y=1.01)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        for key in SOURCE_ORDER:
            color = SOURCE_COLORS[key]
            alert_daily = daily_counts_from_mjd(alert_dict[key]['mjd'])
            obj_daily = daily_counts_from_mjd(object_dict[key]['mjd'])
            src = SOURCE_LABELS[key]
            _plot_smoothed_count(
                ax, alert_daily, color=color, label=f'{src} (alerts)', linestyle='-',
            )
            _plot_smoothed_count(
                ax, obj_daily, color=color, label=f'{src} (objects)', linestyle='--',
            )
        if log_y:
            ax.set_yscale('log')
            ax.set_ylim(bottom=0.8)
        ax.set_ylabel('Count')
        ax.set_title(f'{title}\n{subtitle}')
        _style_time_axis(ax)
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            fontsize=8,
            frameon=True,
        )
        fig.subplots_adjust(bottom=0.30)

    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    return fig


def plot_alerts_objects_ratio_dashboard(alert_dict, object_dict, save_path, log_counts=True):
    """Stacked figure: alerts, objects, and ratio (shared date axis)."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    for key in SOURCE_ORDER:
        color = SOURCE_COLORS[key]
        label = SOURCE_LABELS[key]
        alert_daily = daily_counts_from_mjd(alert_dict[key]['mjd'])
        obj_daily = daily_counts_from_mjd(object_dict[key]['mjd'])
        _plot_smoothed_count(axes[0], alert_daily, color=color, label=label)
        _plot_smoothed_count(axes[1], obj_daily, color=color, label=label)

        ratio = daily_object_alert_ratio(object_dict, alert_dict, key)
        if not ratio.empty:
            _, smoothed = smooth_daily_ratio(ratio)
            plot_gap_aware_lines(
                axes[2], smoothed.index, smoothed.values,
                color=color, label=label, lw=2.0, alpha=0.95, zorder=3,
            )

    if log_counts:
        for ax in axes[:2]:
            ax.set_yscale('log')
            ax.set_ylim(bottom=0.8)
    axes[0].set_ylabel('Alerts')
    axes[1].set_ylabel('Objects')
    axes[2].set_ylabel('Objects / alerts')
    _style_ratio_axis(axes[2])
    axes[0].set_title(
        f'Alerts, objects, and ratio by source\n'
        f'({ROLLING_WINDOW_DAYS}-day rolling mean)'
    )
    for ax in axes[:2]:
        ax.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.3)
    axes[2].legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.9)
    axes[2].set_xlabel('Date')
    axes[2].xaxis.set_major_locator(mdates.MonthLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    return fig


def plot_daily_ratio_faceted(object_dict, alert_dict, title, save_path):
    """Faceted unique objects / alerts per source."""
    n = len(SOURCE_ORDER)
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.6 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, SOURCE_ORDER):
        ratio = daily_object_alert_ratio(object_dict, alert_dict, key)
        color = SOURCE_COLORS[key]
        if ratio.empty:
            ax.set_ylabel(SOURCE_LABELS[key], rotation=0, ha='right', va='center')
            ax.text(0.02, 0.5, 'no data', transform=ax.transAxes, color='0.45')
            continue
        sparse, smoothed = smooth_daily_ratio(ratio)
        plot_gap_aware_lines(
            ax, ratio.index, ratio.values,
            color=color, label='daily',
            lw=0.9, alpha=0.35, zorder=1,
        )
        plot_gap_aware_lines(
            ax, smoothed.index, smoothed.values,
            color=color, label=f'{ROLLING_WINDOW_DAYS}d mean',
            lw=2.0, alpha=1.0, zorder=3,
        )
        ax.set_ylabel(SOURCE_LABELS[key], rotation=0, ha='right', va='center', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color='0.35', ls='--', lw=1, zorder=0)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date')
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    return fig


output_dir = Path(__file__).resolve().parent
#%%
# 1. All alerts — overlay (log scale + smoothed)
plot_daily_counts(
    all_data_dict,
    title='Daily alert count by source',
    ylabel='Number of alerts',
    save_path=output_dir / '260521_daily_alerts_by_source.png',
    log_y=True,
    show_daily=True,
)

# 2. Unique objects — overlay (log scale + smoothed)
plot_daily_counts(
    all_object_dict,
    title='Daily unique object count by source',
    ylabel='Number of objects',
    save_path=output_dir / '260521_daily_objects_by_source.png',
    log_y=True,
    show_daily=True,
)

# 3. Faceted views (optional; clearer per-source trends)
plot_daily_counts_faceted(
    all_data_dict,
    title='Daily alerts by source (faceted)',
    ylabel='Number of alerts',
    save_path=output_dir / '260521_daily_alerts_by_source_faceted.png',
    log_y=True,
)
plot_daily_counts_faceted(
    all_object_dict,
    title='Daily unique objects by source (faceted)',
    ylabel='Number of objects',
    save_path=output_dir / '260521_daily_objects_by_source_faceted.png',
    log_y=True,
)

# 4. Ratio: unique objects / alerts per source
plot_daily_ratio(
    all_object_dict,
    all_data_dict,
    title='Daily ratio: unique objects / alerts by source',
    save_path=output_dir / '260521_daily_object_alert_ratio_by_source.png',
    show_daily=True,
)
plot_daily_ratio_faceted(
    all_object_dict,
    all_data_dict,
    title='Daily ratio: unique objects / alerts (faceted)',
    save_path=output_dir / '260521_daily_object_alert_ratio_by_source_faceted.png',
)

# 5. Alerts + objects together
plot_alerts_and_objects_together(
    all_data_dict,
    all_object_dict,
    title='Daily alerts and unique objects by source',
    save_path=output_dir / '260521_daily_alerts_objects_by_source_faceted.png',
    faceted=True,
)
plot_alerts_and_objects_together(
    all_data_dict,
    all_object_dict,
    title='Daily alerts and unique objects by source',
    save_path=output_dir / '260521_daily_alerts_objects_by_source.png',
    faceted=False,
)
plot_alerts_objects_ratio_dashboard(
    all_data_dict,
    all_object_dict,
    save_path=output_dir / '260521_daily_alerts_objects_ratio_dashboard.png',
)

# %%
