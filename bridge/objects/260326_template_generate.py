# %%
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sncosmo
from astropy.cosmology import Planck18
from scipy.interpolate import interp1d
from scipy.stats import linregress

def absolute_lc_from_source(
    source_name,
    peak_abs_mag,
    band='bessellb',
    magsys='ab',
    z=0.001,
    tmin=-20,
    tmax=100,
    ntime=400,
    model_params=None,
):
    """
    Return rest-frame phase and absolute magnitude light curve
    from a sncosmo source.
    """
    model = sncosmo.Model(source=source_name)

    params = {'z': z, 't0': 0.0}
    if model_params is not None:
        params.update(model_params)
    model.set(**params)

    model.set_source_peakabsmag(peak_abs_mag, band, magsys, cosmo=Planck18)

    t = np.linspace(tmin, tmax, ntime)
    m_app = model.bandmag(band, magsys, t)

    mu = Planck18.distmod(z).value
    M_abs = m_app - mu

    phase = (t - model.get('t0')) / (1.0 + z)

    return phase, M_abs


def build_peak_aligned_template(
    source_name='hsiao',
    peak_abs_mag=-19.3,
    band='bessellv',
    magsys='ab',
    z=0.05,
    native_tmin=-20,
    native_tmax=120,
    native_ntime=5000,
    save_phase_min=-30,
    save_phase_max=150,
    save_phase_step=1.0,
    model_params=None,
    extrapolation='powerlaw', # Changed default to our new method
    edge_points=10,
    save_path='hsiao_bessellb_template.csv',
):
    """
    Build and save an absolute-magnitude template with phase=0 at peak.
    """

    # Dense native LC
    phase_native, Mabs_native = absolute_lc_from_source(
        source_name=source_name,
        peak_abs_mag=peak_abs_mag,
        band=band,
        magsys=magsys,
        z=z,
        tmin=native_tmin,
        tmax=native_tmax,
        ntime=native_ntime,
        model_params=model_params,
    )

    finite = np.isfinite(phase_native) & np.isfinite(Mabs_native)
    phase_native = phase_native[finite]
    Mabs_native = Mabs_native[finite]

    # Find peak in this band: minimum absolute magnitude
    i_peak = np.argmin(Mabs_native)
    phase_peak = phase_native[i_peak]

    # Redefine phase so that peak = 0
    phase_peak_aligned = phase_native - phase_peak

    # Sort
    order = np.argsort(phase_peak_aligned)
    phase_peak_aligned = phase_peak_aligned[order]
    Mabs_native = Mabs_native[order]

    # Interpolate within native range
    interp_func = interp1d(
        phase_peak_aligned,
        Mabs_native,
        kind='linear',
        bounds_error=False,
        fill_value=np.nan,
    )

    phase_out = np.arange(save_phase_min, save_phase_max + save_phase_step, save_phase_step)
    Mabs_out = interp_func(phase_out)

    native_min = np.min(phase_peak_aligned)
    native_max = np.max(phase_peak_aligned)

    left_mask = phase_out < native_min
    right_mask = phase_out > native_max
    is_extrapolated = left_mask | right_mask

    n = min(edge_points, len(phase_peak_aligned))

    if extrapolation == 'powerlaw':
        # --- LEFT: Power Law Fit (idx=2) ---
        phase_early = phase_peak_aligned[:n]
        Mabs_early = Mabs_native[:n]
        
        # Convert Mag to sqrt(Flux) for linear fit
        sqrt_flux = 10**(-0.2 * Mabs_early)
        res = linregress(phase_early, sqrt_flux)
        
        sq_flux_extrap = res.slope * phase_out[left_mask] + res.intercept
        sq_flux_extrap = np.maximum(sq_flux_extrap, 1e-10) # Floor to avoid log issues
        Mabs_out[left_mask] = -5 * np.log10(sq_flux_extrap)

        # --- RIGHT: Linear Fit ---
        coef_right = np.polyfit(phase_peak_aligned[-n:], Mabs_native[-n:], 1)
        poly_right = np.poly1d(coef_right)
        Mabs_out[right_mask] = poly_right(phase_out[right_mask])

    elif extrapolation == 'linear':
        # left linear fit
        coef_left = np.polyfit(phase_peak_aligned[:n], Mabs_native[:n], 1)
        poly_left = np.poly1d(coef_left)
        Mabs_out[left_mask] = poly_left(phase_out[left_mask])

        # right linear fit
        coef_right = np.polyfit(phase_peak_aligned[-n:], Mabs_native[-n:], 1)
        poly_right = np.poly1d(coef_right)
        Mabs_out[right_mask] = poly_right(phase_out[right_mask])

    elif extrapolation == 'flat':
        Mabs_out[left_mask] = Mabs_native[0]
        Mabs_out[right_mask] = Mabs_native[-1]
    
    else:
        raise ValueError("extrapolation must be 'powerlaw', 'linear', or 'flat'")

    df = pd.DataFrame({
        'phase': phase_out,
        'absmag': Mabs_out,
        'is_extrapolated': is_extrapolated,
    })

    df['source_name'] = source_name
    df['band'] = band
    df['magsys'] = magsys
    df['peak_abs_mag_input'] = peak_abs_mag
    df['phase_zero'] = 'peak_in_selected_band'
    df['native_phase_min'] = native_min
    df['native_phase_max'] = native_max
    df['peak_phase_original'] = phase_peak

    df.to_csv(save_path, index=False)

    return df, phase_peak_aligned, Mabs_native
# %%
template_map = {
    'Ia_hsiao'    : ('hsiao',        -19.3, {}, 'Ia', 'hsiao', 'powerlaw'),
    'Ia_salt2'    : ('salt2',        -19.3, {}, 'Ia', 'salt2', 'powerlaw'),
    'Ia_salt3'    : ('salt3',        -19.3, {}, 'Ia', 'salt3', 'powerlaw'),
    'Ia_nugent'    : ('nugent-sn1a',  -19.3, {}, 'Ia', 'nugent', 'powerlaw'),
    'Ib/c_nugent'  : ('nugent-sn1bc', -17.8, {}, 'Ib', 'nugent', 'powerlaw'),
    'Ib/c_nugent'  : ('nugent-sn1bc', -17.8, {}, 'Ic', 'nugent', 'powerlaw'),
    'IIP_nugent'   : ('nugent-sn2p',  -16.7, {}, 'IIP', 'nugent', 'linear'),
    'IIL_nugent'   : ('nugent-sn2l',  -17.6, {}, 'IIL', 'nugent', 'linear'),
    'IIn_nugent'   : ('nugent-sn2n',  -18.5, {}, 'IIn', 'nugent', 'linear'),
    'v19-2016bkv-corr'   : ('v19-2016bkv-corr',  -18, {}, 'II', 'pycoco', 'linear'),
}

filter_list = ['sdssg', 'sdssr', 'sdssi', 'bessellb', 'bessellv', 'bessellr']
#%%
import os
os.makedirs('templates', exist_ok=True) # Ensure the directory exists

for band in filter_list:
    for source_name, peak_abs_mag, extra_params, transient_type, template_name, extrapolation in template_map.values():
        try:
            df_template, phase_native, Mabs_native = build_peak_aligned_template(
                source_name=source_name,
                peak_abs_mag=peak_abs_mag,
                band=band,
                magsys='ab',
                z=0.05,
                native_tmin=-20,
                native_tmax=120,
                native_ntime=5000,
                save_phase_min=-30,
                save_phase_max=300,
                save_phase_step=0.1,
                extrapolation=extrapolation, # Ensure it is using powerlaw
                save_path=f'templates/{transient_type}_{template_name}_{band}.csv',
            )
        except Exception as e:
            print(f'Error building template for {source_name} in {band}: {e}')
# %%
plt.figure(figsize=(8, 5))

from astropy.io import ascii
for band in filter_list:
    for source_name, peak_abs_mag, extra_params, transient_type, template_name, extrapolation in template_map.values():
        try:
            tbl = ascii.read(f'templates/{transient_type}_{template_name}_{band}.csv')
            plt.plot(tbl['phase'], tbl['absmag'], label=f'{source_name} {band}')
            plt.xlabel('Phase [days]')
            plt.ylabel('Absolute magnitude')
            plt.title('Absolute-magnitude templates (Power Law Early Extrapolation)')
            plt.tight_layout()

        except Exception as e:
            print(f'Error plotting template for {source_name} in {band}: {e}')
            
plt.gca().invert_yaxis()
plt.xlim(-20, -10)
# Uncomment the line below if you want the legend shown
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
plt.show()
# %%
