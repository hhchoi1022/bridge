#%%

from astropy.time import Time
import numpy as np
from tqdm import tqdm
from astropy.table import vstack
from astropy.coordinates import match_coordinates_sky
from astropy.coordinates import SkyCoord
import astropy.units as u
from typing import Optional, Union
from astropy.io import ascii

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

from ezphot.utils import DataBrowser
from ezphot.imageobjects import ScienceImage, ImageSet
from ezphot.methods.tract7dt import Tract7DTRunner
from ezphot.dataobjects import Catalog, CatalogSet
from bridge.configuration import Configuration
from bridge.connector import *
from bridge.objects import Alert
from bridge.utils import flexible_time_parser
#%%
# WORKERS FOR PARALLEL PROCESSING
def preprocess_worker(args):
    target_img, clear = args
    bias_img = None
    dark_img = None
    flat_img = None
    corrected_img = None
    try:
        bias_img = target_img.get_masterframe(imagetyp = 'BIAS', max_days = 1000)[0]
        dark_img = target_img.get_masterframe(imagetyp = 'DARK', max_days = 1000)[0]
        flat_img = target_img.get_masterframe(imagetyp = 'FLAT', max_days = 1000)[0]
        corrected_img = target_img.correct_bdf(bias_image = bias_img, dark_image = dark_img, flat_image = flat_img, save = True, verbose = False)
        
        target_img.clear(verbose = False)
        if bias_img is not None: bias_img.clear(verbose = False)
        if dark_img is not None: dark_img.clear(verbose = False)
        if flat_img is not None: flat_img.clear(verbose = False)
        if clear:
            if corrected_img is not None: corrected_img.clear(verbose = False)
        return {"success": True, "image": corrected_img, "error": None, "function": "preprocess"}
    
    except Exception as e:
        target_img.clear(verbose = False)
        if bias_img is not None: bias_img.clear(verbose = False)
        if dark_img is not None: dark_img.clear(verbose = False)
        if flat_img is not None: flat_img.clear(verbose = False)
        if corrected_img is not None: corrected_img.clear(verbose = False)
        return {"success": False, "image": target_img, "error": f"preprocess: {e}", "function": "preprocess"}

def platesolve_worker(args):
    target_img, clear = args
    corrected_img = None
    try:
        corrected_img = target_img.platesolve(
            overwrite=True,
            verbose=False,
            scamp_sexparams=None,
            scamp_params=None
        )
        
        target_img.clear(verbose=False)
        if clear:
            corrected_img.clear(verbose = False)
        return {"success": True, "image": corrected_img, "error": None, "function": "platesolve"}
    
    except Exception as e:
        target_img.clear(verbose=False)
        if corrected_img is not None: corrected_img.clear(verbose = False)
        return {"success": False, "image": target_img, "error": f"platesolve: {e}", "function": "platesolve"}
    
def calculate_invalidmask_worker(args):
    target_img, clear, invalidmask_kwargs = args
    target_invalidmask = None
    try:
        target_invalidmask = target_img.calculate_invalidmask(**invalidmask_kwargs)
        
        if clear:
            target_img.clear(verbose=False)
            if target_invalidmask is not None: target_invalidmask.clear(verbose=False)
        return {"success": True, "image": target_img, "invalidmask": target_invalidmask, "error": None, "function": "calculate_invalidmask"}

    except Exception as e:
        target_img.clear(verbose=False)
        if target_invalidmask is not None: target_invalidmask.clear(verbose=False)
        return {"success": False, "image": target_img, "invalidmask": None, "error": f"calculate_invalidmask: {e}", "function": "calculate_invalidmask"}

def calculate_srcmask_worker(args):
    target_img, target_srcmask, ra, dec, radius_arcsec, clear, srcmask_kwargs = args
    # Mask circular region if coordinate is given
    if ra is not None and dec is not None and radius_arcsec is not None:
        try:
            target_srcmask = target_img.calculate_circularmask(target_srcmask = target_srcmask, 
                                                               x_position = ra, y_position = dec, radius_arcsec = radius_arcsec, unit = 'coord', save = False, verbose = False, visualize = False, save_fig = False)
        
        except Exception as e:
            target_srcmask = None
            pass

    try:
        target_srcmask = target_img.calculate_sourcemask(target_srcmask = target_srcmask, 
                                                         **srcmask_kwargs)
        if clear:
            target_img.clear(verbose=False)
            if target_srcmask is not None: target_srcmask.clear(verbose=False)
        return {"success": True, "image": target_img, "sourcemask": target_srcmask, "error": None, "function": "calculate_srcmask"}
    
    except Exception as e:
        target_img.clear(verbose=False)
        if target_srcmask is not None: target_srcmask.clear(verbose=False)
        return {"success": False, "image": target_img, "sourcemask": None, "error": f"calculate_srcmask: {e}", "function": "calculate_srcmask"}
    
def calculate_bkgmap_worker(args):
    target_img, target_srcmask, target_invalidmask, clear, bkgmap_kwargs = args
    target_bkg = None
    try:
        target_bkg = target_img.calculate_bkg(target_srcmask = target_srcmask, 
                                              target_ivpmask = target_invalidmask, 
                                              **bkgmap_kwargs)
        if clear:
            target_img.clear(verbose=False)
            if target_bkg is not None: target_bkg.clear(verbose=False)
            if target_srcmask is not None: target_srcmask.clear(verbose=False)
            if target_invalidmask is not None: target_invalidmask.clear(verbose=False)
        return {"success": True, "image": target_img, "bkgmap": target_bkg, "error": None, "target_srcmask": target_srcmask, "target_invalidmask": target_invalidmask, "function": "calculate_bkgmap"}
    
    except Exception as e:
        target_img.clear(verbose=False)
        if target_bkg is not None: target_bkg.clear(verbose=False)
        if target_srcmask is not None: target_srcmask.clear(verbose=False)
        if target_invalidmask is not None: target_invalidmask.clear(verbose=False)
        return {"success": False, "image": target_img, "bkgmap": None, "error": f"calculate_bkgmap: {e}", "target_srcmask": target_srcmask, "target_invalidmask": target_invalidmask, "function": "calculate_bkgmap"}

def calculate_bkgrms_worker(args):
    target_img, target_srcmask, target_invalidmask, clear, bkgrms_kwargs = args
    target_bkgrms = None
    try:
        target_bkgrms = target_img.calculate_bkgrms(target_srcmask = target_srcmask, 
                                                    target_ivpmask = target_invalidmask,
                                                    **bkgrms_kwargs)
        if clear:
            target_img.clear(verbose=False)
            if target_bkgrms is not None: target_bkgrms.clear(verbose=False)
            if target_srcmask is not None: target_srcmask.clear(verbose=False)
            if target_invalidmask is not None: target_invalidmask.clear(verbose=False)
        return {"success": True, "image": target_img, "bkgrms": target_bkgrms, "error": None, "target_srcmask": target_srcmask, "target_invalidmask": target_invalidmask, "function": "calculate_bkgrms"}
    
    except Exception as e:
        target_img.clear(verbose=False)
        if target_bkgrms is not None: target_bkgrms.clear(verbose=False)
        if target_srcmask is not None: target_srcmask.clear(verbose=False)
        if target_invalidmask is not None: target_invalidmask.clear(verbose=False)
        return {"success": False, "image": target_img, "bkgrms": None, "error": f"calculate_bkgrms: {e}", "target_srcmask": target_srcmask, "target_invalidmask": target_invalidmask, "function": "calculate_bkgrms"}

def calculate_bkgrms_propagation_worker(args):
    target_img, target_bkg, clear, bkgrms_propagation_kwargs = args
    mbias = None
    mdark = None
    mflat = None
    mflaterr = None
    target_bkgrms = None
    try:
        mbias = target_img.get_masterframe(imagetyp = 'BIAS', max_days = 100)[0]
        mdark = target_img.get_masterframe(imagetyp = 'DARK', max_days = 100)[0]
        mflat = target_img.get_masterframe(imagetyp = 'FLAT', max_days = 100)[0]
        target_bkgrms = target_img.calculate_bkgrms_from_propagation(target_bkg = target_bkg, 
                                                                     mbias = mbias, 
                                                                     mdark = mdark, 
                                                                     mflat = mflat,
                                                                     mflaterr = mflaterr,
                                                                     **bkgrms_propagation_kwargs)
        
        if mbias is not None: mbias.clear(verbose=False)
        if mdark is not None: mdark.clear(verbose=False)
        if mflat is not None: mflat.clear(verbose=False)
        if mflaterr is not None: mflaterr.clear(verbose=False)
        if clear:
            target_img.clear(verbose=False)
            if target_bkg is not None: target_bkg.clear(verbose=False)
            if target_bkgrms is not None: target_bkgrms.clear(verbose=False)

        return {"success": True, "image": target_img, "bkgrms": target_bkgrms, "target_bkg": target_bkg, "error": None, "function": "calculate_bkgrms_propagation"}
    
    except Exception as e:
        target_img.clear(verbose=False)
        if target_bkg is not None: target_bkg.clear(verbose=False)
        if target_bkgrms is not None: target_bkgrms.clear(verbose=False)
        if mbias is not None: mbias.clear(verbose=False)
        if mdark is not None: mdark.clear(verbose=False)
        if mflat is not None: mflat.clear(verbose=False)
        if mflaterr is not None: mflaterr.clear(verbose=False)
        return {"success": False, "image": target_img, "bkgrms": None, "target_bkg": target_bkg, "error": f"calculate_bkgrms_propagation: {e}", "function": "calculate_bkgrms_propagation"}
    
def photometry_worker(args):
    target_img, target_bkg, target_bkgrms, target_mask, sex_params, clear, photometry_kwargs = args
    target_catalog = None
    try:
        target_catalog = target_img.photometry_sex(
            target_bkg = target_bkg,
            target_bkgrms = target_bkgrms,
            target_mask = target_mask, 
            sex_params = sex_params, 
            **photometry_kwargs)
        
        if clear:
            target_img.clear(verbose=False)
            if target_bkg is not None: target_bkg.clear(verbose=False)
            if target_bkgrms is not None: target_bkgrms.clear(verbose=False)
            if target_mask is not None: target_mask.clear(verbose=False)
            if target_catalog is not None: target_catalog.clear(verbose=False)
        return {"success": True, "image": target_img, "catalog": target_catalog, "error": None, "target_bkg": target_bkg, "target_bkgrms": target_bkgrms, "target_mask": target_mask, "function": "photometry"}
    except Exception as e:
        target_img.clear(verbose=False)
        if target_bkg is not None: target_bkg.clear(verbose=False)
        if target_bkgrms is not None: target_bkgrms.clear(verbose=False)
        if target_mask is not None: target_mask.clear(verbose=False)
        if target_catalog is not None: target_catalog.clear(verbose=False)
        return {"success": False, "image": target_img, "catalog": None, "error": f"photometry: {e}", "target_bkg": target_bkg, "target_bkgrms": target_bkgrms, "target_mask": target_mask, "function": "photometry"}
            
def photometric_calibration_worker(args):
    target_img, target_catalog, clear, photometric_calibration_kwargs = args
    photcal_kwargs = photometric_calibration_kwargs.copy()
        
    if target_img.filter.startswith('m'):
        photcal_kwargs['mag_lower'] = photcal_kwargs['mag_lower_MB']
        photcal_kwargs['mag_upper'] = photcal_kwargs['mag_upper_MB']
    else:
        photcal_kwargs['mag_lower'] = photcal_kwargs['mag_lower_BB']
        photcal_kwargs['mag_upper'] = photcal_kwargs['mag_upper_BB']
    # Remove mag_lower_BB, mag_upper_BB, mag_lower_MB, mag_upper_MB
    photcal_kwargs.pop('mag_lower_BB', None)
    photcal_kwargs.pop('mag_upper_BB', None)
    photcal_kwargs.pop('mag_lower_MB', None)
    photcal_kwargs.pop('mag_upper_MB', None)
    
    target_refcatalog = None
    try:
        target_img, target_catalog, target_refcatalog, _ = target_img.photometric_calibration(
            target_catalog = target_catalog,
            **photcal_kwargs)
        
        if clear:
            _ = None
            target_img.clear(verbose=False)
            if target_catalog is not None: target_catalog.clear(verbose=False)
            if target_refcatalog is not None: target_refcatalog.clear(verbose=False)
        return {"success": True, "image": target_img, "refcatalog": target_refcatalog, "error": None, "target_catalog": target_catalog, "function": "photometric_calibration"}
    except Exception as e:
        _ = None
        target_img.clear(verbose=False)
        if target_catalog is not None: target_catalog.clear(verbose=False)
        if target_refcatalog is not None: target_refcatalog.clear(verbose=False)
        return {"success": False, "image": target_img, "refcatalog": None, "error": f"photometric_calibration: {e}", "target_catalog": target_catalog, "function": "photometric_calibration"}

def forced_photometry_worker(args):
    target_img, target_bkg, target_bkgrms, x_arr, y_arr, unit, clear, forced_photometry_kwargs = args
    target_forced_catalog = None
    try:
        target_forced_catalog = target_img.photometry_forced_circular(
            x_arr = x_arr,
            y_arr = y_arr,
            unit = unit,
            target_bkg = target_bkg,
            target_bkgrms = target_bkgrms,
            **forced_photometry_kwargs)
        target_forced_catalog.apply_zp(
            target_img = target_img,
            save = forced_photometry_kwargs['save'],
            verbose = False)
        
        if clear:
            target_img.clear(verbose=False)
            if target_bkg is not None: target_bkg.clear(verbose=False)
            if target_bkgrms is not None: target_bkgrms.clear(verbose=False)
            if target_forced_catalog is not None: target_forced_catalog.clear(verbose=False)
        return {"success": True, "image": target_img, "catalog": target_forced_catalog, "error": None, "function": "forced_photometry_worker"}
    except Exception as e:
        target_img.clear(verbose=False)
        if target_bkg is not None: target_bkg.clear(verbose=False)
        if target_bkgrms is not None: target_bkgrms.clear(verbose=False)
        if target_forced_catalog is not None: target_forced_catalog.clear(verbose=False)
        return {"success": False, "image": target_img, "catalog": None, "error": f"forced_photometry: {e}", "function": "forced_photometry_worker"}
    
def tractor_photometry_worker(args):
    target_imgpathlist, target_filterlist, target_refcatalogpathlist, id_, target_ra, target_dec, hostinfo_dict, clear, tractor_photometry_kwargs = args
    runner = Tract7DTRunner(
        image_paths = target_imgpathlist,
        filter_list = target_filterlist,
        catalog_paths = target_refcatalogpathlist,
        id = id_,
        base_dir = tractor_photometry_kwargs['base_dir']
    )
    try:
        refcatlist = []
        catlist = []
        for refcat_path in target_refcatalogpathlist:
            cat_path = refcat_path.with_suffix('.cat')
            catlist.append(Catalog(path = cat_path))
            refcatlist.append(Catalog(path = refcat_path))
        refcatset = CatalogSet(refcatlist)
        merged_tbl, merged_metadata = refcatset.merge_catalogs(join_type = 'outer')
        
        list_ra = list(merged_tbl['ra_basis'])
        list_dec = list(merged_tbl['dec_basis'])
        list_type = ['STAR'] * len(list_ra)
        list_ellip = [np.nan] * len(list_ra)
        list_Re = [np.nan] * len(list_ra)
        list_theta = [np.nan] * len(list_ra)
        
        # Update transient information
        trans_idx = len(list_ra)
        list_ra.append(target_ra)
        list_dec.append(target_dec)
        list_type.append('STAR')
        list_ellip.append(np.nan)
        list_Re.append(np.nan)
        list_theta.append(np.nan)
        
        # Update host galaxy information
        host_ra = hostinfo_dict.get('hostgalaxy_radeg')
        host_dec = hostinfo_dict.get('hostgalaxy_dedeg')
        host_ellip = hostinfo_dict.get('hostgalaxy_ellipticity')
        host_PA = hostinfo_dict.get('hostgalaxy_pa')
        host_Re = hostinfo_dict.get('hostgalaxy_Re')
        if host_ra is not None and host_dec is not None:
            host_idx = len(list_ra)
            list_ra.append(host_ra)
            list_dec.append(host_dec)
            list_type.append('EXP')
            if host_ellip is not None:
                list_ellip.append(host_ellip)
            else:
                list_ellip.append(np.nan)
            if host_PA is not None:
                list_theta.append(host_PA)
            else:
                list_theta.append(np.nan)
            if host_Re is not None:
                list_Re.append(host_Re)
            else:
                list_Re.append(np.nan)
                
        # Update sources nearby host galaxy
        if host_ra is not None and host_dec is not None:
            coord_tbl = SkyCoord(list_ra, list_dec, unit = 'deg')
            for cat in catlist:
                cat.select_sources(x = host_ra, y = host_dec, unit = 'coord', matching_radius = 300)
            catset = CatalogSet(catlist)
            merged_tbl_nearby, merged_metadata_nearby = catset.merge_catalogs(join_type = 'outer')
            coord_nearby = SkyCoord(merged_tbl_nearby['ra_basis'], merged_tbl_nearby['dec_basis'], unit = 'deg')
            closest_ids, closest_dists, _ = match_coordinates_sky(coord_nearby, coord_tbl)
            max_distance_deg = 10 / 3600
            unmatched_mask = closest_dists.value > max_distance_deg
            
            list_ra.extend(list(merged_tbl_nearby['ra_basis'][unmatched_mask]))
            list_dec.extend(list(merged_tbl_nearby['dec_basis'][unmatched_mask]))
            list_type.extend(['STAR'] * np.sum(unmatched_mask))
            list_ellip.extend([np.nan] * np.sum(unmatched_mask))
            list_Re.extend([np.nan] * np.sum(unmatched_mask))
            list_theta.extend([np.nan] * np.sum(unmatched_mask))
            
        indices = list(range(len(list_ra)))

        special_indices = []
        if 'trans_idx' in locals():
            special_indices.append(trans_idx)
        if 'host_idx' in locals():
            special_indices.append(host_idx)

        # 일반 source index
        normal_indices = [i for i in indices if i not in special_indices]

        # 최종 순서: normal → transient → host
        final_indices = normal_indices + special_indices

        # reorder
        list_ra     = [list_ra[i] for i in final_indices]
        list_dec    = [list_dec[i] for i in final_indices]
        list_type   = [list_type[i] for i in final_indices]
        list_ellip  = [list_ellip[i] for i in final_indices]
        list_Re     = [list_Re[i] for i in final_indices]
        list_theta  = [list_theta[i] for i in final_indices]
            
        runner.register_target(
            list_ra = list_ra,
            list_dec = list_dec,
            list_type = list_type,
            list_ellip = list_ellip,
            list_Re = list_Re,
            list_theta = list_theta,
            **tractor_photometry_kwargs['register_config']
        )
        
        # Update galaxy flux inforamtion
        host_gmag = hostinfo_dict.get('hostgalaxy_gmag')
        host_rmag = hostinfo_dict.get('hostgalaxy_rmag')
        host_imag = hostinfo_dict.get('hostgalaxy_imag')
        if host_gmag is None:
            host_gmag = np.nan
        if host_rmag is None:
            host_rmag = np.nan
        if host_imag is None:
            host_imag = np.nan
        host_mag = np.nanmean([host_gmag, host_rmag, host_imag])
        
        if np.isfinite(host_mag):
            input_catalog_tbl = ascii.read(runner.input_catalogs_path)
            for filter_key in target_filterlist:
                zp = runner.configuration.image_scaling.zp_ref
                filter_key_in_tract7dt = f'FLUX_{filter_key}'
                host_mag_inst = host_mag - zp
                host_flux_inst = 10 ** (-0.4 * (host_mag_inst))
                input_catalog_tbl[-1][filter_key_in_tract7dt] = host_flux_inst
            input_catalog_tbl.write(runner.input_catalogs_path, overwrite=True)
        
        runner.register_reference_catalog(
            list_ra = [target_ra],
            list_dec = [target_dec],
        )
        
        runner.run(tractor_photometry_kwargs['save_result_catalog'])
        result = runner.workdir / 'final_catalog_with_fit.csv'
        return {"success": True, "result": result, "error": None, "function": "tractor_photometry"}
    except Exception as e:
        return {"success": False, "result": None, "error": f"tractor_photometry: {e}", "function": "tractor_photometry"}
    
def pipeline_worker_before_stacking(args):
    target_img, ra, dec, clear, config = args

    target_srcmask = None
    target_bkg = None
    target_bkgrms = None
    target_catalog = None
    target_refcatalog = None
    target_invalidmask = None
    target_mask = None
    try:
        if config.do_preprocess:
            preprocess_args = [target_img, False]
            result_preprocess = preprocess_worker(preprocess_args)
            target_img = result_preprocess["image"]
            if not result_preprocess["success"]:
                raise RuntimeError(result_preprocess["error"])
        
        if config.do_platesolve:
            platesolve_args = [target_img, False]
            result_platesolve = platesolve_worker(platesolve_args)
            target_img = result_platesolve["image"]
            if not result_platesolve["success"]:
                raise RuntimeError(result_platesolve["error"])
        
        if config.do_calculate_sourcemask:
            srcmask_args = [target_img, None, ra, dec, config.circularmask['radius_arcsec'], False, config.sourcemask]
            result_sourcemask = calculate_srcmask_worker(srcmask_args)
            target_img = result_sourcemask["image"]
            target_srcmask = result_sourcemask["sourcemask"]
            if not result_sourcemask["success"]:
                raise RuntimeError(result_sourcemask["error"])
        
        if config.do_calculate_bkgmap:
            bkgmap_args = [target_img, target_srcmask, None, False, config.bkgmap]
            result_bkgmap = calculate_bkgmap_worker(bkgmap_args)
            target_img = result_bkgmap["image"]
            target_bkg = result_bkgmap["bkgmap"]
            target_srcmask = result_bkgmap["target_srcmask"]
            target_invalidmask = result_bkgmap["target_invalidmask"]
            if not result_bkgmap["success"]:
                raise RuntimeError(result_bkgmap["error"])
            
        if config.do_calculate_bkgrms:
            if config.do_calculate_bkgrms_from_propagation:
                bkgrms_kwargs = config.bkgrms.copy()
                bkgrms_kwargs.pop('filter_size', None)
                bkgrms_kwargs.pop('box_size', None)
                bkgrms_args = [target_img, target_bkg, False, bkgrms_kwargs]
                result_bkgrms = calculate_bkgrms_propagation_worker(bkgrms_args)
                target_bkg = result_bkgrms["target_bkg"]    
            else:
                bkgrms_args = [target_img, target_srcmask, target_invalidmask, False, config.bkgrms]
                result_bkgrms = calculate_bkgrms_worker(bkgrms_args)
                target_srcmask = result_bkgrms["target_srcmask"]
                target_invalidmask = result_bkgrms["target_invalidmask"]
            target_img = result_bkgrms["image"]
            target_bkgrms = result_bkgrms["bkgrms"]
            if not result_bkgrms["success"]:
                raise RuntimeError(result_bkgrms["error"])
        
        if config.do_photometry:
            photometry_args = [target_img, target_bkg, target_bkgrms, None, None, True, config.photometry]
            result_photometry = photometry_worker(photometry_args)
            target_img = result_photometry["image"]
            target_catalog = result_photometry["catalog"]
            target_bkg = result_photometry["target_bkg"]
            target_bkgrms = result_photometry["target_bkgrms"]
            target_mask = result_photometry["target_mask"]
            if not result_photometry["success"]:
                raise RuntimeError(result_photometry["error"])
        
        if config.do_photometric_calibration:
            photometric_calibration_args = [target_img, target_catalog, False, config.photcal]
            result_photometric_calibration = photometric_calibration_worker(photometric_calibration_args)
            target_img = result_photometric_calibration["image"]
            target_refcatalog = result_photometric_calibration["refcatalog"]
            target_catalog = result_photometric_calibration["target_catalog"]
            if not result_photometric_calibration["success"]:
                raise RuntimeError(result_photometric_calibration["error"])
            
        if clear:
            target_img.clear(verbose=False)
            if target_invalidmask is not None: target_invalidmask.clear(verbose=False)
            if target_srcmask is not None: target_srcmask.clear(verbose=False)
            if target_bkg is not None: target_bkg.clear(verbose=False)
            if target_bkgrms is not None: target_bkgrms.clear(verbose=False)
            if target_catalog is not None: target_catalog.clear(verbose=False)
            if target_refcatalog is not None: target_refcatalog.clear(verbose=False)
            if target_mask is not None: target_mask.clear(verbose=False)
        return {"success": True, "image": target_img, "sourcemask": target_srcmask, "bkgmap": target_bkg, "bkgrms": target_bkgrms, "catalog": target_catalog, "refcatalog": target_refcatalog, "error": None, "function": "pipeline_worker_before_stacking"}

    except Exception as e:
        print(e)
        target_img.clear(verbose=False)
        if target_invalidmask is not None: target_invalidmask.clear(verbose=False)
        if target_srcmask is not None: target_srcmask.clear(verbose=False)
        if target_bkg is not None: target_bkg.clear(verbose=False)
        if target_bkgrms is not None: target_bkgrms.clear(verbose=False)
        if target_catalog is not None: target_catalog.clear(verbose=False)
        if target_refcatalog is not None: target_refcatalog.clear(verbose=False)
        if target_mask is not None: target_mask.clear(verbose=False)
        return {"success": False, "image": target_img, "sourcemask": target_srcmask, "bkgmap": target_bkg, "bkgrms": target_bkgrms, "catalog": target_catalog, "refcatalog": target_refcatalog, "error": f"pipeline_worker_before_stacking: {e}", "function": "pipeline_worker_before_stacking"}

def stacking_worker(args):
    target_imgset, clear, stack_prepare_config, stack_config = args
    stacked_img = None
    stacked_bkgrms = None
    try:
        target_imgset.prepare_stack(**stack_prepare_config)
        stacked_img, stacked_bkgrms = target_imgset.stack(**stack_config)
        if clear:
            for target_img in target_imgset.target_images:
                target_img.clear(verbose=False)
            for target_bkgrms in target_imgset.bkgrms:
                target_bkgrms.clear(verbose=False)
            if stacked_img is not None: stacked_img.clear(verbose=False)
            if stacked_bkgrms is not None: stacked_bkgrms.clear(verbose=False)
        return {"success": True, "image": stacked_img, "bkgrms": stacked_bkgrms, "error": None, "function": "stacking"}
    except Exception as e:
        return {"success": False, "image": None, "error": f"stacking: {e}", "function": "pipeline_stacking_worker"}
    
def DIA_worker(args):
    target_img, reference_img, clear, DIA_kwargs = args
    subtracted_img = None
    reference_img = None
    subtracted_catalog = None
    candidate_catalog = None
    transient_catalog = None
    try:
        (all_catalogs, 
        candidate_catalogs, 
        transient_catalogs, 
        subframe_target_imglist, 
        subframe_reference_imglist, 
        subframe_subtract_imglist) = target_img.DIA(
            reference_img = reference_img, 
            target_bkg = None, 
            **DIA_kwargs)
        subtracted_catalog = all_catalogs[0]
        candidate_catalog = candidate_catalogs[0]
        transient_catalog = transient_catalogs[0]
        target_img = subframe_target_imglist[0]
        reference_img = subframe_reference_imglist[0]
        subtracted_img = subframe_subtract_imglist[0]
        
        if clear:
            target_img.clear(verbose=False)
            if reference_img is not None: reference_img.clear(verbose=False)
            if subtracted_img is not None: subtracted_img.clear(verbose=False)
            if subtracted_catalog is not None: subtracted_catalog.clear(verbose=False)
            if candidate_catalog is not None: candidate_catalog.clear(verbose=False)
            if transient_catalog is not None: transient_catalog.clear(verbose=False)
        return {"success": True, "image": target_img, "reference_img": reference_img, "subtracted_img": subtracted_img, "subtracted_catalog": subtracted_catalog, "candidate_catalog": candidate_catalog, "transient_catalog": transient_catalog, "error": None, "function": "DIA"}
    except Exception as e:
        target_img.clear(verbose=False)
        if reference_img is not None: reference_img.clear(verbose=False)
        if subtracted_img is not None: subtracted_img.clear(verbose=False)
        if subtracted_catalog is not None: subtracted_catalog.clear(verbose=False)
        if candidate_catalog is not None: candidate_catalog.clear(verbose=False)
        if transient_catalog is not None: transient_catalog.clear(verbose=False)
        return {"success": False, "image": target_img, "reference_img": reference_img, "subtracted_img": subtracted_img, "subtracted_catalog": subtracted_catalog, "candidate_catalog": candidate_catalog, "transient_catalog": transient_catalog, "error": f"DIA: {e}", "function": "DIA"}

def pipeline_worker_after_stacking(args):
    stacked_img, stacked_bkgrms, reference_img, ra, dec, clear, config = args

    stacked_ivpmask = None
    stacked_srcmask = None
    stacked_bkg = None
    stacked_catalog = None
    stacked_refcatalog = None
    stacked_forcatalog = None
    subtracted_img = None
    subtracted_catalog = None
    candidate_catalog = None
    transient_catalog = None
    subtracted_forced_catalog = None
    try:
        if config.do_calculate_invalidmask:
            invalidmask_args = [stacked_img, False, config.invalidmask]
            result_invalidmask = calculate_invalidmask_worker(invalidmask_args)
            stacked_img = result_invalidmask["image"]
            stacked_ivpmask = result_invalidmask["invalidmask"]
            if not result_invalidmask["success"]:
                raise RuntimeError(result_invalidmask["error"])

        if config.do_calculate_sourcemask:
            sourcemask_args = [stacked_img, None, ra, dec, config.circularmask['radius_arcsec'], False, config.sourcemask]
            result_sourcemask = calculate_srcmask_worker(sourcemask_args)
            stacked_img = result_sourcemask["image"]
            stacked_srcmask = result_sourcemask["sourcemask"]
            if not result_sourcemask["success"]:
                raise RuntimeError(result_sourcemask["error"])

        if config.do_calculate_bkgmap:
            bkgmap_args = [stacked_img, stacked_srcmask, stacked_ivpmask, False, config.bkgmap]
            result_bkgmap = calculate_bkgmap_worker(bkgmap_args)
            stacked_img = result_bkgmap["image"]
            stacked_bkg = result_bkgmap["bkgmap"]
            if not result_bkgmap["success"]:
                raise RuntimeError(result_bkgmap["error"])

        if config.do_photometry:
            # Background is set to MANUAL to avoid double background subtraction from the image.
            sex_params = dict(BACK_TYPE = 'MANUAL')
            photometry_args = [stacked_img, stacked_bkg, stacked_bkgrms, None, sex_params, False, config.photometry]
            result_photometry = photometry_worker(photometry_args)
            stacked_img = result_photometry["image"]
            stacked_catalog = result_photometry["catalog"]
            stacked_bkgrms = result_photometry["target_bkgrms"]
            if not result_photometry["success"]:
                raise RuntimeError(result_photometry["error"])
        
        if config.do_photometric_calibration:
            photometric_calibration_args = [stacked_img, stacked_catalog, False, config.photcal]
            result_photometric_calibration = photometric_calibration_worker(photometric_calibration_args)
            stacked_img = result_photometric_calibration["image"]
            stacked_refcatalog = result_photometric_calibration["refcatalog"]
            stacked_catalog = result_photometric_calibration["target_catalog"]
            if not result_photometric_calibration["success"]:
                raise RuntimeError(result_photometric_calibration["error"])
            
        if config.do_forced_photometry:
            forced_photometry_args = [stacked_img, stacked_bkg, stacked_bkgrms, ra, dec, 'coord', False, config.forced_photometry]
            result_forced_photometry = forced_photometry_worker(forced_photometry_args)
            stacked_img = result_forced_photometry["image"]
            stacked_forcatalog = result_forced_photometry["catalog"]
            if not result_forced_photometry["success"]:
                raise RuntimeError(result_forced_photometry["error"])
            
        if config.do_DIA:
            if reference_img is None:
                raise ValueError("Reference image is not found.")
            DIA_args = [stacked_img, reference_img, clear, config.DIA]
            result_DIA = DIA_worker(DIA_args)
            stacked_img = result_DIA["image"]
            reference_img = result_DIA["reference_img"]
            subtracted_img = result_DIA["subtracted_img"]
            subtracted_catalog = result_DIA["subtracted_catalog"]
            candidate_catalog = result_DIA["candidate_catalog"]
            transient_catalog = result_DIA["transient_catalog"]
            if not result_DIA["success"]:
                raise RuntimeError(result_DIA["error"])
            
            if config.do_forced_photometry:
                forced_photometry_args = [subtracted_img, None, None, ra, dec, 'coord', False, config.forced_photometry]
                result_forced_photometry = forced_photometry_worker(forced_photometry_args)
                subtracted_img = result_forced_photometry["image"]
                subtracted_forced_catalog = result_forced_photometry["catalog"]
                if not result_forced_photometry["success"]:
                    raise RuntimeError(result_forced_photometry["error"])
            
        if clear:
            stacked_img.clear(verbose=False)
            if stacked_bkgrms is not None: stacked_bkgrms.clear(verbose=False)
            if stacked_catalog is not None: stacked_catalog.clear(verbose=False)
            if stacked_refcatalog is not None: stacked_refcatalog.clear(verbose=False)
            if stacked_forcatalog is not None: stacked_forcatalog.clear(verbose=False)
            if reference_img is not None: reference_img.clear(verbose=False)
            if subtracted_img is not None: subtracted_img.clear(verbose=False)
            if subtracted_catalog is not None: subtracted_catalog.clear(verbose=False)
            if candidate_catalog is not None: candidate_catalog.clear(verbose=False)
            if transient_catalog is not None: transient_catalog.clear(verbose=False)
            if subtracted_forced_catalog is not None: subtracted_forced_catalog.clear(verbose=False)
            
        return {"success": True, "image": stacked_img, "bkgrms": stacked_bkgrms, "catalog": stacked_catalog, "refcatalog": stacked_refcatalog, "forcatalog": stacked_forcatalog, "error": None, "function": "pipeline_after_stacking_worker"}
    except Exception as e:
        stacked_img.clear(verbose=False)
        if stacked_bkgrms is not None: stacked_bkgrms.clear(verbose=False)
        if stacked_catalog is not None: stacked_catalog.clear(verbose=False)
        if stacked_refcatalog is not None: stacked_refcatalog.clear(verbose=False)
        if stacked_forcatalog is not None: stacked_forcatalog.clear(verbose=False)
        if reference_img is not None: reference_img.clear(verbose=False)
        if subtracted_img is not None: subtracted_img.clear(verbose=False)
        if subtracted_catalog is not None: subtracted_catalog.clear(verbose=False)
        if candidate_catalog is not None: candidate_catalog.clear(verbose=False)
        if transient_catalog is not None: transient_catalog.clear(verbose=False)
        if subtracted_forced_catalog is not None: subtracted_forced_catalog.clear(verbose=False)
        return {"success": False, "image": stacked_img, "bkgrms": stacked_bkgrms, "catalog": stacked_catalog, "refcatalog": stacked_refcatalog, "forcatalog": stacked_forcatalog, "error": f"pipeline_after_stacking: {e}", "function": "pipeline_after_stacking_worker"}

#%%

class AlertProcessor:
    def __init__(self):
        self.config = Configuration(config_filenames=['alertprocessor.config'])
        self.gwportal_connector = GWPortalConnector()
        self.ezphot_connector = DataBrowser('scidata')
        self.all_images = []
        self.target_images = []
        self.failed_history = dict()
        
    def __repr__(self):
        return f"AlertProcessor(Selected/All Images = {len(self.target_images)}/{len(self.all_images)}, failed_history = {len(self.failed_history)})"
        
    def load_images_ezphot(self, 
                           alert_instance: Alert, 
                           file_pattern: str = '7DT*.fits', 
                           obs_start_time: Optional[Union[str, Time]] = None, 
                           obs_end_time: Optional[Union[str, Time]] = None):
        
        if obs_start_time is None and obs_end_time is None:
            print('WARNING: No observation date is provided. Using trigger time as trigger date.')
            obs_start_time = Time(alert_instance.trigger_time)
            
        kwargs = dict()
        if obs_start_time is not None:
            kwargs['obs_start'] = flexible_time_parser(obs_start_time)
        if obs_end_time is not None:
            kwargs['obs_end'] = flexible_time_parser(obs_end_time)
            
        self.ezphot_connector.objname = np.atleast_1d(alert_instance.objname)[0]
        target_imgset_objname = self.ezphot_connector.search(pattern = file_pattern, return_type = 'science')
        all_target_images = target_imgset_objname.target_images

        # Try query with tile_id second
        if alert_instance.tile_id is not None:
            self.ezphot_connector.objname = np.atleast_1d(alert_instance.tile_id)[0]
            target_imgset_tile = self.ezphot_connector.search(pattern = file_pattern, return_type = 'science')
            all_target_images += target_imgset_tile.target_images
        
        if len(all_target_images) == 0:
            print('WARNING: No images are found.')
            return False
        target_imgset = ImageSet(all_target_images)
        target_imgset.select_images(**kwargs)

        self.target_images = target_imgset.target_images
        self.all_images = self.target_images
        
    def load_images_db(self, alert_instance: Alert, 
                       query_type: str = 'raw', 
                       obs_start_time: Optional[Union[str, Time]] = None, 
                       obs_end_time: Optional[Union[str, Time]] = None):
        
        if obs_start_time is None and obs_end_time is None:
            print('WARNING: No observation date is provided. Using trigger time as trigger date.')
            obs_start_time = Time(alert_instance.trigger_time)
        
        kwargs = dict()
        if obs_start_time is not None:
            kwargs['obs_start_date'] = flexible_time_parser(obs_start_time).datetime.strftime("%Y-%m-%d")
        if obs_end_time is not None:
            kwargs['obs_end_date'] = flexible_time_parser(obs_end_time).datetime.strftime("%Y-%m-%d")
        kwargs['object_name'] = alert_instance.objname
            
        # Try query with objname first
        self.gwportal_connector.query_type = query_type
        tbl_observation_objname = self.gwportal_connector.query(**kwargs)
        tbl_observation = tbl_observation_objname
        if ('tile' in tbl_observation_objname.colnames) & ('target' in tbl_observation_objname.colnames):
            tbl_observation_objname.remove_columns(['tile', 'target'])
        # Try query with tile_id second
        if alert_instance.tile_id is not None:
            kwargs.pop('object_name', None)
            kwargs['tile_name'] = alert_instance.tile_id
            tbl_observation_tile = self.gwportal_connector.query(**kwargs)
            tbl_observation = vstack([tbl_observation, tbl_observation_tile])
            
        if len(tbl_observation) == 0:
            print('WARNING: No images are found.')
            return False
        list_filepath = tbl_observation['filepath']
        self.target_images = self._load_images(list_filepath)
        self.all_images = self.target_images
        
    def select_images(self,
                      file_key=None,
                      filter=None,
                      exptime=None,
                      objname=None,
                      obs_start=None,
                      obs_end=None,
                      seeing=None,
                      depth=None,
                      observatory=None,
                      telname=None,
                      ):
        from ezphot.imageobjects import ImageSet
        if self.all_images == []:
            print('WARNING: No images are found.')
            return False
        imgset = ImageSet(self.all_images)
        imgset.select_images(
            file_key=file_key,
            filter=filter,
            exptime=exptime,
            objname=objname,
            obs_start=obs_start,
            obs_end=obs_end,
            seeing=seeing,
            depth=depth,
            observatory=observatory,
            telname=telname,
        )
        self.target_images = imgset.target_images
        
    def load_images_from_path(self, list_filepath: list[str]):
        self.target_images = self._load_images(list_filepath)
        self.all_images = self.target_images
            
    def _load_images(self, list_filepath: list[str]):
        def load_image(filepath: str):
            try:
                image = ScienceImage(
                    path = filepath
                )
                return image
            except Exception as e:
                print(f'WARNING: Failed to load image {filepath}: {e}')
                return 
        # IO bounded operation
        with ThreadPoolExecutor(max_workers = 64) as pool:
            target_images = list(tqdm(pool.map(load_image, list_filepath), desc = 'Loading images...', total = len(list_filepath)))
        return target_images
    
    def _check_input_images(self):
        if self.target_images == []:
            raise RuntimeError('WARNING: No images are found. Loading images with load_images_ezphot() or load_images_db() or load_images_from_path() first')
        return True

    def run_parallel(self, tasks, worker, step_name, desc, batch_size=None):
        """
        Same interface as before, but now supports batching.
        If batch_size is None, behaves exactly like original run_parallel.
        """

        def optimal_workers(n_tasks: int, max_workers: int) -> int:
            import math
            if n_tasks <= 0:
                return 0
            max_workers = max(1, max_workers)

            cycles = math.ceil(n_tasks / max_workers)
            workers = math.ceil(n_tasks / cycles)

            return min(max_workers, n_tasks, workers)
        
        if batch_size is None:
            batch_size = len(tasks)

        all_results = []
        new_target_images = []
        n = len(tasks)
        
        # Iterate in batches
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_tasks = tasks[start:end]
            n_workers = optimal_workers(len(batch_tasks), self.config.max_workers)

            batch_results = []
            futures = []

            try:
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futures = [pool.submit(worker, task) for task in batch_tasks]

                    for f in tqdm(as_completed(futures),
                                total=len(futures),
                                desc=f"{desc} [{start+1}-{end}/{n}]"):

                        try:
                            batch_results.append(f.result())
                        except Exception as e:
                            batch_results.append({"success": False, "image": None, "error": e, "function": step_name})

            except KeyboardInterrupt:
                for fut in futures:
                    fut.cancel()
                try:
                    pool.shutdown(cancel_futures=True)
                except:
                    pass

            # collect results from this batch
            all_results.extend(batch_results)

            # update target_images only from successful results
            for res in batch_results:
                if res.get("success", True) and res.get("image") is not None:
                    new_target_images.append(res["image"])

            # store lightweight failure history
            if step_name not in self.failed_history:
                self.failed_history[step_name] = []

            for res in batch_results:
                if not res.get("success", True):
                    self.failed_history[step_name].append({
                        "error": str(res.get("error")),
                        "function": res.get("function"),
                        "image_path": getattr(res.get("image"), "path", None)
                    })

            print(f"{len(batch_results)}/{len(batch_tasks)} processed in batch")

        # After all batches, update target_images
        self.target_images = new_target_images

        print(f"TOTAL: {len(all_results)}/{len(tasks)} processed")

        return all_results
    
    def run_sequential(self, tasks, worker, step_name, desc):
        """
        Sequential version of run_parallel.
        Behaves consistently with run_parallel:
        - collects results
        - updates self.target_images
        - populates failed_history
        - prints summary
        """
        all_results = []
        new_target_images = []

        # prepare failure history storage
        if step_name not in self.failed_history:
            self.failed_history[step_name] = []

        for task in tqdm(tasks, desc=f"{desc}"):
            try:
                result = worker(task)
            except Exception as e:
                # Soft-fail to match run_parallel structure
                result = {
                    "success": False,
                    "image": None,
                    "error": e,
                    "function": step_name
                }

            all_results.append(result)

            # --- success case ---
            if result.get("success", True):
                img = result.get("image")
                if img is not None:
                    new_target_images.append(img)

            # --- failure case ---
            if not result.get("success", True):
                self.failed_history[step_name].append({
                    "error": str(result.get("error")),
                    "function": step_name,
                    "image_path": getattr(result.get("image"), "path", None)
                })

        # update target_images exactly like run_parallel
        self.target_images = new_target_images

        print(f"TOTAL: {len(all_results)}/{len(tasks)} processed (sequential)")

        return all_results

    
    def preprocess(self, alert_instance: Alert):
        self.load_images_db(alert_instance, 'raw')
        
        tasks = [(img, True) for img in self.target_images]
        return self.run_parallel(tasks, preprocess_worker,
                                step_name='preprocess',
                                desc='Preprocessing...',
                                batch_size = self.config.batch_size)

    def platesolve(self, alert_instance: Alert):
        self._check_input_images()
        tasks = [(img, True) for img in self.target_images]
        return self.run_parallel(tasks, platesolve_worker,
                                step_name='platesolve',
                                desc='Platesolving...',
                                batch_size = self.config.batch_size)
        
    def calculate_sourcemask(self, alert_instance: Alert):
        self._check_input_images()

        circularmask_kwargs = self.config.circularmask
        srcmask_kwargs = self.config.sourcemask

        if alert_instance.is_coordinate_given:
            tasks = [(img, None, alert_instance.ra, alert_instance.dec,
                    circularmask_kwargs['radius_arcsec'], True, srcmask_kwargs)
                    for img in self.target_images]
        else:
            tasks = [(img, None, None, None, None, True, srcmask_kwargs)
                    for img in self.target_images]

        return self.run_parallel(tasks, calculate_srcmask_worker,
                                step_name='sourcemask',
                                desc='Calculating source mask...',
                                batch_size = self.config.batch_size)
        
    def calculate_bkgmap(self, alert_instance: Alert):
        self._check_input_images()

        bkgmap_kwargs = self.config.bkgmap
        tasks = [(img, img.sourcemask, img.invalidmask, True, bkgmap_kwargs)
                for img in self.target_images]

        return self.run_parallel(tasks, calculate_bkgmap_worker,
                                step_name='bkgmap',
                                desc='Calculating background map...',
                                batch_size = self.config.batch_size)

    def calculate_bkgrms(self, alert_instance: Alert):
        self._check_input_images()

        bkgrms_kwargs = self.config.bkgrms

        if self.config.calculate_bkgrms_from_propagation:
            bkgrms_kwargs.pop('filter_size', None)
            bkgrms_kwargs.pop('box_size', None)
            tasks = [(img, img.bkgmap, True, bkgrms_kwargs)
                    for img in self.target_images]
            worker = calculate_bkgrms_propagation_worker
        else:
            tasks = [(img, img.sourcemask, img.invalidmask, True, bkgrms_kwargs)
                    for img in self.target_images]
            worker = calculate_bkgrms_worker

        return self.run_parallel(tasks, worker,
                                step_name='bkgrms',
                                desc='Calculating background rms map...',
                                batch_size = self.config.batch_size)

    def photometry(self, alert_instance: Alert):
        self._check_input_images()

        tasks = [(img, img.bkgmap, img.bkgrms, None,
                None, True, self.config.photometry)
                for img in self.target_images]

        return self.run_parallel(tasks, photometry_worker,
                                step_name='photometry',
                                desc='Photometry...',
                                batch_size = self.config.batch_size)
                
    def photometric_calibration(self, alert_instance: Alert):
        self._check_input_images()

        tasks = [(img, img.catalog, True, self.config.photcal)
                for img in self.target_images]

        return self.run_parallel(tasks, photometric_calibration_worker,
                                step_name='photometric_calibration',
                                desc='Photometric calibration...',
                                batch_size = self.config.batch_size)
    
    def pipeline_before_stacking(self, alert_instance: Alert):
        self._check_input_images()
        
        tasks = [(img, alert_instance.ra, alert_instance.dec, True, self.config) for img in self.target_images]
        
        return self.run_parallel(tasks, pipeline_worker_before_stacking,
                                step_name='pipeline_before_stacking',
                                desc='Pipeline before stacking...',
                                batch_size = self.config.batch_size)
        
    def stacking(self, 
                 by_filter: bool = True,
                 by_exptime: bool = False,
                 by_objname: bool = False,
                 by_telname: bool = False,
                 by_observatory: bool = True,
                 by_obsdate: bool = True,
                 obsdate_delta: float = 0.5,
                 obsdate_key: str = 'obsdate'):
        self._check_input_images()
        target_imgsetlist = ImageSet(self.target_images).divide_images(
            by_filter = by_filter,
            by_exptime = by_exptime,
            by_objname = by_objname,
            by_telname = by_telname,
            by_observatory = by_observatory,
            by_obsdate = by_obsdate,
            obsdate_delta = obsdate_delta,
            obsdate_key = obsdate_key
        )
        original_max_workers = self.config.max_workers
        self.config.max_workers = int(np.ceil(self.config.max_workers/2))
        tasks = [(target_imgset, True, self.config.stack_prepare, self.config.stack) for target_imgset in target_imgsetlist]
        print('Total number of stacking tasks: ', len(tasks))
        return self.run_parallel(tasks, stacking_worker,
                                step_name='stacking',
                                desc='Stacking...',
                                batch_size = int(np.ceil(self.config.batch_size/3)))
    
    def pipeline_after_stacking(self, alert_instance: Alert):
        # sex photometry, photometric calibration, forced photometry, and DIA
        self._check_input_images()

        if self.config.do_DIA:
            reference_imgdict = {}
            for stacked_img in tqdm(self.target_images, desc = 'Querying reference images...'):
                reference_result = stacked_img.get_referenceframe(verbose = False)
                reference_img = None
                if reference_result is not None:
                    reference_img = reference_result[0]
                reference_imgdict[stacked_img] = reference_img
            tasks = [(stacked_img, stacked_img.bkgrms, reference_imgdict[stacked_img], alert_instance.ra, alert_instance.dec, True, self.config) for stacked_img in self.target_images]
        else:
            tasks = [(stacked_img, stacked_img.bkgrms, None, alert_instance.ra, alert_instance.dec, True, self.config) for stacked_img in self.target_images]
        
        return self.run_parallel(tasks, pipeline_worker_after_stacking,
                                 step_name='pipeline_after_stacking',
                                 desc='Pipeline after stacking...',
                                 batch_size = self.config.batch_size)
        
    def tractor_photometry(self, alert_instance: Alert):
        def is_valid(x):
            return x is not None and np.isfinite(x)
        self._check_input_images()

        target_imgsetlist = ImageSet(self.target_images).divide_images(
            by_filter = False,
            by_exptime = False,
            by_objname = True,
            by_telname = False,
            by_observatory = False,
            by_obsdate = True,
            obsdate_delta = 0.5,
            obsdate_key = 'obsdate'
        )
        
        target_ra = alert_instance.ra
        target_dec = alert_instance.dec
        hostinfo_dict = dict()
        for key in alert_instance.__dict__:
            if key.startswith('hostgalaxy_'):
                hostinfo_dict[key] = getattr(alert_instance, key)
                
        r1 = hostinfo_dict.get('hostgalaxy_r1')
        r2 = hostinfo_dict.get('hostgalaxy_r2')
        # check if r1 is None or nan
        if is_valid(r1) and is_valid(r2):
            a = max(r1, r2)
            b = min(r1, r2)
            hostinfo_dict['hostgalaxy_ellipticity'] = 1 - b / a
        else:
            hostinfo_dict['hostgalaxy_ellipticity'] = None
            
        pixelscale = self.target_images[0].pixelscale
        if pixelscale is not None:
            pixelscale = np.mean(pixelscale)
        else:
            pixelscale = 1
            
        if is_valid(r1) and is_valid(r2):
            hostinfo_dict['hostgalaxy_Re'] = np.sqrt(r1 * r2) / pixelscale
        else:
            hostinfo_dict['hostgalaxy_Re'] = None
        # Change PA to tract7dt convention 
        if hostinfo_dict['hostgalaxy_pa'] is not None:
            hostinfo_dict['hostgalaxy_pa'] = hostinfo_dict['hostgalaxy_pa'] - 90
        
        all_imgpathlist = []
        all_filterlist = []
        all_refcatalogpathlist = []
        all_idlist = []
        for target_imgset in target_imgsetlist:

            target_imgset.target_images.sort(key = lambda x: x.filter)
            target_imgpathlist = []
            target_filterlist = []
            target_refcatalogpathlist = []
            for target_img in target_imgset.target_images:
                try:
                    imagepatt = target_img.path
                    filter_ = target_img.filter
                    refcatalogpath = target_img.refcatalog.path
                except Exception as e:
                    continue
                target_imgpathlist.append(imagepatt)
                target_filterlist.append(filter_)
                target_refcatalogpathlist.append(refcatalogpath)
            all_imgpathlist.append(target_imgpathlist)
            all_filterlist.append(target_filterlist)
            all_refcatalogpathlist.append(target_refcatalogpathlist)
            all_idlist.append(f"{alert_instance.objname}_{Time(np.nanmean([img.mjd for img in target_imgset.target_images]), format='mjd').datetime.strftime('%Y%m%d_%H%M%S')}")
        tasks = [(target_imgpathlist, target_filterlist, target_refcatalogpathlist, id_, target_ra, target_dec, hostinfo_dict, True, self.config.tractor_photometry) 
                 for target_imgpathlist, target_filterlist, target_refcatalogpathlist, id_
                 in zip(all_imgpathlist, all_filterlist, all_refcatalogpathlist, all_idlist)]
        
        return self.run_parallel(tasks, tractor_photometry_worker,
                                step_name='tractor_photometry',
                                desc='Tractor photometry...',
                                batch_size = 3)
# %%
if __name__ == "__main__":
    db_connector = SQLConnector()
    db_data = db_connector.get_data(tbl_name = 'transient_status', select_key = '*')
    row = db_data[db_data['objname'] == 'AT2025fep'][0]
    alert_instance = Alert(**row)
    self = AlertProcessor()
    # alert_instance = Alert(objname = 'UDS', trigger_time = '2022-01-01')# %%
#%%
if __name__ == "__main__":
    # all_imagges = self.all_images
    # target_images = self.target_images
    # self.load_images_ezphot(alert_instance, file_pattern = '7DT*.fits')
    # alert_instance = Alert(objname = 'T10058')
    # alert_instance.trigger_time = Time('2001-01-01')
    self = AlertProcessor()
    # self.load_images_db(alert_instance, 'raw', None, alert_instance.trigger_time)
    self.load_images_ezphot(alert_instance, 'coadd_scaled*com.fits', obs_start_time = None)        
    # self.pipeline_after_stacking(alert_instance)
    # self.config.do_stack = False
    # self.config.do_DIA = False

    # self.all_images = all_imagess
    # self.target_images = target_images
    result_dict = self.tractor_photometry(alert_instance)
# %%
