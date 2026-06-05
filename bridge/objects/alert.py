
#%%
from uuid import uuid4
from astropy.time import Time
import numpy as np
import astropy.units as u
from dataclasses import fields
from pprint import pprint
from bridge.utils import HostGalaxyCatalog
from bridge.connector import SDSConnector
from bridge.objects import Targets, mainObserver
from bridge.objects import AlertInfo, AlertStatus
from bridge.objects import AlertExpectation
#%%

class Alert(AlertInfo, AlertStatus, AlertExpectation):
    def __init__(self,
                 ra: float = None, # RA in degrees
                 dec: float = None, # Dec in degrees
                 objname: str = None,
                 alert_id: str = None,
                 radius: float = 0.0,
                 **kwargs
                 ):
        # Basic information
        self.alert_id = alert_id
        if self.alert_id is None:
            self.alert_id = str(uuid4())
        AlertInfo.__init__(self, alert_id = self.alert_id, **kwargs)
        AlertStatus.__init__(self, alert_id = self.alert_id, **kwargs)
        AlertExpectation.__init__(self, alert_id = self.alert_id, **kwargs)
        
        self.ra = float(ra) if ra is not None else None
        self.dec = float(dec) if dec is not None else None
        
        # Object name
        self.objname = str(objname).replace(' ', '') if objname is not None else None
        
        # Radius
        if radius is None:
            radius = 0.0
        self.radius = float(radius)
        
        # Tile ID, RA, and Dec
        self.tile_id = None if 'tile_id' not in kwargs else kwargs['tile_id']
        self.tile_ra = None if 'tile_ra' not in kwargs else kwargs['tile_ra']
        self.tile_dec = None if 'tile_dec' not in kwargs else kwargs['tile_dec']
        self.tile_dec = None
        self.target = None
        if self.is_coordinate_given:
            self.tile_id, self.tile_ra, self.tile_dec = self.get_tile_id()
            self.target = Targets(ra_list = np.array(self.ra), dec_list = np.array(self.dec), name_list = np.array(self.objname), observer = mainObserver())
        
    def __repr__(self):
        data = {"alert_id": self.alert_id, "objname": self.objname, "ra": self.ra, "dec": self.dec, "radius": self.radius, "tile_id": self.tile_id, "tile_ra": self.tile_ra, "tile_dec": self.tile_dec}
        string = "🔷 ALERT SUMMARY"
        string += "\n────────────────────────────"
        for key, val in data.items():
            string += f"\n  • {key:25} = {val}"
        string += "\n────────────────────────────"
        return string 
    
    @property
    def is_coordinate_given(self):
        if self.ra is not None and self.dec is not None:
            return True
        else:
            return False
    
    @property
    def info(self):
        """Show all available attributes from Alert, AlertInfo, and AlertStatus."""
        # Get all dataclass fields from parents
        info_fields = [f.name for f in fields(AlertInfo)]
        
        # Build output dictionary
        data = {"AlertInfo": {k: getattr(self, k, None) for k in info_fields}}

        print("🔷 ALERT INFORMATION SUMMARY")
        print("────────────────────────────")
        for section, values in data.items():
            for key, val in values.items():
                print(f"  • {key:25} = {val}")
        print("────────────────────────────")
        return None
    
    @property
    def status(self):
        """Show all available attributes from Alert, AlertInfo, and AlertStatus."""
        # Get all dataclass fields from parents
        status_fields = [f.name for f in fields(AlertStatus)]
        
        # Build output dictionary
        data = {"AlertStatus": {k: getattr(self, k, None) for k in status_fields}}

        print("🔷 ALERT STATUS SUMMARY")
        print("────────────────────────────")
        for section, values in data.items():
            for key, val in values.items():
                print(f"  • {key:25} = {val}")
        print("────────────────────────────")
        return None
    
    @property
    def expectation(self):
        """Show all available attributes from Alert, AlertInfo, and AlertExpectation."""
        # Get all dataclass fields from parents
        expectation_fields = [f.name for f in fields(AlertExpectation)]
        data = {"AlertExpectation": {k: getattr(self, k, None) for k in expectation_fields}}
        
        print("🔷 ALERT EXPECTATION SUMMARY")
        print("────────────────────────────")
        for section, values in data.items():
            for key, val in values.items():
                print(f"  • {key:25} = {val}")
        print("────────────────────────────")
        return None
    
    def match_host(self, hostgalaxy_catalog: HostGalaxyCatalog,
                   search_radius_arcsec: float = 300.0,
                   max_dell: float = 2.5,
                   return_all: bool = False,
                   plot: bool = True,
                   save_path: str = None):
        if not self.is_coordinate_given:
            print("🔴 Coordinate not given!")
            return False
        host_galaxy = hostgalaxy_catalog.match_host(
            self.ra, self.dec,
            search_radius_arcsec,
            max_dell=max_dell,
            return_all=return_all,
            plot=plot,
            save_path=save_path
        )
        
        if host_galaxy is not None:
            for key, val in dict(host_galaxy).items():
                colname = f'hostgalaxy_{key.lower()}'
                self.__setattr__(colname, val)
            self.hostgalaxy_distance_modulus = 5 * np.log10(host_galaxy['Dist']) + 25   
    
    def is_observable(self, utctime : Time = Time.now()):
        if not self.is_coordinate_given:
            print("🔴 Coordinate not given!")
            return False
            
        is_observable = self.target.is_observable(utctime = utctime)
        return is_observable
    
    def is_observable_tonight(self, utctime : Time = Time.now()):
        if not self.is_coordinate_given:
            print("🔴 Coordinate not given!")
            return False
        from bridge.objects import NightSession
        nightsession = NightSession(utctime = utctime, observer = mainObserver())
        is_observable = self.target.is_ever_observable(utctime_start = nightsession.obsnight_utc.sunset_astro, utctime_end = nightsession.obsnight_utc.sunrise_astro, time_grid_resolution = 10 * u.minute)
        return is_observable
    
    def altaz(self, utctime : Time = Time.now()):
        if not self.is_coordinate_given:
            print("🔴 Coordinate not given!")
            return False
        altaz = self.target.altaz(utctime = utctime)
        return altaz
    
    def risetime(self, utctime : Time = Time.now()):
        if not self.is_coordinate_given:
            print("🔴 Coordinate not given!")
            return False
        risetime = self.target.risetime(utctime = utctime)
        return risetime
    
    def settime(self, utctime : Time = Time.now()):
        if not self.is_coordinate_given:
            print("🔴 Coordinate not given!")
            return False
        settime = self.target.settime(utctime = utctime)
        return settime
    
    def meridiantime(self, utctime : Time = Time.now()):
        if not self.is_coordinate_given:
            print("🔴 Coordinate not given!")
            return False
        meridiantime = self.target.meridiantime(utctime = utctime)
        return meridiantime
    
    def hourangle(self, utctime : Time = Time.now()):
        if not self.is_coordinate_given:
            print("🔴 Coordinate not given!")
            return False
        hourangle = self.target.hourangle(utctime = utctime)
        return hourangle
    
    def moon_sep(self, utctime : Time = Time.now()):
        if not self.is_coordinate_given:
            print("🔴 Coordinate not given!")
            return False
        moon_sep = self.target.moon_sep(utctime = utctime)
        return moon_sep
    
    def get_tile_id(self):
        if not self.is_coordinate_given:
            return None, None, None
        
        self.SDS_connector = SDSConnector()
        ra_list = np.atleast_1d(self.ra)
        dec_list = np.atleast_1d(self.dec)
        if isinstance(self.radius, (int, float)):
            radius_list = np.atleast_1d(len(ra_list)*[self.radius])
        else:
            radius_list = np.atleast_1d(self.radius)
        tile_tbl, matched_indices, _ = self.SDS_connector.find_overlapping_tiles(list_ra = ra_list, list_dec = dec_list, list_aperture = radius_list, visualize = False, save_fig = False, match_tolerance_minutes= 3, fraction_overlap_lower = 0.2)
        if len(tile_tbl) == 0:
            return None, None, None
        return tile_tbl['id'].tolist(), tile_tbl['ra'].tolist(), tile_tbl['dec'].tolist()
    
    def plot_visibility(self, utctime : Time = Time.now(), show : bool = True, save_path: str = None):
        if not self.is_coordinate_given:
            print("🔴 Coordinate not given!")
            return False
        self.target.plot_visibility(utctime = utctime, show = show, save_path = save_path)
#%%
if __name__ == '__main__':
    hg = HostGalaxyCatalog()
    self = Alert(ra = 320.21751, dec = -78.57789, objname = 'AT2025aard', radius = 0, alert_time = Time.now())
# %%
if __name__ == '__main__':
    from bridge.connector import SQLConnector
    db_status = SQLConnector().get_data(tbl_name = 'transient_status', select_key = '*')
# %%
