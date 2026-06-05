

#%%
"""
WRITER: Hyeonho Choi (hhchoi1022@gmail.com)
DATE: 2026-03-13
DESCRIPTION: This module is a standalone module for the targets class. It is used to calculate the rise, transit, and set times of the targets for each day of the specified year.
ALL CALCULATION IS BASED ON astroplan library.
"""
#%%
import json
import yaml
from pathlib import Path
from astropy.time import Time
import numpy as np
from typing import Union, Any, Dict

#%%
class Configuration:
    """
    Base Configuration class for all modules.
    Loads multiple YAML (or JSON) config files and merges them.
    Later files override earlier ones.

    - Attribute-style access: config.foo -> self._dict["foo"]
    - Is picklable (can be sent to multiprocessing workers)
    """

    def __init__(self, config_filenames: Union[str, list[str]]):
        # use __dict__ directly here to avoid __setattr__ logic during init
        self.__dict__["_dict"] = {}
        self.__dict__["_config_paths"] = []

        config_filenames = np.atleast_1d(config_filenames)

        for filename in config_filenames:
            config_path = Path(__file__).parent / filename
            try:
                self._dict.update(self._load_config(config_path))
                self._config_paths.append(config_path)
            except FileNotFoundError:
                print(f"⚠️ Config file {config_path} not found, skipping.")

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self):
        # Convert Time objects nicely for printing
        attrs: Dict[str, Any] = {
            k: (v.iso if isinstance(v, Time) else v)
            for k, v in self._dict.items()
        }
        max_key_len = max((len(key) for key in attrs.keys()), default=0)
        attrs_str = "\n".join(
            [f"{k:{max_key_len}} : {v}" for k, v in attrs.items()]
        )
        return (f"===== Configuration =====\n"
                f"{attrs_str}")

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        """
        Called only if normal attribute lookup fails.
        We redirect public attribute access to the internal dict.
        """
        # avoid touching private attributes here, prevent recursion
        if name.startswith("_"):
            raise AttributeError(f"Attribute {name} not found")

        d = self.__dict__.get("_dict", {})
        if name in d:
            return d[name]
        raise AttributeError(f"Attribute {name} not found")

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Public attributes go into the config dict.
        Private attributes (starting with "_") go to the instance dict.
        """
        if name.startswith("_"):
            # bypass our own logic for private attrs
            super().__setattr__(name, value)
        else:
            # store user-facing config values in _dict
            self._dict[name] = value

    # ------------------------------------------------------------------
    # Pickling support
    # ------------------------------------------------------------------
    def __getstate__(self):
        """
        Return a picklable representation of the object.
        Only store the essential data, not methods or file handles.
        """
        return {
            "_dict": self._dict,
            "_config_paths": [str(p) for p in self._config_paths],
        }

    def __setstate__(self, state):
        """
        Restore object from pickled state.
        """
        self.__dict__["_dict"] = state["_dict"]
        self.__dict__["_config_paths"] = [Path(p) for p in state["_config_paths"]]

    # ------------------------------------------------------------------
    # Loading & helpers
    # ------------------------------------------------------------------
    def _load_config(self, config_path: Path) -> dict:
        suffix = config_path.suffix.lower()

        with open(config_path, "r") as f:
            if suffix in [".yaml", ".yml", ".config"]:
                return yaml.safe_load(f) or {}
            elif suffix == ".json":
                return json.load(f) or {}
            else:
                raise ValueError(f"❌ Unsupported config format: {suffix}")

    def update(self, **kwargs) -> None:
        """Update config dictionary with provided key-value pairs."""
        self._dict.update(kwargs)

    def to_plain_dict(self) -> dict:
        """
        Return a plain Python dict of the configuration contents.
        This is often better to send to multiprocessing workers than the
        full Configuration object itself.
        """
        # Note: if you have non-JSON-serializable objects in _dict,
        # you can customize this behavior.
        def default(o):
            if isinstance(o, Time):
                # store as ISO string or MJD if you want
                return o.iso
            return str(o)

        # round-trip through JSON for a deep copy of only serializable content
        return json.loads(json.dumps(self._dict, default=default))


#%%
from astropy.coordinates import EarthLocation, get_body
import astropy.units as u
from datetime import datetime
from astropy.time import Time
import pytz
import json
from pathlib import Path
from astroplan import Observer
import numpy as np

#%%
class mainObserver:
    """
    Class for observing astronomical objects and events from a specific location on Earth.

    Parameters
    ==========
    1. latitude : str
        The latitude of the observer's location in degrees.
    2. longitude : str
        The longitude of the observer's location in degrees.
    3. elevation : str
        The elevation of the observer's location in meters.
    4. name : str, optional
        The name of the observer.
    5. timezone : str, optional
        The timezone of the observer's location, in the format 'Area/Location'.

    Methods
    =======
    1. localtime(utctime: datetime = None) -> datetime
        Converts the provided UTC time to the observer's local time.
    2. siderialtime(time: datetime or Time = None, mode: str = 'mean') -> astropy.coordinates.Angle
        Calculates the local sidereal time at the provided UTC time.
    3. now() -> astropy.time.Time
        Returns the current UTC time.
    4. is_night(time: datetime or Time = None) -> bool
        Returns True if it is night at the observer's location at the provided UTC time.
    5. tonight(time: datetime or Time = None, horizon: float = -18) -> tuple
        Calculates the start and end times of tonight at the observer's location, starting from the provided UTC time.
    6. sun_radec(time: datetime or Time = None) -> astropy.coordinates.SkyCoord
        Calculates the RA and Dec of the Sun at the provided UTC time.
    7. sun_altaz(time: datetime or Time = None) -> astropy.coordinates.AltAz
        Calculates the altitude and azimuth of the Sun at the observer's location at the provided UTC time.
    8. sun_risetime(time: datetime or Time = None, mode: str = 'nearest', horizon: float = -18) -> astropy.time.Time
        Calculates the next rise time of the Sun at the observer's location, starting from the provided UTC time.
    9. sun_settime(time: datetime or Time = None, mode: str = 'nearest', horizon: float = -18) -> astropy.time.Time
        Calculates the next set time of the Sun at the observer's location, starting from the provided UTC time.
    10. moon_radec(time: datetime or Time = None) -> astropy.coordinates.SkyCoord
        Calculates the RA and Dec of the Moon at the provided UTC time.
    11. moon_altaz(time: datetime or Time = None) -> astropy.coordinates.AltAz
        Calculates the altitude and azimuth of the Moon at the observer's location at the provided UTC time.
    12. moon_risetime(time: datetime or Time = None, mode: str = 'nearest', horizon: float = -18) -> astropy.time.Time
        Calculates the next rise time of the Moon at the observer's location, starting from the provided UTC time.
    13. moon_settime(time: datetime or Time = None, mode: str = 'nearest', horizon: float = -18) -> astropy.time.Time
        Calculates the next set time of the Moon at the observer's location, starting from the
    """
    
    def __init__(self,
                 latitude: float = None,
                 longitude: float = None,
                 elevation: float = None,
                 name: str = None,
                 timezone: str = None):
        self.config = Configuration(config_filenames=['observer.config'])
        if latitude is None:
            latitude = self.config.latitude
        if longitude is None:
            longitude = self.config.longitude
        if elevation is None:
            elevation = self.config.elevation
        if name is None:
            name = self.config.telname
        if timezone is None:
            timezone = self.config.timezone
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.name = name
        self.timezone = timezone
        self._pytz_timezone = pytz.timezone(self.timezone)
        self._earthlocation = EarthLocation.from_geodetic(lat=self.latitude, lon=self.longitude, height=self.elevation)
        self._observer = Observer(location = self._earthlocation, name = name, timezone = self._pytz_timezone)

    ############ Time ############
    
    def localtime(self, 
                  utctimes : datetime or np.array = None):
        """
        Returns the datetime object representing the corresponding local time in the timezone 
        specified by the object's `_timezone` attribute.

        Parameters
        ==========
        1. utctime : datetime, optional
            The datetime object representing the time to convert to local time. If not provided,
            the current UTC time will be used.
            
        Returns
        =======
        1. localtime : datetime
            The datetime object representing the corresponding local time in the timezone 
            specified by the object's `_timezone` attribute.
        """
        
        if utctimes is None:
            utctimes = Time.now().datetime
        localtime = pytz.utc.localize(utctimes).astimezone(self._pytz_timezone)
        return localtime
    
    def siderialtime(self,
                     utctimes : datetime or Time or np.array = None,
                     mode : str = 'mean'): 
        """
        Calculate the local sidereal time at a given UTC time and mode.
        
        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to calculate the local sidereal time. If not provided, the current time is used.
        2. mode : str, optional
            The mode to use when calculating the local sidereal time. Can be either 'mean' (default) or 'apparent'.

        Returns
        =======
        1. local_sidereal_time : astropy.coordinates.Angle
            The local sidereal time at the given time, as an Angle object.
        """
        
        if utctimes is None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
        return self._observer.local_sidereal_time(utctimes, kind = mode)
    
    def now(self):
        """
        Get the current UTC time.
        
        Returns
        =======
        1. time : astropy.time.Time
            The current UTC time.
        """
        
        return Time.now()
    
    def is_night(self,
                 utctimes : datetime or Time or np.array = None):
        """
        Check if it is night at a given UTC time and location.
        
        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to check if it is night. If not provided, the current time is used.

        Returns
        =======
        1. is_night : bool
            True if it is night at the given time and location, False otherwise.
        """
        
        if utctimes is None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
        return self._observer.is_night(utctimes, horizon = -18*u.deg)
    
    def tonight(self,
                time : datetime or Time or np.array = None,
                horizon = -18):
        """
        Get the start and end times of tonight at a given UTC time and location.

        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to start the calculation of the start and end of tonight. If not provided, the current time is used.
        2. horizon : float, optional
            The horizon angle to use when calculating the start and end of tonight. Default is -18 degrees.

        Returns
        =======
        1. tonight : tuple
            A tuple of two astropy.time.Time objects representing the start and end times of tonight at the given time and location.
        """

        if time is None:
            time = Time.now()
        if not isinstance(time, Time):
            time = Time(time)
        return self._observer.tonight(time, horizon = horizon*u.deg)

    ############ Sun ############
    def sun_radec(self,
                  utctimes : datetime or Time or np.array = None):
        """
        Get the RA and Dec of the Sun at a given UTC time.

        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to calculate the RA and Dec of the Sun. If not provided, the current time is used.

        Returns
        =======
        1. sun_radec : astropy.coordinates.SkyCoord
            The RA and Dec of the Sun at the given time, as a SkyCoord object.
        """

        if utctimes is None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
        return get_body("sun", utctimes)
    
    def sun_altaz(self,
                  utctimes : datetime or Time or np.array = None):
        """
        Calculates the altitude and azimuth of the Sun at the given time and location.

        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to calculate the altitude and azimuth of the Sun. If not provided, the current time is used.

        Returns
        =======
        1. sun_altaz : astropy.coordinates.AltAz
            The altitude and azimuth of the Sun at the given time and location, as an AltAz object.
        """

        if utctimes is None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
        return self._observer.sun_altaz(utctimes)
    
    def sun_risetime(self,
                     utctimes : datetime or Time or np.array = None,
                     mode = 'nearest',
                     horizon = -18):
        """
        Calculates the next rise time of the Sun at the given time and location.

        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to calculate the next rise time of the Sun. If not provided, the current time is used.
        2. mode : str, optional
            The method to use for calculating the rise time of the Sun. Can be either 'nearest' (default), 'next', or 'previous'.
        3. horizon : float, optional
            The horizon angle to use when calculating the rise time of the Sun. Default is -18 degrees.

        Returns
        =======
        1. sun_rise_time : astropy.time.Time
            The next rise time of the Sun at the given time and location, as a Time object.
        """

        if utctimes is None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
        return self._observer.sun_rise_time(utctimes, which = mode, horizon = horizon * u.deg)
    
    def sun_settime(self,
                    utctimes : datetime or Time or np.array = None,
                    mode = 'nearest',
                    horizon = -18):
        """
        Calculates the next rise time of the Sun at the given time and location.

        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to calculate the next set time of the Sun. If not provided, the current time is used.
        2. mode : str, optional
            The method to use for calculating the set time of the Sun. Can be either 'nearest' (default), 'next', or 'previous'.
        3. horizon : float, optional
            The horizon angle to use when calculating the set time of the Sun. Default is -18 degrees.

        Returns
        =======
        1. sun_set_time : astropy.time.Time
            The next set time of the Sun at the given time and location, as a Time object.
        """

        if utctimes is None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
        return self._observer.sun_set_time(utctimes, which = mode, horizon = horizon * u.deg)
    
    ############ Moon ############
    def moon_radec(self,
                   utctimes : datetime or Time or np.array = None):
        """
        Calculates the RA and Dec of the Moon at the given time and location.
        
        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to calculate the RA and Dec of the Moon. If not provided, the current time is used.

        Returns
        =======
        1. moon_radec : astropy.coordinates.SkyCoord
            The RA and Dec of the Moon at the given time, as a SkyCoord object.
        """

        if utctimes == None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
        #moon_coord_icrs = moon_radec_gcrs.transform_to('icrs')
        return get_body("moon", utctimes)
    
    def moon_altaz(self,
                   utctimes : datetime or Time or np.array = None):
        """
        Calculates the altitude and azimuth of the Moon at the given time and location.

        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to calculate the altitude and azimuth of the Moon. If not provided, the current time is used.

        Returns
        =======
        1. moon_altaz : astropy.coordinates.AltAz
            The altitude and azimuth of the Moon at the given time and location, as an AltAz object.
        """

        if utctimes is None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
            
        return self._observer.moon_altaz(utctimes) #self._moon_altaz(radec = get_moon(time), time = time)

    def moon_risetime(self,
                      utctimes : datetime or Time or np.array = None,
                      mode = 'nearest',
                      horizon = -18):
        """
        Calculates the next rise time of the Moon at the given time and location.

        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to calculate the next rise time of the Moon. If not provided, the current time is used.
        2. mode : str, optional
            The method to use for calculating the rise time of the Moon. Can be either 'nearest' (default), 'next', or 'previous'.
        3. horizon : float, optional
            The horizon angle to use when calculating the rise time of the Moon. Default is -18 degrees.

        Returns
        =======
        1. moon_rise_time : astropy.time.Time
            The next rise time of the Moon at the given time and location, as a Time object.
        """

        if utctimes is None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
        return self._observer.moon_rise_time(utctimes, which = mode, horizon = horizon * u.deg)
    
    def moon_settime(self,
                     utctimes : datetime or Time or np.array = None,
                     mode = 'nearest',
                     horizon = -18):
        """
        Calculates the next set time of the Moon at the given time and location.

        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to calculate the next set time of the Moon. If not provided, the current time is used.
        2. mode : str, optional
            The method to use for calculating the set time of the Moon. Can be either 'nearest' (default), 'next', or 'previous'.
        3. horizon : float, optional
            The horizon angle to use when calculating the set time of the Moon. Default is -18 degrees.

        Returns
        =======
        1. moon_set_time : astropy.time.Time
            The next set time of the Moon at the given time and location, as a Time object.
        """

        if utctimes is None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
        return self._observer.moon_set_time(utctimes, which = mode, horizon = horizon * u.deg)
    
    def moon_phase(self,
                   utctimes : datetime or Time or np.array = None):
        """
        Calculates the phase of the Moon at the given time and location.
        
        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to calculate the phase of the Moon. If not provided, the current time is used.

        Returns
        =======
        1. k : float
            Fraction of moon illuminated
        """
        
        if utctimes is None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
        return self._observer.moon_illumination(utctimes)

#%%
# Other modules
from astroplan import FixedTarget, is_event_observable, is_observable, is_always_observable
from astroplan import AltitudeConstraint, MoonSeparationConstraint, AtNightConstraint
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import numpy as np
import datetime
from typing import List
import json
from pathlib import Path
# TCSpy modules
import warnings
from astropy.utils.exceptions import AstropyWarning

warnings.filterwarnings("ignore", category=AstropyWarning)
#%%
class Targets:
    """
    A class representing multiple astronomical targets for observation.

    Parameters
    ----------s
    observer : mainObserver
        An instance of mainObserver representing the observer.
    ra : numpy.array
        An array containing the right ascension values of the targets, in degrees.
    dec : numpy.array
        An array containing the declination values of the targets, in degrees.
    name : numpy.array, optional
        An array containing the names of the targets. Default is None.
        
    Attributes
    ----------
    ra : numpy.array
        An array of right ascension coordinates of the targets in degrees.
    dec : numpy.array
        An array of declination coordinates of the targets in degrees.
    coordinate : SkyCoord
        The astropy SkyCoord object representing the coordinates of the targets.
    target_astroplan : list
        A list of astroplan FixedTarget objects representing the targets.
    name : numpy.array
        An array of names corresponding to the targets.

    Methods
    -------
    rts_date(year=None, time_grid_resolution=3)
        Calculate the rise, transit, and set times of the targets for each day of the specified year.
    is_observable
    
    is_ever_observable(utctime=None, time_grid_resolution=1*u.hour)
        Determines whether the targets are observable during the specified time.
    is_always_observable(utctimes=None)
        Determines whether the targets are always observable during the specified time.
    is_event_observable(utctimes=None)
        Determines whether the targets are observable at the specified time or at the current time.
    altaz(utctime=None)
        Calculate the alt-az coordinates of the targets for the given time(s) in UTC.
    risetime(utctime=None, mode='nearest', horizon=30, n_grid_points=50)
        Calculate the next rise time of the targets as seen by the observer.
    settime(utctime=None, mode='nearest', horizon=30, n_grid_points=50)
        Calculate the time when the targets set below the horizon.
    meridiantime(utctime=None, mode='nearest', n_grid_points=50)
        Calculate the time at which the targets pass through the observer's meridian.
    hourangle(utctime=None)
        Calculate the hour angle of the targets for the given time(s) in UTC.
    """
    
    def __init__(self,
                 ra_list : np.array,
                 dec_list : np.array,
                 name_list : np.array = [],
                 observer : mainObserver = None,
                 ):
        config_path_list = [f'target.config',
                            f'nightsession.config']
        self.config = Configuration(config_filenames=config_path_list)
        
        if observer is None:
            self.observer = mainObserver()
        else:
            self.observer = observer
        self._observer = self.observer._observer
        self.ra = np.atleast_1d(ra_list)
        self.dec = np.atleast_1d(dec_list)     
        self.name = name_list
        
        self._coordinate = self._get_coordinate_radec(ra = ra_list, dec = dec_list)
        self._target = self._get_target(self._coordinate, name_list)
        self._constraints = self._get_constraints() # Contrasints from the configuration file
        self.n_targets = len(self._coordinate) if len(self.ra) > 1 else 1
        
    def __repr__(self):
        txt = f'Targets[n_targets = {self.n_targets}]'
        return txt
    
    def rts_date(self,
                 year : int = None,
                 time_grid_resolution : float = 3 # timegrid for checking the observability 
                 ):
        """
        Calculate the rise, transit, and set times of the targets for each day of the specified year.

        Parameters
        ----------
        year : int, optional
            The year for which to calculate the rise, transit, and set times. Default is None.
        time_grid_resolution : float, optional
            The time grid resolution for checking the observability. Default is 3.

        Returns
        -------
        numpy.array
            An array containing the calculated rise, transit, and set times of the targets for each day of the year.
        """
        # If start_date & end_date are not specified, defaults to current year
        if year == None:
            year = Time.now().datetime.year
        start_date = Time(datetime.datetime(year = year, month = 1, day = 1))
        end_date = Time(datetime.datetime(year = year+1 , month = 1, day = 1))

        expanded_arrays_observability = []
        expanded_arrays_altitude_midnight = []
        expanded_arrays_date = []
        current_date = start_date
        while current_date <= end_date:
            print(f"Calculating observability of the {len(self._coordinate)} targets on {current_date.strftime('%Y-%m-%d')}")
            midnight = Time((self._observer.tonight(current_date)[0].jd + self._observer.tonight(current_date)[1].jd )/2, format = 'jd')
            alt_at_midnight = self.altaz(midnight).alt.value
            expanded_arrays_altitude_midnight.append(alt_at_midnight)
            observablity = self.is_ever_observable(current_date, None, time_grid_resolution= time_grid_resolution * u.hour)
            expanded_arrays_observability.append(observablity)
            expanded_arrays_date.append(current_date.datetime)
            current_date += 1 * u.day

        observablity_array = np.array(expanded_arrays_observability).T
        altitude_array = np.array(expanded_arrays_altitude_midnight).T
        date_array = np.array(expanded_arrays_date)
        
        all_observability = []
        # Find the indices where the value changes
        for target_observability, target_altitude in zip(observablity_array, altitude_array):
            if all(target_observability):
                risedate = 'Always'
                setdate = 'Always'
                bestdate = date_array[np.argmax(target_altitude)]
            elif all(~target_observability):
                risedate = 'Never'
                setdate = 'Never'
                bestdate = 'Never'
            else:
                # Find index where observability False to True
                risedate_index = np.where(np.diff(target_observability.astype(int)) == 1)[0] + 1
                risedate = date_array[risedate_index[0]]
                # Find index where observability True to False
                setdate_index = np.where(np.diff(target_observability.astype(int)) == -1)[0] + 1
                setdate = date_array[setdate_index[0]]
                bestdate = date_array[np.argmax(target_altitude)]
            all_observability.append((risedate, bestdate, setdate))
        
        return np.array(all_observability)
        
    def moon_sep(self,
                 utctime : datetime or Time or np.array = None):
        """
        Calculate the separation between the Moon and the target.

        Parameters
        ----------
        utctime : datetime or Time, optional
            The time at which to calculate the separation. If not provided, the current time will be used.

        Returns
        -------
        moonsep : astropy.coordinates.Angle
            The separation between the Moon and the target.
        """
            
        if utctime is None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)
        moon_altaz = self.observer.moon_altaz(utctime)
        target_altaz = self.altaz(utctime)
        moonsep = np.round(moon_altaz.separation(target_altaz),2)
        return moonsep.value
    
    def is_observable(self,
                      utctime : datetime or Time or np.array = None) -> bool:
        """
        Determines whether the targets are observable at the specified time.
        """
        if utctime is None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)
        return is_event_observable(constraints = self._constraints, observer = self._observer, target = self._target, times = utctime)
   
    def is_ever_observable(self,
                           utctime_start : datetime or Time = None,
                           utctime_end : datetime or Time = None,
                           time_grid_resolution = 1 * u.hour) -> List[bool]:
        """
        Determines whether the targets are observable during the specified time.

        Parameters
        ----------
        utctime : datetime or Time, optional
            The time at which to check observability. Defaults to the current time.
        time_grid_resolution : astropy.units.Quantity, optional
            The time grid resolution for checking the observability. Default is 1 hour.

        Returns
        -------
        List[bool]
            A list of boolean values indicating whether each target is observable during the specified time.

        Raises
        ------
        TypeError
            If the provided time is not a valid datetime or Time object.
        """
        # Ensure both start and end times are Time objects
        if utctime_start is not None and not isinstance(utctime_start, Time):
            utctime_start = Time(utctime_start)
        if utctime_end is not None and not isinstance(utctime_end, Time):
            utctime_end = Time(utctime_end)
  
        # Handle case where both times are None
        if utctime_start is None and utctime_end is None:
            tonight = self.observer.tonight(Time.now())
            starttime, endtime = tonight[0], tonight[1]
        # Handle case where one time is None
        elif utctime_start is None:
            tonight = self.observer.tonight(utctime_end)
            starttime, endtime = tonight[0], utctime_end
        elif utctime_end is None:
            tonight = self.observer.tonight(utctime_start)
            starttime, endtime = utctime_start, tonight[1]
        # Handle case where both times are provided
        else:
            starttime, endtime = utctime_start, utctime_end              

        time_range = [starttime, endtime]
        return is_observable(constraints = self._constraints, observer = self._observer, targets = self._target, time_range = time_range, time_grid_resolution = time_grid_resolution)
    
    def is_always_observable(self,
                             utctimes : datetime or Time or np.array = None) -> bool:
        """
        Determines whether the targets are always observable during the specified time.

        Parameters
        ----------
        utctimes : datetime or Time or numpy.array, optional
            The time at which to check observability. Defaults to the current time.

        Returns
        -------
        bool
            True if all targets are always observable, False otherwise.

        Raises
        ------
        TypeError
            If the provided time is not a valid datetime, Time, or numpy.array object.
        """
        if utctimes is None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
        return is_always_observable(constraints = self._constraints, observer = self._observer, targets = self._target, times = utctimes)
    
    def altaz(self,
              utctime : datetime or Time or np.array = None) -> SkyCoord:
        """
        Calculate the alt-az coordinates of the targets for the given time(s) in UTC.

        Parameters
        ----------
        utctime : datetime or Time or numpy.array, optional
            The time(s) to calculate the alt-az coordinates for, in UTC. If not provided, the current time will be used.

        Returns
        -------
        SkyCoord
            The alt-az coordinates of the targets at the specified time(s).

        Raises
        ------
        TypeError
            If the provided time is not a valid datetime, Time, or numpy.array object.
        """
        if utctime is None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)
        return self._observer.altaz(utctime, target = self._target)
    
    def risetime(self,
                 utctime : datetime or Time = None ,
                 mode : str = 'nearest',
                 horizon : float = 30,
                 n_grid_points : int = 50) -> Time:
        """
        Calculate the next rise time of the targets as seen by the observer.

        Parameters
        ----------
        utctime : datetime or Time, optional
            The time to start searching for the next rise time. If not provided, the current time will be used.
        mode : str, optional
            The method used to determine the rise time. Possible values are 'next' (the next rise time), 'previous' (the previous rise time), or 'nearest' (the nearest rise time). Default is 'next'.
        horizon : float, optional
            The altitude of the horizon, in degrees. Default is 30.
        n_grid_points : int, optional
            The number of grid points to use in the interpolation. Default is 50.

        Returns
        -------
        Time
            The rise time of the targets as seen by the observer.

        """
        if utctime == None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)
        return self._observer.target_rise_time(utctime, target = self._target, which = mode, horizon = horizon*u.deg, n_grid_points = n_grid_points)
    
    def settime(self,
                utctime : datetime or Time or np.array = None,
                mode : str = 'nearest',
                horizon : float = 30,
                n_grid_points : int = 50) -> Time:
        """
        Calculate the time when the targets set below the horizon.

        Parameters
        ----------
        utctime : datetime or Time or numpy.array, optional
            The time to use as the reference time for the calculation, by default the current time.
        mode : str, optional
            Set to 'nearest', 'next' or 'previous', by default 'nearest'.
        horizon : float, optional
            The altitude of the horizon in degrees. Default is 30.
        n_grid_points : int, optional
            The number of grid points to use in the interpolation. Default is 50.

        Returns
        -------
        Time
            The time when the targets set below the horizon.

        """
        if utctime is None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)
        return self._observer.target_set_time(utctime, self._target, which = mode, horizon = horizon*u.deg , n_grid_points = n_grid_points)
    
    def meridiantime(self,
                     utctime : datetime or Time or np.array = None,
                     mode : str = 'nearest',
                     n_grid_points : int = 50) -> Time:
        """
        Calculate the time at which the targets pass through the observer's meridian.

        Parameters
        ----------
        utctime : datetime or Time or numpy.array, optional
            The time at which to calculate the meridian transit time. If not provided, the current time will be used.
        mode : str, optional
            Set to 'nearest', 'next' or 'previous', by default 'nearest'.
        n_grid_points : int, optional
            The number of grid points to use in the interpolation. Default is 50.

        Returns
        -------
        Time
            The time at which the targets pass through the observer's meridian.

        """
        if utctime is None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)
        return self._observer.target_meridian_transit_time(utctime, self._target, which = mode, n_grid_points = n_grid_points)
    
    def hourangle(self,
                  utctime : datetime or Time or np.array = None):
        """
        Calculate the hour angle of the targets for the given time(s) in UTC.

        Parameters
        ----------
        utctime : datetime or Time or numpy.array, optional
            The time(s) to calculate the hour angle of the targets for, in UTC. If not provided, the current time will be used.

        Returns
        -------
        astropy.coordinates.Angle
            The hour angle of the targets at the specified time(s).

        Raises
        ------
        ValueError
            If no target is specified for hourangle.

        """
        if utctime is None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)
        if not isinstance(self._target, FixedTarget):
            raise ValueError('No target is specified for hourangle')
        return self._observer.target_hour_angle(utctime, self._target)
    
    def plot_visibility(self,
                        utctime : datetime or Time or np.array = None,
                        show : bool = True,
                        save_path: str = None):
        """
        Creates a plot of the altitude and azimuth of a celestial object.
        
        Parameters
        ----------
        utctime : datetime or Time or np.array, optional
            The time(s) for which to calculate the altitude and azimuth of the celestial object. 
            If not provided, the current time is used.
        save_path: str, optional
            The path to save the plot. If not provided, the plot will not be saved.
        """
        import matplotlib.pyplot as plt
        now = Time.now()
        if utctime is None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)

        astro_sunsettime  = self.observer.sun_settime(utctime, horizon=self.config.sunalt_observation)
        astro_sunrisetime = self.observer.sun_risetime(astro_sunsettime, horizon=self.config.sunalt_observation, mode='next')
        sunsettime = self.observer.sun_settime(utctime, horizon=0)
        sunrisetime = self.observer.sun_risetime(sunsettime, horizon=0, mode='next')

        time_range_start = sunsettime.datetime - datetime.timedelta(hours=2)
        time_range_end   = sunrisetime.datetime + datetime.timedelta(hours=2)

        time_axis = np.arange(time_range_start, time_range_end, datetime.timedelta(minutes=5))

        moon_sep = self.moon_sep(time_axis)
        moon_altaz = self.observer.moon_altaz(time_axis)
        sun_altaz = self.observer.sun_altaz(time_axis)
        target_altaz = self.altaz(time_axis)

        plt.figure(dpi=300, figsize=(10, 4))

        if (now.datetime < time_range_end + datetime.timedelta(hours=3)) & (now.datetime > time_range_start - datetime.timedelta(hours=3)):
            plt.axvline(now.datetime, linestyle='--', c='r', label='Now')

        observable = is_event_observable(
            constraints=self._constraints,
            observer=self._observer,
            target=self._target,
            times=Time(time_axis)
        )[0]

        color_target = ['k' if obs else 'r' for obs in observable]

        plt.scatter(moon_altaz.obstime.datetime, moon_altaz.alt.value, c='b', s=10, marker='.', label='Moon')
        plt.scatter(sun_altaz.obstime.datetime, sun_altaz.alt.value, c='r', s=15, marker='.', label='Sun')
        plt.scatter(target_altaz.obstime.datetime, target_altaz.alt.value, c=color_target, s=30, marker='*', label='Target')

        # moon separation text every 1 hour
        step = 12  # 5 min x 12 = 60 min
        for i in range(0, len(time_axis), step):
            x = target_altaz.obstime.datetime[i]
            y = target_altaz.alt.value[i]
            sep = moon_sep[i]

            # 너무 낮은 고도에서는 글씨 안 보이니 조건 추가 가능
            if y > 10:
                plt.text(
                    x, y + 2, f'{sep:.0f}°',
                    fontsize=7,
                    ha='center',
                    va='bottom',
                    color='navy'
                )

        plt.fill_betweenx([10, 90], astro_sunsettime.datetime, astro_sunrisetime.datetime, alpha=0.1)
        plt.fill_betweenx([10, 90], sunsettime.datetime, sunrisetime.datetime, alpha=0.1)

        plt.axvline(x=astro_sunrisetime.datetime, linestyle='-', c='k', linewidth=0.5)
        plt.axvline(x=astro_sunsettime.datetime, linestyle='-', c='k', linewidth=0.5)
        plt.axvline(x=sunrisetime.datetime, linestyle='--', c='k', linewidth=0.5)
        plt.axvline(x=sunsettime.datetime, linestyle='--', c='k', linewidth=0.5)

        plt.xlim(time_range_start - datetime.timedelta(hours=1), time_range_end + datetime.timedelta(hours=1))
        plt.ylim(10, 90)
        plt.legend(loc=1)
        plt.xlabel('UTC [mm-dd hh]')
        plt.ylabel('Altitude [deg]')
        plt.grid()
        plt.xticks(rotation=45)
        # --- after all plotting commands ---

        # Build constraint text from config
        constraint_text = ['CONSTRAINTS', '-------------------------']
        if self.config.altitude_lower is not None:
            constraint_text.append(f"Min Alt: {self.config.altitude_lower}°")
        if self.config.altitude_upper is not None:
            constraint_text.append(f"Max Alt: {self.config.altitude_upper}°")
        if self.config.moonsep_lower is not None:
            constraint_text.append(f"Min Moon Sep: {self.config.moonsep_lower}°")

        # Join as multi-line string
        constraint_str = "\n".join(constraint_text)

        # Place text box inside the axes
        plt.gca().text(
            0.03, 0.85,             # position (axes fraction: right side, vertically centered)
            constraint_str,
            transform=plt.gca().transAxes,
            fontsize=9,
            va="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=1.0, edgecolor="gray")
        )
        if save_path is not None:
            save_path = Path(save_path)
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def _get_coordinate_radec(self,
                              ra,
                              dec,
                              frame : str = 'icrs') -> SkyCoord:
        return SkyCoord(ra = ra, dec = dec, frame = frame, unit = (u.deg, u.deg))

    def _get_target(self,
                    coord,
                    target_name : str = '') -> FixedTarget:
        return FixedTarget(coord = coord, name = target_name)

    def _get_constraints(self) -> list:
        constraint_all = []
        if (self.config.altitude_lower != None) & (self.config.altitude_upper != None):
            constraint_altitude = AltitudeConstraint(min = self.config.altitude_lower * u.deg, max = self.config.altitude_upper * u.deg, boolean_constraint = True)
            constraint_all.append(constraint_altitude)
        if self.config.moonsep_lower != None:
            constraint_moonsep = MoonSeparationConstraint(min = self.config.moonsep_lower * u.deg, max = None)
            constraint_all.append(constraint_moonsep)
        if self.config.sunalt_observation != None:
            constraint_sunalt_observation = AtNightConstraint(max_solar_altitude = self.config.sunalt_observation * u.deg)
            constraint_all.append(constraint_sunalt_observation)
        
        return constraint_all
    

# %%
if __name__ == '__main__':
    # Define observer 
    observer = mainObserver()
    print('Is night: ', observer.is_night())
    print('Tonight: ', observer.tonight())
    print('Sun rise time: ', observer.sun_risetime())
    print('Sun set time: ', observer.sun_settime())
    print('Moon rise time: ', observer.moon_risetime())
    print('Moon set time: ', observer.moon_settime())
    print('Moon phase: ', observer.moon_phase())
    print('Local time: ', observer.localtime())
    print('Sidereal time: ', observer.siderialtime())
    print('Now: ', observer.now())
    print('Sun RA/Dec: ', observer.sun_radec())
    print('Sun Alt/Az: ', observer.sun_altaz())
    print('Moon RA/Dec: ', observer.moon_radec())
    print('Moon Alt/Az: ', observer.moon_altaz())
    print('Moon phase: ', observer.moon_phase())
    
    # Define multiple targets
    targets = Targets(ra_list = np.array([180, 190, 200]), dec_list = np.array([-60, -60, -60]), name_list = np.array(['test1', 'test2', 'test3']), observer = observer)
    print('Alt/Az: ', targets.altaz())
    print('Moon Sep: ', targets.moon_sep())
    print('Is Observable: ', targets.is_observable())
    print('Is Always Observable: ', targets.is_always_observable())
    print('Is Ever Observable: ', targets.is_ever_observable())
    print('Rise Time: ', targets.risetime())
    print('Set Time: ', targets.settime())
    print('Meridian Time: ', targets.meridiantime())
    print('Hour Angle: ', targets.hourangle())
    
    # plot visibility function is not available for multiple targets. (Limitation of the astroplan library)
    #targets.plot_visibility()
    
    # Rise/Transit/Set date for each target (It may takes a while to calculate)
    rts_date = targets.rts_date()
    for r, t, s in rts_date:
        print(f'Rise date: {r}, Transit date: {t}, Set date: {s}')

#%%
if __name__ == '__main__':
    # Define single target
    target = Targets(ra_list = np.array([180.857]), dec_list = np.array([-60.736]), name_list = np.array(['test']), observer = observer)
    print('Alt/Az: ', target.altaz())
    print('Moon Sep: ', target.moon_sep())
    print('Is Observable: ', target.is_observable())
    print('Is Always Observable: ', target.is_always_observable())
    print('Is Ever Observable: ', target.is_ever_observable())
    print('Rise Time: ', target.risetime())
    print('Set Time: ', target.settime())
    print('Meridian Time: ', target.meridiantime())
    print('Hour Angle: ', target.hourangle())
    target.plot_visibility()
    
    
    
# %%
