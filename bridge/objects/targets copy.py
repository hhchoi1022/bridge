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
from pathlib import Path
import warnings
from astropy.utils.exceptions import AstropyWarning
from bridge.objects.observer import mainObserver
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
                 constraint_alaitude_lower: float = 30,
                 constraint_alaitude_upper: float = 88,
                 constraint_moonsep_lower: float = 30,
                 ):
        self.constraint_alaitude_lower = constraint_alaitude_lower
        self.constraint_alaitude_upper = constraint_alaitude_upper
        self.constraint_moonsep_lower = constraint_moonsep_lower
        
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
        moon_coord = self.observer.moon_radec(utctime)
        target_coord = self._coordinate
        moonsep = np.round(moon_coord.separation(target_coord),2)
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
        astro_sunsettime  = self.observer.sun_settime(utctime, horizon = -15)
        astro_sunrisetime = self.observer.sun_risetime(astro_sunsettime, horizon = -15, mode = 'next')
        sunsettime = self.observer.sun_settime(utctime, horizon = 0)
        sunrisetime = self.observer.sun_risetime(sunsettime, horizon = 0, mode = 'next')
        time_range_start, time_range_end = sunsettime.datetime - datetime.timedelta(hours = 2), sunrisetime.datetime + datetime.timedelta(hours = 2)
        time_axis = np.arange(time_range_start, time_range_end, datetime.timedelta(minutes = 5))
        moon_altaz = self.observer.moon_altaz(time_axis)
        sun_altaz = self.observer.sun_altaz(time_axis)
        target_altaz = self.altaz(time_axis)
        plt.figure(dpi = 300, figsize = (10, 4))
        if (now.datetime < time_range_end + datetime.timedelta(hours = 3)) & (now.datetime > time_range_start - datetime.timedelta(hours = 3)):
            plt.axvline(now.datetime, linestyle = '--', c='r', label = 'Now')
        # Check per-time observability under constraints
        observable = is_event_observable(
            constraints=self._constraints,
            observer=self._observer,
            target=self._target,
            times=Time(time_axis)
        )[0]

        # observable is a boolean array, same length as time_axis
        color_target = ['k' if obs else 'r' for obs in observable]
        plt.scatter(moon_altaz.obstime.datetime, moon_altaz.alt.value, c = 'b', s = 10, marker = '.', label ='Moon')
        plt.scatter(sun_altaz.obstime.datetime, sun_altaz.alt.value, c = 'r', s = 15, marker = '.', label = 'Sun')
        plt.scatter(target_altaz.obstime.datetime, target_altaz.alt.value, c = color_target, s = 30, marker = '*', label = 'Target')
        plt.fill_betweenx([10,90], astro_sunsettime.datetime, astro_sunrisetime.datetime, alpha = 0.1)
        plt.fill_betweenx([10,90], sunsettime.datetime, sunrisetime.datetime, alpha = 0.1)
        # Fill between the target and the horizon
        plt.axvline(x=astro_sunrisetime.datetime, linestyle = '-', c='k', linewidth = 0.5)
        plt.axvline(x=astro_sunsettime.datetime, linestyle = '-', c='k', linewidth = 0.5)
        plt.axvline(x=sunrisetime.datetime, linestyle = '--', c='k', linewidth = 0.5)
        plt.axvline(x=sunsettime.datetime, linestyle = '--', c='k', linewidth = 0.5)
        plt.xlim(time_range_start - datetime.timedelta(hours = 1), time_range_end + datetime.timedelta(hours = 1))
        plt.ylim(10, 90)
        plt.legend(loc = 1)
        plt.xlabel('UTC [mm-dd hh]')
        plt.ylabel('Altitude [deg]')
        plt.grid()
        plt.colorbar(label = 'Azimuth [deg]')
        plt.xticks(rotation = 45)
        
        # --- after all plotting commands ---

        # Build constraint text from config
        constraint_text = ['CONSTRAINTS', '-------------------------']
        if self.constraint_alaitude_lower is not None:
            constraint_text.append(f"Min Alt: {self.constraint_alaitude_lower}°")
        if self.constraint_alaitude_upper is not None:
            constraint_text.append(f"Max Alt: {self.constraint_alaitude_upper}°")
        if self.constraint_moonsep_lower is not None:
            constraint_text.append(f"Min Moon Sep: {self.constraint_moonsep_lower}°")

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
        if (self.constraint_alaitude_lower != None) & (self.constraint_alaitude_upper != None):
            constraint_altitude = AltitudeConstraint(min = self.constraint_alaitude_lower * u.deg, max = self.constraint_alaitude_upper * u.deg, boolean_constraint = True)
            constraint_all.append(constraint_altitude)
        if self.constraint_moonsep_lower != None:
            constraint_moonsep = MoonSeparationConstraint(min = self.constraint_moonsep_lower * u.deg, max = None)
            constraint_all.append(constraint_moonsep)
        
        return constraint_all
    

# %%
if __name__ == '__main__':
    from bridge.objects import mainObserver
    observer = mainObserver()
    targets = Targets(ra_list = np.array([24.857]), dec_list = np.array([-60.736]), name_list = np.array(['test']), observer = observer)
    # targets.plot_visibility()
# %%