from dataclasses import dataclass
from astropy.time import Time
import uuid

@dataclass
class AlertInfo:
    """Static alert information (exposure, host, timestamps)."""

    # --- Identifiers & timestamps ---
    alert_id: str
    alert_time: Time
    create_time: Time

    # --- Exposure setup ---
    exptime: float
    count: int
    obsmode: str
    filter_: str
    specmode: str
    ntelescope: int
    priority: int
    weight: int
    binning: int
    gain: int
    objtype: str
    is_ToO: bool
    is_rapid_ToO: bool
    note: str 
    source: str
    broker: str
    
    # --- Host galaxy info ---
    hostgalaxy_name: str
    hostgalaxy_radeg: float
    hostgalaxy_dedeg: float
    hostgalaxy_separation_arcsec: float
    hostgalaxy_dist: float
    hostgalaxy_distance_modulus: float
    hostgalaxy_z: float
    hostgalaxy_r1: float
    hostgalaxy_r2: float
    hostgalaxy_pa: float
    hostgalaxy_d_ell: float
    hostgalaxy_gmag: float
    hostgalaxy_bpmag: float
    hostgalaxy_rmag: float
    hostgalaxy_imag: float
    hostgalaxy_zmag: float
    hostgalaxy_w1mag: float
    hostgalaxy_w2mag: float
    hostgalaxy_e_bv: float
    hostgalaxy_logm: float

    # ===========================================================
    # Custom initializer accepting **kwargs
    # ===========================================================
    def __init__(self, alert_id: str, **kwargs):
        """Accept arbitrary keyword arguments (extra keys ignored)."""
        self.alert_id = str(alert_id)
        self.alert_time = Time(kwargs.get("alert_time", Time.now()))
        self.create_time = Time(kwargs.get("create_time", Time.now()))

        # Exposure
        self.exptime = float(kwargs.get("exptime", 100.0))
        self.count = int(kwargs.get("count", 3))
        self.obsmode = str(kwargs.get("obsmode", "Spec"))
        self.filter_ = kwargs.get("filter_")
        self.specmode = kwargs.get("specmode")
        self.ntelescope = kwargs.get("ntelescope") 
        self.priority = int(kwargs.get("priority", 50))
        self.weight = int(kwargs.get("weight", 1))
        self.binning = int(kwargs.get("binning", 1))
        self.gain = int(kwargs.get("gain", 2750))
        self.objtype = str(kwargs.get("objtype", "Request"))
        self.is_rapid_ToO = bool(kwargs.get("is_rapid_ToO", False))
        self.is_ToO = bool(kwargs.get("is_ToO", True))
        self.note = str(kwargs.get("note"))
        self.source = str(kwargs.get("source"))
        self.broker = str(kwargs.get("broker"))

        # Host galaxy
        self.hostgalaxy_name = kwargs.get("hostgalaxy_name")
        self.hostgalaxy_radeg = kwargs.get("hostgalaxy_radeg")
        self.hostgalaxy_dedeg = kwargs.get("hostgalaxy_dedeg")
        self.hostgalaxy_separation_arcsec = kwargs.get("hostgalaxy_separation_arcsec")
        self.hostgalaxy_dist = kwargs.get("hostgalaxy_dist")
        self.hostgalaxy_distance_modulus = kwargs.get("hostgalaxy_distance_modulus")
        self.hostgalaxy_z = kwargs.get("hostgalaxy_z")
        self.hostgalaxy_r1 = kwargs.get("hostgalaxy_r1")
        self.hostgalaxy_r2 = kwargs.get("hostgalaxy_r2")
        self.hostgalaxy_pa = kwargs.get("hostgalaxy_pa")
        self.hostgalaxy_d_ell = kwargs.get("hostgalaxy_d_ell")
        self.hostgalaxy_gmag = kwargs.get("hostgalaxy_gmag")
        self.hostgalaxy_bpmag = kwargs.get("hostgalaxy_bpmag")
        self.hostgalaxy_rmag = kwargs.get("hostgalaxy_rmag")
        self.hostgalaxy_imag = kwargs.get("hostgalaxy_imag")
        self.hostgalaxy_zmag = kwargs.get("hostgalaxy_zmag")
        self.hostgalaxy_w1mag = kwargs.get("hostgalaxy_w1mag")
        self.hostgalaxy_w2mag = kwargs.get("hostgalaxy_w2mag")
        self.hostgalaxy_e_bv = kwargs.get("hostgalaxy_e_bv")
        self.hostgalaxy_logm = kwargs.get("hostgalaxy_logm")
        self.mag_expected = kwargs.get("mag_expected")
        self.absmag_expected = kwargs.get("absmag_expected")
        
        if self.filter_ is not None:
            self.filter_ = str(self.filter_)
        if self.specmode is not None:
            self.specmode = str(self.specmode)
        if self.ntelescope is not None:
            self.ntelescope = int(self.ntelescope)
        if self.hostgalaxy_name is not None:
            self.hostgalaxy_name = str(self.hostgalaxy_name)
        if self.hostgalaxy_radeg is not None:
            self.hostgalaxy_radeg = float(self.hostgalaxy_radeg)
        if self.hostgalaxy_dedeg is not None:
            self.hostgalaxy_dedeg = float(self.hostgalaxy_dedeg)
        if self.hostgalaxy_separation_arcsec is not None:
            self.hostgalaxy_separation_arcsec = float(self.hostgalaxy_separation_arcsec)
        if self.hostgalaxy_dist is not None:
            self.hostgalaxy_dist = float(self.hostgalaxy_dist)
        if self.hostgalaxy_distance_modulus is not None:
            self.hostgalaxy_distance_modulus = float(self.hostgalaxy_distance_modulus)
        if self.hostgalaxy_z is not None:
            self.hostgalaxy_z = float(self.hostgalaxy_z)
        if self.hostgalaxy_r1 is not None:
            self.hostgalaxy_r1 = float(self.hostgalaxy_r1)
        if self.hostgalaxy_r2 is not None:
            self.hostgalaxy_r2 = float(self.hostgalaxy_r2)
        if self.hostgalaxy_pa is not None:
            self.hostgalaxy_pa = float(self.hostgalaxy_pa)
        if self.hostgalaxy_d_ell is not None:
            self.hostgalaxy_d_ell = float(self.hostgalaxy_d_ell)
        if self.hostgalaxy_gmag is not None:
            self.hostgalaxy_gmag = float(self.hostgalaxy_gmag)
        if self.hostgalaxy_bpmag is not None:
            self.hostgalaxy_bpmag = float(self.hostgalaxy_bpmag)
        if self.hostgalaxy_rmag is not None:
            self.hostgalaxy_rmag = float(self.hostgalaxy_rmag)
        if self.hostgalaxy_imag is not None:
            self.hostgalaxy_imag = float(self.hostgalaxy_imag)
        if self.hostgalaxy_zmag is not None:
            self.hostgalaxy_zmag = float(self.hostgalaxy_zmag)
        if self.hostgalaxy_w1mag is not None:
            self.hostgalaxy_w1mag = float(self.hostgalaxy_w1mag)
        if self.hostgalaxy_w2mag is not None:
            self.hostgalaxy_w2mag = float(self.hostgalaxy_w2mag)
        if self.hostgalaxy_e_bv is not None:
            self.hostgalaxy_e_bv = float(self.hostgalaxy_e_bv)
        if self.hostgalaxy_logm is not None:
            self.hostgalaxy_logm = float(self.hostgalaxy_logm)
        if self.mag_expected is not None:
            self.mag_expected = float(self.mag_expected)
        if self.absmag_expected is not None:
            self.absmag_expected = float(self.absmag_expected)

