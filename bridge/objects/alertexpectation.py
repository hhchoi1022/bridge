from dataclasses import dataclass

@dataclass
class AlertExpectation:
    alert_id: str
    # --- Expected photometry -
    
    expected_mjd: float
    last_observed_mjd: float
    last_observed_mag: float
    last_observed_absmag: float
    last_observed_filter: str
    last_observed_source: str    
    
    last_expected_mjd: float
    last_expected_phase: float
    last_expected_transient_type: str
    last_expected_mag: float
    last_expected_absmag: float
    last_expected_trend: str
    last_expected_filter: str
    last_expected_source: str    

    # ===========================================================
    # Custom initializer accepting **kwargs
    # ===========================================================
    def __init__(self, **kwargs):
        """Accept arbitrary keyword arguments (extra keys ignored)."""
        self.alert_id = kwargs.get("alert_id")
        
        self.expected_mjd = kwargs.get("expected_mjd")
        self.observed_mjd = kwargs.get("observed_mjd")
        self.observed_mag = kwargs.get("observed_mag")
        self.observed_absmag = kwargs.get("observed_absmag")
        self.observed_filter = kwargs.get("observed_filter")
        self.observed_source = kwargs.get("observed_source")

        self.last_expected_transient_type = kwargs.get("last_expected_transient_type")
        self.last_expected_mjd = kwargs.get("last_expected_mjd")
        self.last_expected_phase = kwargs.get("last_expected_phase")
        self.last_expected_mag = kwargs.get("last_expected_mag")
        self.last_expected_absmag = kwargs.get("last_expected_absmag")
        self.last_expected_trend = kwargs.get("last_expected_trend")
        self.last_expected_filter = kwargs.get("last_expected_filter")
        self.last_expected_source = kwargs.get("last_expected_source")