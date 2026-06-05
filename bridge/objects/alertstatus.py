#%%
from dataclasses import dataclass, field
from astropy.time import Time
import uuid
#%%
def _to_time(value):
    if value is None:
        return None
    return Time(value)

@dataclass
class AlertStatus:
    """Minimal standalone container for alert runtime and processing status."""
    alert_id: str
    status_id: str
    is_triggered: bool
    is_observed: bool
    last_observed_time: Time
    is_processed: bool
    last_processed_time: Time
    is_reference_ready: bool
    is_completed: bool
    trigger_time: Time
    update_time: Time
    tile_id: str
    tile_ra: float
    tile_dec: float

    def __init__(self, alert_id: str, **kwargs):
        """Accepts arbitrary keyword arguments (extra keys ignored)."""
        self.alert_id = str(alert_id)
        self.status_id = str(kwargs.get("status_id", uuid.uuid4()))
        self.is_triggered = bool(kwargs.get("is_triggered", False))
        self.is_observed = bool(kwargs.get("is_observed", False))
        self.last_observed_time = _to_time(kwargs.get("last_observed_time"))
        self.is_processed = bool(kwargs.get("is_processed", False))
        self.last_processed_time = _to_time(kwargs.get("last_processed_time"))
        self.is_reference_ready = bool(kwargs.get("is_reference_ready", False))
        self.is_completed = bool(kwargs.get("is_completed", False))
        self.trigger_time = _to_time(kwargs.get("trigger_time"))
        self.update_time = _to_time(kwargs.get("update_time", Time.now()))
        self.tile_id = kwargs.get("tile_id", None)
        self.tile_ra = kwargs.get("tile_ra", None)
        self.tile_dec = kwargs.get("tile_dec", None)
        
        if self.trigger_time is not None:
            self.trigger_time = Time(self.trigger_time)
        
