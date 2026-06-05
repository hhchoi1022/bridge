from astropy.time import Time
from datetime import datetime

#%%
def flexible_time_parser(value) -> Time:
    # If already a Time instance
    if isinstance(value, Time):
        return value

    # If datetime object
    if isinstance(value, datetime):
        return Time(value)

    # Convert to string
    value = str(value).strip()

    # Handle raw digits: YYMMDD or YYYYMMDD
    if value.isdigit():
        if len(value) == 6:  # YYMMDD
            value = '20' + value
        if len(value) == 8:  # YYYYMMDD
            value = f"{value[:4]}-{value[4:6]}-{value[6:]}"
        return Time(value, format='iso')

    # Handle ISO or ISO with time
    try:
        return Time(value, format='iso', scale='utc')
    except ValueError:
        pass

    # Fallback to automatic parsing
    return Time(value)