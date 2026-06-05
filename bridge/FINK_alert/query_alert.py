
# %%
""" Poll the Fink servers only once at a time """
from fink_client.consumer import AlertConsumer
from fink_client.configuration import load_credentials

from astropy.time import Time

import time
import tabulate

def poll_single_alert(myconfig, topics) -> None:
    """ Connect to and poll fink servers once.

    Parameters
    ----------
    myconfig: dic
        python dictionnary containing credentials
    topics: list of str
        List of string with topic names
    """
    maxtimeout = 100

    # Instantiate a consumer
    consumer = AlertConsumer(topics, myconfig)

    # Poll the servers
    topic, alert, key = consumer.poll(maxtimeout)

    # Analyse output - we just print some values for example
    if topic is not None:
        utc = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
        table = [
            [
                Time(alert['candidate']['jd'], format='jd').iso,
                utc,
                topic,
                alert['objectId'],
                alert['cdsxmatch'],
                alert['candidate']['magpsf']
            ],
        ]
        headers = [
            'Emitted at (UTC)',
            'Received at (UTC)',
            'Topic',
            'objectId',
            'Simbad',
            'Magnitude'
        ]
        print(tabulate(table, headers, tablefmt="pretty"))
    else:
        print(
            'No alerts received in the last {} seconds'.format(
                maxtimeout
            )
        )

    # Close the connection to the servers
    consumer.close()


if __name__ == "__main__":
    """ Poll the servers only once at a time """

    # to fill
    myconfig = {
        'bootstrap.servers': 'kafka-ztf.fink-broker.org:24499',
        'group.id': 'hyeonho_snia'
    }

    topics = ['fink_early_sn_candidates_ztf']

    poll_single_alert(myconfig, topics)


# %%
import io
import requests
import pandas as pd

# Get all classified SN Ia from TNS between March 1st 2021 and March 5th 2021
r = requests.post(
  "https://api.fink-portal.org/api/v1/latests",
  json={
    "class": "Early SN Ia candidate",
    # "trend": "rising",
    "n": "1000", 
    "startdate": "2025-12-20"
  }
)

# Format output in a DataFrame
pdf = pd.read_json(io.BytesIO(r.content))
# %%
from astropy.table import Table
tbl = Table.from_pandas(pdf)
# %%
tbl.columns
# %%
requests.post('https://api.fink-portal.org/api/v1/classes', json={}).json()
# %%

tbl['d:spicy_class']



# %%
Index(['d:DR3Name', 'd:Plx', 'd:anomaly_score', 'd:blazar_stats_m0',
       'd:blazar_stats_m1', 'd:blazar_stats_m2', 'd:cdsxmatch', 'd:delta_time',
       'd:e_Plx', 'd:from_upper', 'd:gaiaClass', 'd:gaiaVarFlag', 'd:gcvs',
       'd:is_transient', 'd:jd_first_real_det', 'd:jdstarthist_dt',
       'd:lc_features_g', 'd:lc_features_r', 'd:lower_rate', 'd:mag_rate',
       'd:mangrove_2MASS_name', 'd:mangrove_HyperLEDA_name',
       'd:mangrove_ang_dist', 'd:mangrove_lum_dist', 'd:mulens',
       'd:nalerthist', 'd:rf_kn_vs_nonkn', 'd:rf_snia_vs_nonia', 'd:roid',
       'd:sigma_rate', 'd:slsn_score', 'd:snn_sn_vs_all',
       'd:snn_snia_vs_nonia', 'd:spicy_class', 'd:spicy_id', 'd:tns',
       'd:upper_rate', 'd:vsx', 'd:x3hsp', 'd:x4lac', 'i:candid', 'i:chipsf',
       'i:classtar', 'i:dec', 'i:diffmaglim', 'i:distnr', 'i:distpsnr1',
       'i:drb', 'i:fid', 'i:field', 'i:isdiffpos', 'i:jd', 'i:jdendhist',
       'i:jdstarthist', 'i:maggaia', 'i:magnr', 'i:magpsf', 'i:magzpsci',
       'i:ndethist', 'i:neargaia', 'i:nid', 'i:nmtchps', 'i:objectId',
       'i:publisher', 'i:ra', 'i:rb', 'i:rcid', 'i:sgscore1', 'i:sigmagnr',
       'i:sigmapsf', 'i:ssdistnr', 'i:ssmagnr', 'i:ssnamenr', 'i:tooflag',
       'i:xpos', 'i:ypos', 'd:tracklet', 'v:classification', 'v:lastdate',
       'v:firstdate', 'v:lapse', 'v:constellation'],
      dtype='object')