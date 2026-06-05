
#%%
import requests
import json
import os
import shutil
#%%
class AlertDownloader:

    def __init__(self):
        self.api_url = "https://gracedb.ligo.org/apiweb/superevents/"

    def get_latest_event(self):
        """Fetch the latest gravitational wave event from GraceDB, ignoring events with 'MS' prefix."""
        response = requests.get(self.api_url)
        if response.status_code == 200:
            events = response.json().get("superevents", [])
            for event in events:
                superevent_id = event.get("superevent_id", "")
                if superevent_id.startswith("S"):
                    return superevent_id  # Return the first valid event
        return None
    
    def search_event(self, event_id):
        """Search for a specific gravitational wave event by event_id."""
        response = requests.get(f"{self.api_url}{event_id}/")
        if response.status_code == 200:
            return response.json()  # Return full event details
        return None  # Event not found or request failed
    
    def save_localization_file(self, event_id, save_path='./alerts'):
        """Retrieve the sky localization FITS file URL for a given event."""
        response = requests.get(f"{self.api_url}{event_id}/files/")
        if response.status_code == 200:
            data = response.json()
            bilby_file_key = 'Bilby.fits.gz'
            bilby_figure_key = 'Bilby.png'
            bayestar_file_key = 'bayestar.fits.gz'
            bayestar_figure_key = 'bayestar.png'

            for file_key in [bilby_file_key, bilby_figure_key, bayestar_file_key, bayestar_figure_key]:
                if file_key in data.keys():
                    url = data[file_key]
                    os.makedirs(os.path.join(save_path, event_id), exist_ok=True)
                    path = os.path.join(save_path, event_id, file_key)

                    # Download the file
                    self._download_file(url=url, abspath=path)


    def _download_file(self, url, abspath):
        """Download a file from the given URL."""
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(abspath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded: {abspath}")
            return abspath
        else:
            print("Failed to download the file.")
            return None
# %%
if __name__ == '__main__':
    A = AlertDownloader()
    alertlist = A.search_event('S250205bk')
# %%
A.save_localization_file('S250205bk')
# %%
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
#%%
class AlertDecoder:

    def __init__(self):
        pass

    def load_sky_localization(self, fits_file):
        """Load and decode sky localization from an LVK alert FITS file."""
        hpx, header = hp.read_map(fits_file, h=True)
        return hpx, header
    


    def get_high_probability_regions(self, hpx, probability_threshold=0.9):
        """Identify HEALPix pixels covering the highest probability regions."""
        i = np.flipud(np.argsort(hpx))
        sorted_credible_levels = np.cumsum(hpx[i])
        credible_levels = np.empty_like(sorted_credible_levels)
        credible_levels[i] = sorted_credible_levels
        credible_levels <= probability_threshold
        return credible_levels

    def generate_tiling_pattern(self, selected_pixels, nside):
        """Generate a matched tiling pattern based on high-probability sky regions."""
        # Convert HEALPix pixel indices to celestial coordinates
        theta, phi = hp.pix2ang(nside, selected_pixels)
        ra = np.degrees(phi)
        dec = 90 - np.degrees(theta)

        # Define tiling pattern (e.g., grid-based, predefined telescope FOV)
        tile_size = 4.0  # Example tile size in degrees
        tiling = []

        for r, d in zip(ra, dec):
            tiling.append((r, d, tile_size))  # Store RA, Dec, and tile size

        return tiling

    def plot_sky_localization(self, hpx):
        """Plot the decoded sky localization and the selected tiling pattern."""
        hp.mollview(hpx)
# %%
B = AlertDecoder()
# %%
hpx, _ = B.load_sky_localization(fits_file = './alerts/S250205bk/Bilby.fits.gz')
# %%
C = B.get_high_probability_regions(hpx, 0.9)
#B.plot_sky_localization(hpx)
# %%
npix = len(hpx)
# %%
nside = hp.npix2nside(npix)
# %%
ipix = 123
theta, phi = hp.pix2ang(nside, ipix)
ra = np.rad2deg(phi)
dec = np.rad2deg(0.5 * np.pi - theta)

# %%
