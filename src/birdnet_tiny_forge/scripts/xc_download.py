import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import click

_BASE_URL = "https://xeno-canto.org/api/3/recordings"
_RATE_LIMIT_DELAY = 0.01  # xenocanto API docs says no rate limit enforced, but let's be good citizens
_OUTDIR = Path(__file__).parents[3] / "data" / "01_raw" / "audio_clips"
_DEFAULT_QUERY_PARAMS = {"len": '"<30"'}


@click.command()
@click.option('--api-key', required=True, help='Xeno-canto API key')
@click.option('--species-file', required=True, help='File where each line is a species to download (scientific name).')
@click.option('--outdir', default=_OUTDIR, help='Base directory path to store downloaded recordings')
@click.option('--n-recs', default=100, help='Number of recordings per species')
def xc_download(
    api_key: str,
    outdir: str,
    species_file: str,
    n_recs: int = 100,
):
    """
    Downloads audio recordings from the Xeno-canto API based on
    species queries and caches them locally to avoid re-downloading.

    :param outdir: Base directory path to store downloaded recordings
    :param api_key: Xeno-canto API key
    :param species: List of species names to download
    :param n_recs: Number of recordings per species
    :param query_params: Additional query parameters for API calls
    """
    outdir = Path(outdir).expanduser().absolute()
    species_file = Path(species_file).expanduser().absolute()
    outdir.mkdir(parents=True, exist_ok=True)
    query_params = {**_DEFAULT_QUERY_PARAMS}

    if not species_file.exists():
        raise FileNotFoundError(f"Species file {species_file} does not exist")
    species = [x.strip() for x in species_file.read_text().splitlines()]

    for species in species:
        species_dir = Path(outdir) / species
        species_dir.mkdir(exist_ok=True)

        print(f"Species {species}: Downloading {n_recs} recordings")

        # build query
        query_parts = [f'sp:"{species}"']
        for key, value in query_params.items():
            if isinstance(value, str) and " " in value:
                query_parts.append(f'{key}:"{value}"')
            else:
                query_parts.append(f'{key}:{value}')
        query = " ".join(query_parts)
        page = 1
        per_page = min(10, n_recs)  # API max is 500, but let's use smaller batches

        n_downloaded = 0
        while n_downloaded < n_recs:

            # Build API request
            params = {
                "query": query,
                "key": api_key,
                "page": page,
                "per_page": per_page
            }

            try:
                start = time.time()
                response = requests.get(_BASE_URL, params=params)
                response.raise_for_status()
                elapsed = time.time() - start
                data = response.json()
                to_wait = max(0.0, _RATE_LIMIT_DELAY - elapsed)
                time.sleep(to_wait)

                if "error" in data:
                    print(f"API Error for {species}: {data['error']}")
                    break

                recordings = data.get("recordings", [])
                if not recordings:
                    print(f"No more recordings available for {species}")
                    break

                # Download audio files
                for i, recording in enumerate(recordings):
                    if n_downloaded >= n_recs:
                        break

                    recording_id = recording["id"]
                    file_url = recording['file']
                    ext = recording['file-name'].split(".")[-1]

                    # Generate filename
                    filename = f"{recording_id}.{ext}"
                    filepath = outdir / species / filename

                    # Skip if already exists
                    if filepath.exists():
                        print(f"{n_downloaded}/{n_recs} File already exists: {filename}")
                        n_downloaded += 1
                        continue

                    # Download audio file
                    try:
                        start = time.time()
                        audio_response = requests.get(file_url, stream=True)
                        audio_response.raise_for_status()
                        elapsed = time.time() - start

                        with open(filepath, 'wb') as f:
                            for chunk in audio_response.iter_content(chunk_size=8192):
                                f.write(chunk)

                        print(f"{n_downloaded}/{n_recs} Downloaded: {filename}")
                        to_wait = max(0.0, _RATE_LIMIT_DELAY - elapsed)
                        time.sleep(to_wait)
                        n_downloaded += 1

                    except requests.RequestException as e:
                        print(f"Failed to download {filename}: {e}")
                        continue

                # Check if we consumed all pages
                if page >= data.get("numPages", 1):
                    break

                page += 1

            except requests.RequestException as e:
                print(f"API request failed for {species}: {e}")
                break


if __name__ == "__main__":
    xc_download()
