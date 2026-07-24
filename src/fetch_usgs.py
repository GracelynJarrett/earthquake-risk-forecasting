"""
fetch_usgs.py — Historical earthquake data puller.

Purpose:
    Pull the full history (2000 to present) of magnitude 2.0+ earthquakes
    for our three regions (California, Japan, Greece) from the USGS
    earthquake web service, and save each region as a raw CSV file in
    data/raw/.

    USGS caps each request at 20,000 earthquakes, so for each region we pull
    ONE YEAR AT A TIME and combine the years together. This keeps every
    request small enough to succeed.

    The saved CSVs are the "raw" data. Cleaning happens later (Day 3), and
    the final cleaned data goes into a SQLite database (Day 4).
"""

# 'requests' calls the USGS web service and gets earthquake data back.
import requests

# 'pandas' organizes the earthquakes into a table and saves it as CSV.
import pandas as pd

# 'time' lets us pause briefly between requests, to be polite to USGS.
import time

# 'datetime' gives us the current year, so "present" updates automatically;
# timedelta + timezone let the inference pull grab a trailing "last N days" window.
from datetime import datetime, timedelta, timezone

# 'Path' builds file paths that work correctly on any operating system.
from pathlib import Path


# The USGS earthquake service address (same one the test pull used).
USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# The three regions we care about, each as a rectangle on the map.
# Numbers are degrees: latitude (north/south) and longitude (east/west).
# Negative longitude = west (California); positive = east (Japan, Greece).
REGIONS = {
    "california": {"min_lat": 32.0, "max_lat": 42.5, "min_lon": -125.0, "max_lon": -114.0},
    "japan":      {"min_lat": 24.0, "max_lat": 46.0, "min_lon": 122.0,  "max_lon": 146.0},
    "greece":     {"min_lat": 34.0, "max_lat": 42.0, "min_lon": 19.0,   "max_lon": 29.0},
}

# The time span to pull: from the year 2000 up through the current year.
START_YEAR = 2000
END_YEAR = datetime.now().year  # e.g. 2026 — updates itself automatically

# The smallest magnitude we want. Below this, USGS ignores the quake.
MIN_MAGNITUDE = 2.0

# How many days of history the INFERENCE pull grabs each run (enough for 7/30-day features).
RECENT_DAYS = 30

# Where the raw CSV files get saved: <project>/data/raw/
# Path(__file__) is this script's location; ".parent.parent" climbs up to the
# project root (out of src/), then we point into data/raw.
RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# USGS refuses any single request that would return more than this many quakes.
# We watch for it so we know if a one-year chunk was too big and got cut off.
USGS_MAX_RESULTS = 20000


def fetch_one_year(box, year):
    """
    Pull one region's magnitude 2.0+ earthquakes for a single calendar year.

    Args:
        box (dict): the region's rectangle (min/max lat and lon).
        year (int): the calendar year to pull (e.g. 2005).

    Returns:
        list: the earthquakes for that year, each as a GeoJSON "feature" dict.

    Raises:
        requests.HTTPError: if USGS responds with an error status.
    """
    # Build the time window for this year: Jan 1 up to the last second of Dec 31.
    # Using the last second (not next Jan 1) avoids grabbing the same quake twice
    # at the boundary between two years.
    start_time = f"{year}-01-01T00:00:00"
    end_time = f"{year}-12-31T23:59:59"

    # The filters we send to USGS: format, time window, the map rectangle,
    # and the minimum magnitude. The lat/lon keys come straight from our box.
    params = {
        "format": "geojson",
        "starttime": start_time,
        "endtime": end_time,
        "minlatitude": box["min_lat"],
        "maxlatitude": box["max_lat"],
        "minlongitude": box["min_lon"],
        "maxlongitude": box["max_lon"],
        "minmagnitude": MIN_MAGNITUDE,
    }

    # Send the request and stop loudly if USGS returns an error.
    response = requests.get(USGS_URL, params=params, timeout=60)
    response.raise_for_status()

    # Pull out the list of earthquakes (each quake is one "feature").
    quakes = response.json()["features"]

    # Safety check: if we got exactly the max, the year was too big and USGS
    # likely cut some quakes off. Warn so we can pull that year in smaller pieces.
    if len(quakes) >= USGS_MAX_RESULTS:
        print(f"    WARNING: {year} hit the {USGS_MAX_RESULTS} limit — data may be incomplete.")

    return quakes


def fetch_region(name, box):
    """
    Pull one region's full history (2000 to present) by looping year by year.

    Args:
        name (str): the region's name, used only for progress messages.
        box (dict): the region's rectangle (min/max lat and lon).

    Returns:
        list: every earthquake for the region across all years, combined into
              one list of GeoJSON "feature" dicts.
    """
    # Start with an empty list, then add each year's quakes onto it.
    all_quakes = []

    # Walk through every year from START_YEAR up to and including END_YEAR.
    # range() stops before its end value, so we add 1 to include END_YEAR.
    for year in range(START_YEAR, END_YEAR + 1):
        # Pull just this one year for this region.
        year_quakes = fetch_one_year(box, year)

        # Add this year's quakes to our running list (extend adds them one by one).
        all_quakes.extend(year_quakes)

        # Show progress so a long pull doesn't look frozen.
        print(f"  {name} {year}: {len(year_quakes)} quakes  (running total: {len(all_quakes)})")

        # Pause half a second before the next request. This is polite to USGS's
        # free service and avoids hammering it with rapid-fire calls.
        time.sleep(0.5)

    return all_quakes


def save_region_csv(name, quakes):
    """
    Flatten a region's GeoJSON quakes into a table and save it as a CSV.

    Args:
        name (str): the region's name, used for the file name and a column.
        quakes (list): the region's earthquakes as GeoJSON "feature" dicts.

    Returns:
        Path: the location of the CSV file that was written.
    """
    # Build one flat row (a dictionary) per earthquake, keeping a generous set
    # of fields. We keep more than we need now and trim during cleaning (Day 3).
    rows = []
    for quake in quakes:
        props = quake["properties"]                 # magnitude, place, time, quality metrics
        lon, lat, depth = quake["geometry"]["coordinates"]  # order is [lon, lat, depth]

        rows.append({
            "id": quake["id"],          # USGS's unique ID for this quake
            "region": name,             # which region this row belongs to
            "time": props["time"],      # when it happened (epoch milliseconds, UTC)
            "place": props["place"],    # human-readable location text
            "magnitude": props["mag"],  # the size of the quake
            "mag_type": props["magType"],  # how the magnitude was measured
            "longitude": lon,           # east/west position
            "latitude": lat,            # north/south position
            "depth_km": depth,          # how deep below the surface (km)
            "nst": props["nst"],        # number of stations that recorded it
            "gap": props["gap"],        # largest gap between stations (data quality)
            "dmin": props["dmin"],      # distance to nearest station (data quality)
            "rms": props["rms"],        # measurement error estimate (data quality)
            "type": props["type"],      # event type: earthquake, quarry blast, etc.
            "status": props["status"],  # "reviewed" by a human or "automatic"
        })

    # Turn our list of rows into a pandas table (DataFrame).
    df = pd.DataFrame(rows)

    # Convert the raw epoch-millisecond time into a real UTC date/time we can
    # read and sort by. This "time" column is what we'll split on later
    # (train/validate/test by date), so getting it right matters.
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)

    # Sort oldest-to-newest so the file reads in time order.
    df = df.sort_values("time").reset_index(drop=True)

    # Make sure the data/raw/ folder exists before writing into it.
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save the table to data/raw/<region>.csv. index=False leaves out pandas'
    # automatic row numbers, which we don't need in the file.
    output_path = RAW_DATA_DIR / f"{name}.csv"
    df.to_csv(output_path, index=False)

    return output_path


def fetch_recent(box, days=RECENT_DAYS):
    """
    Pull one region's magnitude 2.0+ quakes from the last `days` days (inference input).

    Same request as the historical pull, but with a trailing date window instead of a
    calendar year — 30 days is enough to compute today's 7- and 30-day features.

    Args:
        box (dict): the region's rectangle (min/max lat and lon).
        days (int): how many days back to pull. Defaults to RECENT_DAYS.

    Returns:
        list: the recent earthquakes as GeoJSON "feature" dicts.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "minlatitude": box["min_lat"], "maxlatitude": box["max_lat"],
        "minlongitude": box["min_lon"], "maxlongitude": box["max_lon"],
        "minmagnitude": MIN_MAGNITUDE,
    }
    response = requests.get(USGS_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.json()["features"]


def run_recent(days=RECENT_DAYS):
    """Pull the last `days` days for all regions and save as raw CSVs (the inference input)."""
    print(f"Pulling last {days} days of USGS quakes, magnitude {MIN_MAGNITUDE}+\n")
    for name, box in REGIONS.items():
        try:
            quakes = fetch_recent(box, days)
            path = save_region_csv(name, quakes)
            print(f"  {name}: {len(quakes)} quakes -> {path}")
        except requests.RequestException as error:
            print(f"  Could not pull {name}: {error}")


def main():
    """
    Pull all three regions (2000 to present) and save each as a raw CSV.

    For every region we fetch its full history, save it to data/raw/, and
    print a summary line. If USGS can't be reached for a region, we report
    it and move on to the next region instead of stopping everything.
    """
    print(f"Pulling USGS earthquakes {START_YEAR}-{END_YEAR}, magnitude {MIN_MAGNITUDE}+\n")

    # Go through each region one at a time (california, japan, greece).
    for name, box in REGIONS.items():
        print(f"Region: {name}")

        # Try to pull and save this region. If USGS errors out, print the
        # problem and keep going with the other regions.
        try:
            quakes = fetch_region(name, box)
            output_path = save_region_csv(name, quakes)
            print(f"  Saved {len(quakes)} quakes to {output_path}\n")
        except requests.RequestException as error:
            print(f"  Could not pull {name}: {error}\n")

    print("Done.")


# Only run main() when this file is launched directly (e.g.
# "python src/fetch_usgs.py"). If another file imports these functions
# later (like the pipeline), main() will NOT auto-run.
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "recent":
        run_recent()          # inference: last 30 days (python src/fetch_usgs.py recent)
    else:
        main()                # full history 2000-present (python src/fetch_usgs.py)
