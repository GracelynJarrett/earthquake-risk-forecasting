"""
usgs_test_pull.py — USGS connection smoke test.

Purpose:
    A tiny script to confirm we can successfully reach the USGS earthquake
    web service and get real data back. It asks for the last 24 hours of
    magnitude 2.0+ earthquakes worldwide, then prints how many came back
    and shows one sample record.

    This is NOT the real data pipeline. It only answers one question:
    "Does the connection to USGS work?" Cleaning and storing come later.
"""

# 'requests' lets us call a web URL and get its response, like a browser does.
# We use it to send our request to the USGS earthquake service.
import requests

# 'datetime' and 'timedelta' let us work with dates and times.
# 'timezone' lets us mark a time as UTC (the world time standard USGS uses).
# We use these to calculate "24 hours ago" so we can ask for recent quakes.
from datetime import datetime, timedelta, timezone


# The USGS earthquake service lives at this web address (the "endpoint").
# Every request we make for earthquake data gets sent to this URL.
USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def fetch_recent_quakes():
    """
    Ask USGS for the last 24 hours of magnitude 2.0+ earthquakes worldwide.

    Returns:
        dict: the earthquake data USGS sends back, already turned into a
              Python dictionary (from its GeoJSON format).

    Raises:
        requests.HTTPError: if USGS responds with an error (e.g. bad request
                            or the service is down).
    """
    # Figure out the time window: from 24 hours ago until right now.
    # USGS expects dates as text (ISO format), so we convert them with isoformat().
    end_time = datetime.now(timezone.utc)        # "now" in UTC (USGS uses UTC time)
    start_time = end_time - timedelta(hours=24)  # 24 hours before now

    # These are the filters we send to USGS, like options on a search form.
    params = {
        "format": "geojson",                 # ask for GeoJSON (easy to read as a dict)
        "starttime": start_time.isoformat(),  # earliest quake time we want
        "endtime": end_time.isoformat(),      # latest quake time we want
        "minmagnitude": 2.0,                  # ignore anything smaller than 2.0
    }

    # Send the request to USGS with our filters attached.
    # 'timeout=30' means: give up after 30 seconds instead of waiting forever.
    response = requests.get(USGS_URL, params=params, timeout=30)

    # If USGS returned an error status (like 400 or 500), stop and raise it now
    # so we see the problem clearly instead of working with bad data.
    response.raise_for_status()

    # Turn the GeoJSON text USGS sent into a Python dictionary we can use.
    return response.json()


def main():
    """
    Run the smoke test: pull recent quakes and print a quick summary.

    Prints how many earthquakes came back and shows one sample record so we
    can confirm the connection works and see the shape of the data.
    """
    # Let the user know the request is starting (it may take a second or two).
    print("Asking USGS for the last 24 hours of magnitude 2.0+ earthquakes...")

    # Call our function. If USGS is unreachable or errors out, catch it and
    # print a friendly message instead of a scary crash.
    try:
        data = fetch_recent_quakes()
    except requests.RequestException as error:
        print(f"Could not reach USGS: {error}")
        return

    # In GeoJSON, every earthquake is one item in the "features" list.
    # So the length of that list is how many quakes came back.
    quakes = data["features"]
    print(f"Success! USGS returned {len(quakes)} earthquakes.")

    # If we got at least one quake, show the first one as a sample so we can
    # see what the data looks like (magnitude, place, and coordinates).
    if quakes:
        sample = quakes[0]
        props = sample["properties"]              # magnitude, place, time, etc.
        coords = sample["geometry"]["coordinates"]  # [longitude, latitude, depth]

        print("\nSample earthquake:")
        print(f"  Magnitude: {props['mag']}")
        print(f"  Place:     {props['place']}")
        print(f"  Longitude: {coords[0]}")
        print(f"  Latitude:  {coords[1]}")
        print(f"  Depth(km): {coords[2]}")


# This line means: only run main() when we launch THIS file directly
# (e.g. "python src/usgs_test_pull.py"). If another file imports this one
# later, main() will NOT auto-run, which keeps the code reusable.
if __name__ == "__main__":
    main()
