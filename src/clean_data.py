"""
clean_data.py — Data cleaning pipeline.

Purpose:
    Take the raw USGS earthquake CSVs (produced by src/fetch_usgs.py and stored
    in data/raw/) and turn them into one cleaned, consistent dataset saved in
    data/processed/.

    Cleaning steps (decided during Week 2 EDA):
      - Keep only real earthquakes (drop quarry blasts, explosions, etc.)
      - Normalize the 'status' text so capitalization duplicates merge
      - Keep automatic events and deep/negative depths (all valid data)

    IMPORTANT: Besides cleaning, this script adds ONE leakage-safe feature
    (distance_to_fault_km) — safe because it depends only on fixed locations and a
    fixed boundary map, not on the dataset. It still does NOT impute missing values,
    scale, or split the data; those steps happen at modeling time (Week 3), fit on
    the training set only, to avoid data leakage.
"""

# pandas: load, filter, and save the tabular earthquake data.
import pandas as pd

# Path: build file paths that work on any operating system.
from pathlib import Path

# json + shapely: read the plate-boundary map and compute distance-to-boundary.
import json
import shapely
from shapely.geometry import shape


# Where the raw CSVs come from (input) and where cleaned data goes (output).
RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# Fixed reference map of tectonic plate boundaries (committed to the repo).
PLATE_BOUNDARIES = Path(__file__).parent.parent / "data" / "reference" / "plate_boundaries.geojson"

# The three regions to clean (each has its own raw CSV in data/raw/).
REGIONS = ["california", "japan", "greece"]

# --- Column roles: our safeguard against data leakage -------------------------
# We KEEP every column in storage, but clearly label each column's role. Week-3
# modeling will ONLY ever read from FEATURE_COLUMNS, so the leakage-prone
# detection metrics can be stored safely without any risk of reaching the model.

# Columns allowed to become model features (depth is a candidate we'll test;
# distance_to_fault_km is the leakage-safe geospatial feature added below).
FEATURE_COLUMNS = ["region", "time", "magnitude", "depth_km", "longitude", "latitude",
                   "distance_to_fault_km"]

# Detection-quality metrics: kept in storage but EXCLUDED from modeling because
# they leak information about magnitude (see EDA Section 5 — regional confound
# for dmin/rms, and target leakage for nst).
EXCLUDED_COLUMNS = ["nst", "gap", "dmin", "rms"]

# Identifiers and descriptive text: kept for reference, never used as features.
METADATA_COLUMNS = ["id", "place", "mag_type", "type", "status"]


def load_raw_data():
    """
    Read the raw region CSVs and combine them into one DataFrame.

    Returns:
        pd.DataFrame: all three regions stacked together, with 'time' parsed
                      into real timestamps.

    Raises:
        FileNotFoundError: if a region's raw CSV is missing (run fetch_usgs.py first).
    """
    # Read each region's raw CSV into its own DataFrame.
    frames = []
    for region in REGIONS:
        csv_path = RAW_DATA_DIR / f"{region}.csv"

        # Fail clearly if the raw data hasn't been pulled yet, instead of a
        # confusing error later on.
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing raw data: {csv_path}. Run fetch_usgs.py first."
            )

        frames.append(pd.read_csv(csv_path))

    # Stack all regions into one table (each row keeps its own 'region' value).
    df = pd.concat(frames, ignore_index=True)

    # Convert 'time' from text into real timestamps so later steps can sort and
    # compare by date. format='ISO8601' matches how USGS writes its timestamps.
    df["time"] = pd.to_datetime(df["time"], format="ISO8601")

    return df


def clean(df):
    """
    Apply the cleaning steps decided during EDA and report what changed.

    Args:
        df (pd.DataFrame): the combined raw data from load_raw_data().

    Returns:
        pd.DataFrame: the cleaned data (a new DataFrame; the input is unchanged).
    """
    # Work on a copy so the caller's original DataFrame is never modified.
    cleaned = df.copy()
    start_rows = len(cleaned)

    # 1) Keep only real earthquakes. USGS also records quarry blasts, explosions,
    #    and nuclear tests, which are not tectonic events and would be noise for
    #    an earthquake forecaster.
    cleaned = cleaned[cleaned["type"] == "earthquake"]
    removed = start_rows - len(cleaned)

    # 2) Normalize 'status' to lowercase so capitalization duplicates merge
    #    (e.g. "REVIEWED" and "reviewed" become the single category "reviewed").
    cleaned["status"] = cleaned["status"].str.lower()

    # Renumber the rows 0..N-1 after dropping some, so the index stays tidy.
    cleaned = cleaned.reset_index(drop=True)

    # Print a short before/after summary so the numbers are easy to cite in the
    # data-understanding report.
    print(f"Rows before cleaning: {start_rows:,}")
    print(f"  Removed non-earthquake events: {removed:,}")
    print(f"Rows after cleaning:  {len(cleaned):,}")
    print(f"  status categories now: {sorted(cleaned['status'].unique())}")

    return cleaned


def add_distance_to_fault(df):
    """
    Feature enrichment (LEAKAGE-SAFE): add 'distance_to_fault_km' — the approximate
    distance from each earthquake to the nearest tectonic plate boundary.

    Safe to compute here because it depends only on a quake's fixed location and a
    fixed reference map of plate boundaries. It does not "learn" anything from the
    dataset, so (unlike imputation or scaling) it cannot leak information across the
    train/validate/test splits.

    Args:
        df (pd.DataFrame): the cleaned data (needs 'longitude' and 'latitude').

    Returns:
        pd.DataFrame: the data with a new 'distance_to_fault_km' column.
    """
    enriched = df.copy()

    # Load the fixed map of plate boundaries (a set of line shapes) from GeoJSON.
    with open(PLATE_BOUNDARIES, encoding="utf-8") as f:
        boundaries_geojson = json.load(f)
    boundary_lines = [shape(feature["geometry"]) for feature in boundaries_geojson["features"]]

    # Put the boundary lines in a spatial index (STRtree) so we can find each
    # quake's nearest boundary quickly, instead of checking every line every time.
    tree = shapely.STRtree(boundary_lines)

    # Turn each quake's (longitude, latitude) into a point shape.
    quake_points = shapely.points(enriched["longitude"].values, enriched["latitude"].values)

    # Find the nearest boundary for every quake and the distance to it. With
    # all_matches=False we get exactly one distance per quake, in row order. The
    # distance is in degrees, which we convert to approximate km (1 degree ~ 111 km).
    _, distance_degrees = tree.query_nearest(
        quake_points, return_distance=True, all_matches=False
    )
    enriched["distance_to_fault_km"] = distance_degrees * 111.0

    return enriched


def save_clean_data(df):
    """
    Save the cleaned data to data/processed/earthquakes_clean.csv.

    Args:
        df (pd.DataFrame): the cleaned data from clean().

    Returns:
        Path: the location of the CSV file that was written.
    """
    # Make sure the output folder exists before writing into it.
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save all cleaned regions to one CSV. index=False leaves out pandas' row numbers.
    output_path = PROCESSED_DATA_DIR / "earthquakes_clean.csv"
    df.to_csv(output_path, index=False)

    return output_path


def main():
    """
    Run the full cleaning pipeline: load raw data, clean it, and save the result.
    """
    print("Loading raw data...")
    raw = load_raw_data()

    print("Cleaning...\n")
    cleaned = clean(raw)

    print("\nAdding leakage-safe distance-to-fault feature...")
    cleaned = add_distance_to_fault(cleaned)

    output_path = save_clean_data(cleaned)
    print(f"\nSaved cleaned data to {output_path}")


# Only run main() when this file is launched directly (e.g.
# "python src/clean_data.py"). Importing its functions elsewhere (like the
# future Airflow pipeline) will NOT auto-run it.
if __name__ == "__main__":
    main()
