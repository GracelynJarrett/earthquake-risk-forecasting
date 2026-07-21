"""
build_features.py — Turn the per-earthquake catalog into a per-(region, day)
feature table for modeling (leakage-safe).

Each row is one region on one calendar day. Features are computed looking BACKWARD
from that day (only data available by then); the label (added later) looks FORWARD
7 days. This file builds that table step by step; piece 1 lays down the skeleton:
load the cleaned data and create the full (region x day) grid, including quiet days.
"""

# pandas + numpy: build and reshape the region-day table.
import pandas as pd
import numpy as np
# Path: OS-independent file paths.
from pathlib import Path
# sqlite3: write the finished table into the database (same approach as load_to_sqlite.py).
import sqlite3
# Reuse the exact split logic from Day 1 so features and modeling never disagree.
from temporal_split import assign_split


# Input: the cleaned catalog produced by clean_data.py.
CLEAN_CSV = Path(__file__).parent.parent / "data" / "processed" / "earthquakes_clean.csv"

# The three regions we model (same names/order as the rest of the pipeline).
REGIONS = ["california", "japan", "greece"]

# Region-significant thresholds — define which quakes count as "large".
THRESHOLDS = {"california": 4.5, "greece": 5.0, "japan": 5.5}

# How many days ahead the forecast looks (matches the target definition + embargo).
LABEL_HORIZON_DAYS = 7

# Output database + table name (writes alongside the existing 'earthquakes' table).
DB_PATH = Path(__file__).parent.parent / "data" / "earthquakes.db"
FEATURES_TABLE = "features"


def load_clean_data(path=CLEAN_CSV):
    """
    Load the cleaned catalog and add a plain calendar 'date' column.

    Args:
        path (Path): location of the cleaned CSV. Defaults to CLEAN_CSV.

    Returns:
        pd.DataFrame: the catalog with 'time' as timezone-naive timestamps and a
                      new 'date' column (midnight of each quake's day).
    """
    df = pd.read_csv(path)

    # Parse timestamps the same leakage-safe way as temporal_split.py: read as UTC
    # (handles the mixed fractional-seconds format), then drop the timezone so the
    # dates compare cleanly against everything else.
    df["time"] = pd.to_datetime(df["time"], format="ISO8601", utc=True).dt.tz_localize(None)

    # A day-level stamp (midnight) — the grain our feature table is built on.
    df["date"] = df["time"].dt.normalize()
    return df


def build_daily_grid(df):
    """
    Build one row per (region, calendar day) across the whole record.

    Args:
        df (pd.DataFrame): the loaded catalog (needs the 'date' column).

    Returns:
        pd.DataFrame: columns ['region', 'date'] — every region paired with every
                      day, including quiet days with no earthquakes.
    """
    # One shared calendar for all regions (pooled), from the first to the last day
    # in the data, so every region is evaluated on the same timeline as our split.
    all_days = pd.date_range(df["date"].min(), df["date"].max(), freq="D")

    # Cross every region with every day → the empty skeleton features will fill.
    grid = pd.MultiIndex.from_product(
        [REGIONS, all_days], names=["region", "date"]
    ).to_frame(index=False)
    return grid


def compute_window_features(df, grid):
    """
    For every (region, day), compute the backward-looking recent-activity features.

    Uses the "daily tally -> rolling window" approach: collapse quakes to one row
    per day, then slide trailing 7- and 30-day windows over those daily tallies.
    All windows include the current day and look only backward (leakage-safe).

    Args:
        df (pd.DataFrame): loaded catalog (needs 'region', 'date', 'magnitude',
                           'depth_km', 'distance_to_fault_km', 'latitude', 'longitude').
        grid (pd.DataFrame): the (region, date) skeleton from build_daily_grid().

    Returns:
        pd.DataFrame: one row per (region, date) with the recent-activity features.
                      Average features are blank (NaN) on stretches with no quakes;
                      those quiet-day gaps are imputed later, at modeling time.
    """
    # One shared daily calendar for every region (matches the grid's span).
    days = pd.date_range(grid["date"].min(), grid["date"].max(), freq="D")

    results = []
    for region in REGIONS:
        reg = df[df["region"] == region].copy()

        # Flag each quake as "large" for this region (at/above its threshold).
        reg["is_large"] = (reg["magnitude"] >= THRESHOLDS[region]).astype(int)

        # --- Daily tally: one row per day with the raw quantities we'll roll up.
        # Quiet days become NaN here, then every calendar day is put back via reindex.
        daily = reg.groupby("date").agg(
            n=("magnitude", "size"),          # how many quakes that day
            mag_sum=("magnitude", "sum"),     # sum of magnitudes (for averages)
            mag_max=("magnitude", "max"),     # biggest that day (for rolling max)
            large=("is_large", "sum"),        # how many were "large"
            depth_sum=("depth_km", "sum"),
            dist_sum=("distance_to_fault_km", "sum"),
            lat_sum=("latitude", "sum"),
            lon_sum=("longitude", "sum"),
        ).reindex(days)

        # Rolling counts of quakes (0 on quiet days). These also form the
        # denominator for the average features below.
        n7 = daily["n"].rolling(7, min_periods=1).sum().fillna(0)
        n30 = daily["n"].rolling(30, min_periods=1).sum().fillna(0)

        feat = pd.DataFrame(index=days)
        # Counts of recent activity.
        feat["quakes_7d"] = n7
        feat["quakes_30d"] = n30
        feat["large_30d"] = daily["large"].rolling(30, min_periods=1).sum().fillna(0)
        # Long-horizon "large quake" counts — a data-driven, TIME-VARYING base rate per
        # region: how many region-significant quakes in the trailing ~1 / 3 / 5 years,
        # plus an all-time running total (large quakes since records began). Same
        # backward-looking rolling pattern as large_30d, so leakage-safe; 0 on stretches
        # with no large quakes yet. Aimed at Day-1's finding that the model leaned on
        # 'region' only as a frozen base-rate proxy — these move as a region's activity does.
        feat["large_365d"] = daily["large"].rolling(365, min_periods=1).sum().fillna(0)
        feat["large_1095d"] = daily["large"].rolling(1095, min_periods=1).sum().fillna(0)
        feat["large_1825d"] = daily["large"].rolling(1825, min_periods=1).sum().fillna(0)
        feat["large_alltime"] = daily["large"].expanding(min_periods=1).sum().fillna(0)
        # Averages = rolling sum of the quantity / rolling count of quakes (0/0 -> NaN).
        feat["avg_mag_7d"] = daily["mag_sum"].rolling(7, min_periods=1).sum() / n7.replace(0, np.nan)
        feat["avg_mag_30d"] = daily["mag_sum"].rolling(30, min_periods=1).sum() / n30.replace(0, np.nan)
        feat["max_mag_30d"] = daily["mag_max"].rolling(30, min_periods=1).max()
        feat["avg_depth_30d"] = daily["depth_sum"].rolling(30, min_periods=1).sum() / n30.replace(0, np.nan)
        feat["avg_dist_30d"] = daily["dist_sum"].rolling(30, min_periods=1).sum() / n30.replace(0, np.nan)
        # Recent-activity centroid (plain mean is fine — none of our regions cross ±180°).
        feat["centroid_lat_30d"] = daily["lat_sum"].rolling(30, min_periods=1).sum() / n30.replace(0, np.nan)
        feat["centroid_lon_30d"] = daily["lon_sum"].rolling(30, min_periods=1).sum() / n30.replace(0, np.nan)
        # Activity trend: this week's count minus the previous week's count.
        feat["trend_7d"] = n7 - n7.shift(7)

        # Days since the most recent quake: stamp each quake day with its own date,
        # carry that forward across quiet days, then subtract. 0 on a quake day;
        # grows through a quiet stretch; blank before the region's first quake.
        day_of = pd.Series(days, index=days)
        last_quake_day = day_of.where(daily["n"].fillna(0) > 0).ffill()
        feat["days_since_last"] = (day_of - last_quake_day).dt.days

        feat["region"] = region
        results.append(feat.reset_index(names="date"))

    return pd.concat(results, ignore_index=True)


def compute_label(df, grid):
    """
    Forward-looking target: 1 if a region-significant quake occurs in the next
    7 days (days D+1..D+7), else 0. Blank (NA) for the final 7 days, whose future
    is not yet observable — those rows are dropped when the table is assembled.

    Args:
        df (pd.DataFrame): loaded catalog (needs 'region', 'date', 'magnitude').
        grid (pd.DataFrame): the (region, date) skeleton from build_daily_grid().

    Returns:
        pd.DataFrame: columns ['region', 'date', 'label_7d'] (label_7d is 0/1/NA).
    """
    days = pd.date_range(grid["date"].min(), grid["date"].max(), freq="D")

    results = []
    for region in REGIONS:
        reg = df[df["region"] == region]

        # Count this region's SIGNIFICANT quakes per day (0 on days with none).
        large = reg[reg["magnitude"] >= THRESHOLDS[region]].groupby("date").size()
        large_daily = large.reindex(days).fillna(0)

        # Trailing 7-day sum shifted back 7 days => sum over D+1..D+7. A trailing
        # window at day D+7 covers D+1..D+7, so shift(-7) reads that value at day D.
        future_7 = (large_daily.rolling(LABEL_HORIZON_DAYS, min_periods=1).sum()
                    .shift(-LABEL_HORIZON_DAYS))

        # 1 if any significant quake in the window, else 0; NA where the future
        # isn't observable yet (the last 7 days).
        label = pd.Series(pd.NA, index=days, dtype="Int64")
        seen = future_7.notna()
        label[seen] = (future_7[seen] >= 1).astype(int)

        results.append(pd.DataFrame({"region": region, "date": days, "label_7d": label}))

    return pd.concat(results, ignore_index=True)


def build_feature_table(df=None):
    """
    Assemble the full modeling-ready table: features + label + split tag.

    Args:
        df (pd.DataFrame | None): loaded catalog; loaded from disk if not provided.

    Returns:
        pd.DataFrame: one row per (region, day) with all features, 'label_7d', and
                      a 'split' column. The unobservable-label tail and the 7-day
                      embargo gaps are already removed.
    """
    if df is None:
        df = load_clean_data()

    grid = build_daily_grid(df)
    feats = compute_window_features(df, grid)
    labels = compute_label(df, grid)

    # Attach each row's label, then drop the final 7 days (label not yet knowable).
    table = feats.merge(labels, on=["region", "date"])
    table = table[table["label_7d"].notna()].reset_index(drop=True)

    # Tag train/validate/test (this also drops the 7-day embargo gaps between blocks).
    table = assign_split(table, date_column="date")
    return table


def save_to_sqlite(table, db_path=DB_PATH, table_name=FEATURES_TABLE):
    """
    Write the feature table into SQLite, replacing any prior copy, and index it.

    Args:
        table (pd.DataFrame): the assembled feature table.
        db_path (Path): the database file. Defaults to data/earthquakes.db.
        table_name (str): destination table name. Defaults to 'features'.
    """
    with sqlite3.connect(db_path) as conn:
        # Rebuild the table from scratch each run (index=False drops pandas' row ids).
        table.to_sql(table_name, conn, if_exists="replace", index=False)
        # Index on (region, date) for fast lookups, mirroring load_to_sqlite.py.
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_features_region_date ON {table_name} (region, date)"
        )


def main():
    """Build the full feature table and save it to the database."""
    table = build_feature_table()
    save_to_sqlite(table)

    print(f"\nSaved {len(table):,} rows x {table.shape[1]} cols "
          f"to {DB_PATH.name} (table '{FEATURES_TABLE}')")
    print("Columns:", list(table.columns))
    # Positive rate per region as a final leakage-safe sanity check.
    print("\nPositive rate by region:")
    for region in REGIONS:
        lab = table[table["region"] == region]["label_7d"]
        print(f"  {region:<11} {lab.mean():.1%}  (n={len(lab):,})")


# Only run the smoke test when launched directly (e.g. "python src/build_features.py").
if __name__ == "__main__":
    main()