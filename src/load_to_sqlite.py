"""
load_to_sqlite.py — Load cleaned earthquake data into a SQLite database.

Purpose:
    Read the cleaned dataset (data/processed/earthquakes_clean.csv, produced by
    src/clean_data.py) and store it in a SQLite database at data/earthquakes.db,
    in a table called 'earthquakes'. Also builds an index on (region, time) so
    the model and the future Airflow pipeline can query it quickly.

    SQLite is built into Python (the 'sqlite3' module), so there is nothing extra
    to install. The database is the single source of truth later steps read from.

    Run order:
        1. python src/fetch_usgs.py     (pull raw data)
        2. python src/clean_data.py      (clean -> earthquakes_clean.csv)
        3. python src/load_to_sqlite.py  (this script -> earthquakes.db)
"""

# sqlite3: Python's built-in library for talking to a SQLite database file.
import sqlite3

# pandas: reads the cleaned CSV and can write a whole table to SQLite in one call.
import pandas as pd

# Path: build file paths that work on any operating system.
from pathlib import Path


# Input (the cleaned CSV) and output (the database file).
PROCESSED_CSV = Path(__file__).parent.parent / "data" / "processed" / "earthquakes_clean.csv"
DATABASE_PATH = Path(__file__).parent.parent / "data" / "earthquakes.db"

# The table we create inside the database, plus the columns we index for speed.
TABLE_NAME = "earthquakes"
INDEX_COLUMNS = ["region", "time"]


def load_clean_data():
    """
    Read the cleaned earthquake CSV into a DataFrame.

    Returns:
        pd.DataFrame: the cleaned data.

    Raises:
        FileNotFoundError: if the cleaned CSV is missing (run clean_data.py first).
    """
    # Stop early with a clear message if the cleaning step hasn't been run yet.
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(
            f"Missing cleaned data: {PROCESSED_CSV}. Run clean_data.py first."
        )

    # Read the cleaned CSV. We intentionally leave 'time' as text: SQLite has no
    # native date type, and ISO8601 text sorts correctly by date anyway.
    return pd.read_csv(PROCESSED_CSV)


def build_database(df):
    """
    Write the data into the SQLite database and add an index for fast queries.

    Args:
        df (pd.DataFrame): the cleaned data to store.
    """
    # Make sure the data/ folder exists before creating the database file in it.
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Open a connection to the database file (SQLite creates the file if needed).
    # The 'with' block guarantees the connection is closed properly at the end.
    with sqlite3.connect(DATABASE_PATH) as conn:
        # Write the whole DataFrame into the 'earthquakes' table.
        # if_exists="replace" rebuilds the table from scratch each run, so
        # re-running the pipeline never piles up duplicate rows.
        # index=False leaves out pandas' automatic row numbers.
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

        # Add an index on (region, time). An index works like the index in a book:
        # it lets the database jump to the rows for a region/date range without
        # scanning the whole table — important once the pipeline queries this a lot.
        index_cols = ", ".join(INDEX_COLUMNS)
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_region_time ON {TABLE_NAME} ({index_cols})"
        )


def verify():
    """
    Read summary counts back FROM the database to confirm the load worked.

    Returns:
        int: the total number of rows stored in the earthquakes table.
    """
    # Query the database itself (not the CSV) so we're proving the data really
    # landed in SQLite and is queryable.
    with sqlite3.connect(DATABASE_PATH) as conn:
        # Total rows now stored in the table.
        total = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]

        # Rows per region, as a quick sanity check against our known counts.
        per_region = conn.execute(
            f"SELECT region, COUNT(*) FROM {TABLE_NAME} GROUP BY region"
        ).fetchall()

    print(f"Database now holds {total:,} rows in '{TABLE_NAME}'.")
    for region, count in per_region:
        print(f"  {region}: {count:,}")

    return total


def main():
    """
    Run the full load: read the cleaned CSV, build the database, and verify it.
    """
    print("Loading cleaned data...")
    df = load_clean_data()

    print(f"Writing {len(df):,} rows to {DATABASE_PATH.name}...")
    build_database(df)

    print("Verifying...")
    verify()
    print("\nDone.")


# Only run main() when this file is launched directly (e.g.
# "python src/load_to_sqlite.py"). Importing its functions elsewhere won't auto-run.
if __name__ == "__main__":
    main()
