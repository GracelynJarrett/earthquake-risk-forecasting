"""
temporal_split.py — Chronological train / validate / test split (leakage-safe).

Purpose:
    Split the data BY DATE (never shuffled) into three time-ordered blocks:
      - train:    2000 → 2018   (the model learns only from here)
      - validate: 2019 → 2021   (used to tune and pick the ablation winner)
      - test:     2022 → present (the final honest score, looked at once)

    Why by date and not random? Our target asks "will a big quake happen in the
    NEXT 7 days?". A random split would let the model train on future days and be
    tested on past ones — time-travel that leaks the answer. Splitting in strict
    date order makes the test set genuinely "the future" the model never saw.

    The 7-day EMBARGO: because each row's label looks 7 days ahead, the last week
    of one block secretly overlaps the start of the next. We drop the final 7 days
    of train and of validate to seal that gap. (See EMBARGO_DAYS below.)

    This module only LABELS rows by split; it does not impute or scale. Those steps
    happen later, fit on the training block only, to keep the split leakage-free.
"""

# pandas: compare each row's date against the cutoffs and build the split masks.
import pandas as pd


# --- Split boundaries (the single source of truth for the whole project) ------
# These are the calendar cutoffs agreed during Week 3 Day 1. The same dates apply
# to ALL regions (pooled), since it is one model with 'region' as a feature.

# Train covers everything up to and including this date.
TRAIN_END = "2018-12-31"

# Validate runs from the day after train to this end date.
VALIDATE_START = "2019-01-01"
VALIDATE_END = "2021-12-31"

# Test runs from this date to whatever the latest data is (the "future").
TEST_START = "2022-01-01"

# Embargo gap (in days) = the label horizon. We trim this many days off the END of
# the train and validate blocks so a block's 7-day-ahead label cannot peek into the
# next block. It matches the 7-day prediction window in the target definition.
EMBARGO_DAYS = 7


def assign_split(df, date_column="time"):
    """
    Label each row train / validate / test by date, and drop the embargo gaps.

    Args:
        df (pd.DataFrame): any table with a date/time column (the cleaned catalog
            today, or the region-day feature table in later steps).
        date_column (str): name of the column holding the date. Defaults to "time".

    Returns:
        pd.DataFrame: a NEW DataFrame (input untouched) containing only the rows
            that belong to a split, with an added "split" column whose value is
            "train", "validate", or "test". Rows falling in a 7-day embargo gap are
            removed so they can never leak between blocks.
    """
    # Work on a copy so the caller's DataFrame is never modified.
    labelled = df.copy()

    # Read the dates and reduce each to a plain calendar day, so a row is judged
    # purely by its date. Three steps:
    #   - format="ISO8601" (same as clean_data.py) handles USGS timestamps whether
    #     or not they include fractional seconds, which a plain parse trips over;
    #   - utc=True lands everything on one clock (the data is already UTC);
    #   - tz_localize(None) drops the timezone so these compare cleanly against our
    #     plain-date cutoffs; normalize() then snaps each to midnight.
    dates = (
        pd.to_datetime(labelled[date_column], format="ISO8601", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )

    # Turn the string cutoffs into real timestamps we can compare against, and
    # express the embargo as a 7-day span to subtract from each block's end.
    embargo = pd.Timedelta(EMBARGO_DAYS, unit="D")
    train_end = pd.Timestamp(TRAIN_END)
    validate_start = pd.Timestamp(VALIDATE_START)
    validate_end = pd.Timestamp(VALIDATE_END)
    test_start = pd.Timestamp(TEST_START)

    # Build one true/false mask per block. Subtracting the embargo from train_end
    # and validate_end is what carves out the 7-day safety gaps between blocks.
    train_mask = dates <= (train_end - embargo)
    validate_mask = (dates >= validate_start) & (dates <= (validate_end - embargo))
    test_mask = dates >= test_start

    # Start every row as unassigned, then stamp the three blocks onto it. Any row
    # left unassigned sits in an embargo gap (or outside all ranges) and is dropped.
    labelled["split"] = pd.NA
    labelled.loc[train_mask, "split"] = "train"
    labelled.loc[validate_mask, "split"] = "validate"
    labelled.loc[test_mask, "split"] = "test"

    dropped = labelled["split"].isna().sum()
    labelled = labelled[labelled["split"].notna()].reset_index(drop=True)

    # Print a short summary so the split sizes are easy to sanity-check and cite.
    print(f"Total rows in:        {len(df):,}")
    print(f"  Dropped (embargo):  {dropped:,}")
    for name in ["train", "validate", "test"]:
        count = (labelled["split"] == name).sum()
        share = count / len(labelled) if len(labelled) else 0
        print(f"  {name:<9} {count:>8,}  ({share:.1%})")

    return labelled


def main():
    """
    Self-check: run the split on the cleaned catalog and print the block sizes.

    This is a smoke test of the split MECHANICS on real dates — the split we
    actually model on is Day 2's region-day feature table, but the same function
    handles both, so a sensible ~70/15/15 here plus a non-zero embargo drop
    confirms the leakage guardrail works.
    """
    from pathlib import Path

    # Locate the cleaned CSV produced by clean_data.py.
    processed = Path(__file__).parent.parent / "data" / "processed" / "earthquakes_clean.csv"

    print(f"Loading cleaned data from {processed} ...\n")
    df = pd.read_csv(processed)

    print("Assigning temporal split...\n")
    assign_split(df, date_column="time")


# Only run the self-check when this file is launched directly
# (e.g. "python src/temporal_split.py"); importing assign_split elsewhere will not.
if __name__ == "__main__":
    main()
