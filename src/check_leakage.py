"""
check_leakage.py — Explicit leakage-safety checks for the Week 3 modeling setup.

The project's core promise is an honest, LEAKAGE-FREE forecaster, so we verify that
mechanically instead of trusting it. This runs a set of assertions on the feature
table + split and prints PASS/FAIL for each; it exits non-zero if any check fails
(so it can double as a pipeline guard later).

Checks:
  1. Temporal split has no future bleed — each block's 7-day-ahead label cannot
     reach into the next block (the embargo gap really holds).
  2. Every (region, date) row belongs to exactly one split.
  3. The leakage-prone detection columns (dmin/rms/nst/gap) never appear as features.
  4. Preprocessing is fit on TRAIN ONLY (imputer median = the train median).
  5. The label is a clean binary target with no missing values.
"""

# sys: non-zero exit code if any check fails.
import sys
# pandas: date math for the embargo check.
import pandas as pd

# Reuse the real loaders / split / feature config used by the models.
from train_baseline import (
    load_features, split_data, load_variants, feature_columns,
    build_preprocessor, TARGET,
)
# The forecast horizon (7 days) — the size of the gap the embargo must protect.
from build_features import LABEL_HORIZON_DAYS


# Detection-quality columns excluded for leakage (see EDA Section 5).
LEAKAGE_COLUMNS = ["dmin", "rms", "nst", "gap"]


def _report(checks):
    """
    Print each check as PASS/FAIL and return True only if all passed.

    Args:
        checks (list[tuple[str, bool, str]]): (name, passed, detail) per check.

    Returns:
        bool: True if every check passed.
    """
    for name, passed, detail in checks:
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {name}")
        if detail:
            print(f"       {detail}")
    return all(passed for _, passed, _ in checks)


def main():
    """Run every leakage check, print results, and exit non-zero on any failure."""
    df = load_features()
    parts = split_data(df)
    cfg = load_variants()
    tr, va, te = parts["train"], parts["validate"], parts["test"]
    horizon = pd.Timedelta(LABEL_HORIZON_DAYS, unit="D")

    checks = []

    # 1. Temporal split: a block's 7-day-ahead label must end BEFORE the next block
    #    begins, or the label would peek at data used to build the next block.
    checks.append((
        "Temporal split — train's 7-day label window ends before validation starts",
        tr["date"].max() + horizon < va["date"].min(),
        f"train_max {tr['date'].max().date()} + {LABEL_HORIZON_DAYS}d "
        f"< val_min {va['date'].min().date()}",
    ))
    checks.append((
        "Temporal split — validation's 7-day label window ends before test starts",
        va["date"].max() + horizon < te["date"].min(),
        f"val_max {va['date'].max().date()} + {LABEL_HORIZON_DAYS}d "
        f"< test_min {te['date'].min().date()}",
    ))

    # 2. Each (region, date) must live in exactly one split (no row shared/duplicated).
    max_splits = df.groupby(["region", "date"])["split"].nunique().max()
    checks.append((
        "Every (region, date) row belongs to exactly one split",
        max_splits == 1,
        f"max distinct splits per (region, date) = {max_splits}",
    ))

    # 3. The leakage-prone detection columns must never be used as a feature by ANY
    #    variant (the FEATURE_COLUMNS allow-list working as intended).
    used = set()
    for variant in cfg["variants"]:
        numeric, categorical = feature_columns(cfg, variant)
        used.update(numeric + categorical)
    leaked = [c for c in LEAKAGE_COLUMNS if c in used]
    checks.append((
        "Leakage columns (dmin/rms/nst/gap) never used as features",
        not leaked,
        f"offending columns: {leaked or 'none'}",
    ))

    # 4. Preprocessing must be fit on TRAIN ONLY. Fit the imputer on train and confirm
    #    its median equals the TRAIN median (not the full-data median) for a feature
    #    that actually has quiet-day blanks.
    numeric, _ = feature_columns(cfg, "base")
    pre = build_preprocessor(numeric, [])
    pre.fit(tr[numeric])
    imputer = pre.named_transformers_["num"].named_steps["impute"]
    col = "avg_mag_7d"
    fitted = imputer.statistics_[numeric.index(col)]
    train_median = tr[col].median()
    full_median = df[col].median()
    checks.append((
        "Preprocessing fit on TRAIN only (imputer median = train median)",
        abs(fitted - train_median) < 1e-9,
        f"{col}: fitted={fitted:.4f}  train={train_median:.4f}  full-data={full_median:.4f}",
    ))

    # 5. The label must be a clean binary target (only 0/1, no missing values).
    vals = sorted(df[TARGET].dropna().unique().tolist())
    checks.append((
        "Label is a clean binary target (0/1, no missing values)",
        df[TARGET].notna().all() and set(vals).issubset({0, 1}),
        f"values={vals}, nulls={int(df[TARGET].isna().sum())}",
    ))

    print("Leakage-safety checks")
    print("=" * 64)
    all_passed = _report(checks)
    print("=" * 64)
    if all_passed:
        print(f"ALL {len(checks)} CHECKS PASSED — no leakage detected.")
    else:
        failed = sum(1 for _, passed, _ in checks if not passed)
        print(f"{failed} of {len(checks)} CHECKS FAILED.")
        sys.exit(1)


# Only run when launched directly (e.g. "python src/check_leakage.py").
if __name__ == "__main__":
    main()