"""Load and clean the PaySim synthetic transaction dataset.

``archive/paysim.csv`` columns: step, type, amount, nameOrig, oldbalanceOrg,
newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud,
isFlaggedFraud.

IMPORTANT: ``isFraud`` and ``isFlaggedFraud`` are NOT ring-membership ground
truth. ``isFraud`` flags individual fraudulent transactions (unrelated to
ring membership); ``isFlaggedFraud`` is a near-constant Kaggle artifact (16
positives out of 6.36M rows). Ring ground truth is manufactured separately in
``plant_rings.py``. Both columns are kept in the loaded frame for reference
only -- do not use either as a label downstream.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "archive" / "paysim.csv"

EXPECTED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def load_paysim(
    path: str | Path = DEFAULT_PATH,
    nrows: int | None = None,
    sample_frac: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load, validate, and lightly clean the PaySim CSV.

    Parameters
    ----------
    path: CSV location. Defaults to ``archive/paysim.csv`` at the repo root.
    nrows: read only the first N rows (fast dev iteration, cheap since PaySim
        is already ordered by ``step``).
    sample_frac: instead (or in addition to) ``nrows``, take a random sample
        of the loaded rows. Useful because the full file is 6.36M rows --
        too large for a snappy dev loop when building the graph downstream.
    random_state: seed for ``sample_frac``.
    """
    path = Path(path)
    df = pd.read_csv(path, nrows=nrows)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"paysim csv missing expected columns: {sorted(missing)}")

    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

    # Tighter dtypes -- roughly halves memory on the full 6.3M-row file.
    df["type"] = df["type"].astype("category")
    df["step"] = df["step"].astype("int32")
    for col in ("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"):
        df[col] = df[col].astype("float32")
    for col in ("isFraud", "isFlaggedFraud"):
        df[col] = df[col].astype("int8")

    # Entity kind from PaySim's naming convention: "C" = customer/personal
    # account, "M" = merchant. nameOrig is always "C" in this dataset;
    # nameDest is "C" or "M".
    df["orig_kind"] = df["nameOrig"].str[0].astype("category")
    df["dest_kind"] = df["nameDest"].str[0].astype("category")

    return df


def summarize(df: pd.DataFrame) -> None:
    """Print a quick sanity report -- used to eyeball load_paysim's output."""
    print(f"rows: {len(df):,}")
    print(f"columns: {list(df.columns)}")
    print(f"unique nameOrig: {df['nameOrig'].nunique():,}  unique nameDest: {df['nameDest'].nunique():,}")
    print(f"step range: {df['step'].min()}-{df['step'].max()}")
    print("\ntype counts:")
    print(df["type"].value_counts())
    print("\ndest_kind counts (C=customer, M=merchant):")
    print(df["dest_kind"].value_counts())
    print(f"\nisFraud=1: {df['isFraud'].sum():,} ({df['isFraud'].mean():.4%} of rows)")
    print(f"isFlaggedFraud=1: {df['isFlaggedFraud'].sum():,}")
    print("\nhead:")
    print(df.head())
    print("\ndtypes:")
    print(df.dtypes)


if __name__ == "__main__":
    frame = load_paysim()
    summarize(frame)
