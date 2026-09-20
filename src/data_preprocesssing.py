import pandas as pd
from pathlib import Path


# ============================================================
# ASTRORISK - DATA PREPROCESSING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "neo_data.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_FILE = PROCESSED_DIR / "clean_neo_data.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("ASTRORISK - DATA PREPROCESSING")
    print("=" * 60)

    print()
    print("Loading NASA/JPL dataset...")

    df = pd.read_csv(RAW_FILE)

    print()
    print("Dataset loaded successfully!")

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):

    print()
    print("=" * 60)
    print("CLEANING DATA")
    print("=" * 60)

    # --------------------------------------------------------
    # Remove duplicate asteroid records
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset="spkid"
    )

    after = len(df)

    print()
    print("Duplicate records removed:", before - after)

    # --------------------------------------------------------
    # Convert Y/N columns to numerical values
    # --------------------------------------------------------

    df["neo"] = df["neo"].map({
        "Y": 1,
        "N": 0
    })

    df["pha"] = df["pha"].map({
        "Y": 1,
        "N": 0
    })

    print()
    print("Converted:")
    print("neo: Y/N -> 1/0")
    print("pha: Y/N -> 1/0")

    # --------------------------------------------------------
    # Handle missing numerical values
    # --------------------------------------------------------

    numerical_columns = [
        "diameter",
        "albedo"
    ]

    print()
    print("Handling missing numerical values...")

    for column in numerical_columns:

        missing_before = df[column].isna().sum()

        if missing_before > 0:

            median_value = df[column].median()

            df[column] = df[column].fillna(
                median_value
            )

            print(
                f"{column}: "
                f"{missing_before} missing values "
                f"filled with median "
                f"({median_value:.4f})"
            )

    # --------------------------------------------------------
    # Handle missing categorical values
    # --------------------------------------------------------

    categorical_columns = [
        "class"
    ]

    for column in categorical_columns:

        missing_before = df[column].isna().sum()

        if missing_before > 0:

            df[column] = df[column].fillna(
                "Unknown"
            )

            print(
                f"{column}: "
                f"{missing_before} missing values "
                f"filled with 'Unknown'"
            )

    # --------------------------------------------------------
    # Convert class into numerical categories
    # --------------------------------------------------------

    df = pd.get_dummies(
        df,
        columns=["class"],
        prefix="class",
        dtype=int
    )

    print()
    print("Converted orbital class into numerical features.")

    return df


# ============================================================
# REMOVE UNNECESSARY COLUMNS
# ============================================================

def remove_unnecessary_columns(df):

    print()
    print("=" * 60)
    print("REMOVING UNNECESSARY COLUMNS")
    print("=" * 60)

    columns_to_remove = [
        "spkid",
        "pdes",
        "full_name"
    ]

    existing_columns = [
        column
        for column in columns_to_remove
        if column in df.columns
    ]

    df = df.drop(
        columns=existing_columns
    )

    print()
    print("Removed columns:")

    for column in existing_columns:
        print("-", column)

    return df


# ============================================================
# CHECK DATA
# ============================================================

def check_data(df):

    print()
    print("=" * 60)
    print("FINAL DATA CHECK")
    print("=" * 60)

    print()
    print("Rows:", len(df))

    print(
        "Columns:",
        len(df.columns)
    )

    print()
    print("Remaining missing values:")

    missing = df.isnull().sum()

    missing = missing[
        missing > 0
    ]

    if len(missing) == 0:

        print("None")

    else:

        print(missing)

    print()
    print("Data types:")

    print(df.dtypes)


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_data(df):

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_FILE,
        index=False
    )

    print()
    print("=" * 60)
    print("PROCESSED DATA SAVED")
    print("=" * 60)

    print()
    print("Location:")

    print(PROCESSED_FILE)


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        # Load raw NASA data
        df = load_data()

        # Clean data
        df = clean_data(df)

        # Remove unnecessary columns
        df = remove_unnecessary_columns(df)

        # Check final dataset
        check_data(df)

        # Save processed dataset
        save_data(df)

        print()
        print("=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)

    except FileNotFoundError:

        print()
        print("ERROR:")
        print("Raw dataset was not found.")

        print()
        print("Expected file:")
        print(RAW_FILE)

    except Exception as error:

        print()
        print("ERROR:")
        print(error)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()