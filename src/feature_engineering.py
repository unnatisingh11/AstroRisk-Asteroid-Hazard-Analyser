import pandas as pd
from pathlib import Path


# ============================================================
# ASTRORISK - FEATURE ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clean_neo_data.csv"
)


def load_data():

    print("=" * 60)
    print("ASTRORISK - FEATURE ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(
        PROCESSED_FILE
    )

    print()
    print("Dataset loaded successfully!")

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        len(df.columns)
    )

    return df


def analyze_target(df):

    print()
    print("=" * 60)
    print("PHA TARGET ANALYSIS")
    print("=" * 60)

    print()

    print(
        "PHA value counts:"
    )

    print(
        df["pha"].value_counts()
    )

    print()

    print(
        "PHA percentages:"
    )

    print(
        df["pha"]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )


def analyze_features(df):

    print()
    print("=" * 60)
    print("NUMERICAL FEATURE SUMMARY")
    print("=" * 60)

    numerical_columns = [
        "moid",
        "e",
        "a",
        "q",
        "i",
        "per",
        "H",
        "diameter",
        "albedo",
        "condition_code",
        "data_arc",
        "n_obs_used"
    ]

    existing_columns = [
        column
        for column in numerical_columns
        if column in df.columns
    ]

    print()

    print(
        df[existing_columns].describe().round(3)
    )


def analyze_correlations(df):

    print()
    print("=" * 60)
    print("CORRELATION WITH PHA")
    print("=" * 60)

    numerical_df = df.select_dtypes(
        include="number"
    )

    correlations = (
        numerical_df.corr()["pha"]
        .sort_values(
            ascending=False
        )
    )

    print()

    print(
        correlations.round(3)
    )


def main():

    try:

        df = load_data()

        analyze_target(df)

        analyze_features(df)

        analyze_correlations(df)

        print()
        print("=" * 60)
        print("FEATURE ANALYSIS COMPLETE")
        print("=" * 60)

    except FileNotFoundError:

        print()
        print("ERROR:")
        print(
            "Processed dataset was not found."
        )

        print()
        print(
            "Expected:"
        )

        print(
            PROCESSED_FILE
        )

    except Exception as error:

        print()
        print("ERROR:")
        print(error)


if __name__ == "__main__":
    main()