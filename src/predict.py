import pandas as pd
import joblib


RAW_DATA_PATH = "data/raw/neo_data.csv"
PROCESSED_DATA_PATH = "data/processed/clean_neo_data.csv"
MODEL_PATH = "models/astrorisk_rf.pkl"


# ---------------------------------
# 1. Load data and trained model
# ---------------------------------

raw_df = pd.read_csv(RAW_DATA_PATH)
processed_df = pd.read_csv(PROCESSED_DATA_PATH)

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
features = model_data["features"]


# ---------------------------------
# 2. Find asteroid
# ---------------------------------

def find_asteroid(search_text):

    search_text = str(search_text).strip().lower()

    matches = raw_df[
        raw_df["pdes"].astype(str).str.lower().str.contains(
            search_text,
            na=False
        )
        |
        raw_df["spkid"].astype(str).str.lower().str.contains(
            search_text,
            na=False
        )
        |
        raw_df["full_name"].astype(str).str.lower().str.contains(
            search_text,
            na=False
        )
    ]

    return matches


# ---------------------------------
# 3. Analyze asteroid
# ---------------------------------

def analyze_asteroid(asteroid):

    # Select only the features used by the ML model
    input_data = asteroid[features].to_frame().T

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    return prediction, probability


# ---------------------------------
# 4. Command-line interface
# ---------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 50)
    print("             ASTRORISK")
    print("       ASTEROID HAZARD ANALYZER")
    print("=" * 50)

    search = input(
        "\nEnter asteroid designation/name: "
    ).strip()

    matches = find_asteroid(search)

    if matches.empty:

        print("\nNo asteroid found.")

    else:

        print(
            f"\nFound {len(matches)} matching asteroid(s).\n"
        )

        # Show up to 10 results
        displayed_matches = matches.head(10)

        for index, (_, asteroid) in enumerate(
            displayed_matches.iterrows(),
            start=1
        ):

            print(
                f"{index}. "
                f"{asteroid['full_name']} "
                f"(SPK-ID: {asteroid['spkid']})"
            )

        # ---------------------------------
        # Select asteroid
        # ---------------------------------

        if len(displayed_matches) > 1:

            choice = input(
                "\nEnter the number to analyze "
                "(press Enter for first result): "
            ).strip()

            if choice.isdigit():

                choice = int(choice)

                if 1 <= choice <= len(displayed_matches):

                    asteroid = displayed_matches.iloc[choice - 1]

                else:

                    print("Invalid selection. Using first result.")

                    asteroid = displayed_matches.iloc[0]

            else:

                asteroid = displayed_matches.iloc[0]

        else:

            asteroid = displayed_matches.iloc[0]


        # ---------------------------------
        # Run model
        # ---------------------------------

        prediction, probability = analyze_asteroid(
            asteroid
        )


        # ---------------------------------
        # Display result
        # ---------------------------------

        print("\n" + "=" * 50)
        print("           ASTRORISK ANALYSIS")
        print("=" * 50)

        print(
            f"\nAsteroid: {asteroid['full_name']}"
        )

        print(
            f"Designation: {asteroid['pdes']}"
        )

        print(
            f"SPK-ID: {asteroid['spkid']}"
        )

        print(
            f"\nPHA probability: "
            f"{probability * 100:.2f}%"
        )

        if prediction == 1:

            print("Classification: PHA")

        else:

            print("Classification: Non-PHA")


        print("\nOrbital parameters:")

        print(f"  Eccentricity (e): {asteroid['e']}")
        print(f"  Semi-major axis (a): {asteroid['a']} AU")
        print(f"  Perihelion distance (q): {asteroid['q']} AU")
        print(f"  Inclination (i): {asteroid['i']}°")
        print(f"  Orbital period: {asteroid['per']} days")


        print("\nObservation information:")

        print(
            f"  Data arc: {asteroid['data_arc']} days"
        )

        print(
            f"  Observations: {asteroid['n_obs_used']}"
        )

        print(
            f"  Condition code: "
            f"{asteroid['condition_code']}"
        )


        print("\nNote:")
        print(
            "This is an ML classification of PHA status."
        )
        print(
            "It is NOT an asteroid impact probability."
        )

        print("=" * 50)