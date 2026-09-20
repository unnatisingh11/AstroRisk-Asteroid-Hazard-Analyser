import requests
import pandas as pd
from pathlib import Path
import time


# ============================================================
# ASTRORISK - NASA/JPL NEO DATA DOWNLOADER
# ============================================================

API_URL = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_FILE = RAW_DIR / "neo_data.csv"


# ============================================================
# SETTINGS
# ============================================================

TARGET_RECORDS = 3500

BATCH_SIZE = 100

MAX_RETRIES = 5

RETRY_DELAY = 15


# ============================================================
# NASA/JPL FIELDS
# ============================================================

FIELDS = (
    "spkid,"
    "pdes,"
    "full_name,"
    "neo,"
    "pha,"
    "class,"
    "moid,"
    "e,"
    "a,"
    "q,"
    "i,"
    "per,"
    "H,"
    "diameter,"
    "albedo,"
    "condition_code,"
    "data_arc,"
    "n_obs_used"
)


# ============================================================
# FETCH ONE BATCH
# ============================================================

def fetch_batch(start, limit):

    params = {
        "fields": FIELDS,
        "sb-kind": "a",
        "sb-group": "neo",
        "limit": str(limit),
        "limit-from": str(start),
        "full-prec": "true"
    }

    headers = {
        "User-Agent": "AstroRisk/1.0"
    }

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            print(
                f"Request: records {start + 1}-{start + limit} "
                f"| Attempt {attempt}/{MAX_RETRIES}"
            )

            response = requests.get(
                API_URL,
                params=params,
                headers=headers,
                timeout=60
            )

            print(
                f"Status: {response.status_code}"
            )

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if response.status_code == 200:

                data = response.json()

                if "error" in data:

                    print(
                        "NASA/JPL API error:",
                        data["error"]
                    )

                    return None

                fields = data.get("fields")

                records = data.get("data")

                if not records:

                    print(
                        "No records returned."
                    )

                    return None

                return pd.DataFrame(
                    records,
                    columns=fields
                )

            # ------------------------------------------------
            # TEMPORARY SERVER ERROR
            # ------------------------------------------------

            if response.status_code in [500, 502, 503, 504]:

                print(
                    "NASA/JPL server temporarily unavailable."
                )

                if attempt < MAX_RETRIES:

                    print(
                        f"Waiting {RETRY_DELAY} seconds..."
                    )

                    time.sleep(RETRY_DELAY)

                    continue

                print(
                    "Maximum retries reached for this batch."
                )

                return None

            # ------------------------------------------------
            # OTHER ERROR
            # ------------------------------------------------

            print(
                "Unexpected HTTP response:"
            )

            print(
                response.text[:500]
            )

            return None

        except requests.exceptions.Timeout:

            print(
                "Request timed out."
            )

            if attempt < MAX_RETRIES:

                time.sleep(RETRY_DELAY)

        except requests.exceptions.ConnectionError:

            print(
                "Connection error."
            )

            if attempt < MAX_RETRIES:

                time.sleep(RETRY_DELAY)

    return None


# ============================================================
# SAVE DATA
# ============================================================

def save_dataset(df):

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


# ============================================================
# MAIN DOWNLOAD FUNCTION
# ============================================================

def download_dataset():

    print("=" * 60)
    print("ASTRORISK - NASA/JPL NEO DATA DOWNLOADER")
    print("=" * 60)

    print()
    print("Target records:", TARGET_RECORDS)
    print("Batch size:", BATCH_SIZE)
    print()

    # --------------------------------------------------------
    # Start with an empty dataframe
    # --------------------------------------------------------

    all_data = []

    start = 0

    total_downloaded = 0

    # --------------------------------------------------------
    # Download batches
    # --------------------------------------------------------

    while total_downloaded < TARGET_RECORDS:

        remaining = TARGET_RECORDS - total_downloaded

        current_batch = min(
            BATCH_SIZE,
            remaining
        )

        print()
        print("-" * 60)

        df_batch = fetch_batch(
            start,
            current_batch
        )

        # ----------------------------------------------------
        # Failed batch
        # ----------------------------------------------------

        if df_batch is None:

            print()
            print(
                "This batch failed."
            )

            print(
                "Stopping download safely."
            )

            break

        # ----------------------------------------------------
        # Add successful batch
        # ----------------------------------------------------

        all_data.append(df_batch)

        total_downloaded += len(df_batch)

        start += len(df_batch)

        print(
            f"Batch downloaded: {len(df_batch)}"
        )

        print(
            f"Total downloaded: {total_downloaded}"
        )

        # ----------------------------------------------------
        # SAVE IMMEDIATELY
        # ----------------------------------------------------

        current_df = pd.concat(
            all_data,
            ignore_index=True
        )

        current_df = current_df.drop_duplicates(
            subset="spkid"
        )

        save_dataset(
            current_df
        )

        print(
            "Progress saved to neo_data.csv"
        )

        # Small delay between requests
        time.sleep(2)

    # --------------------------------------------------------
    # FINAL DATA
    # --------------------------------------------------------

    if not all_data:

        return None

    final_df = pd.concat(
        all_data,
        ignore_index=True
    )

    final_df = final_df.drop_duplicates(
        subset="spkid"
    )

    return final_df


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        df = download_dataset()

        if df is None:

            print()
            print("=" * 60)
            print("NO NEW DATA DOWNLOADED")
            print("=" * 60)

            return

        # ----------------------------------------------------
        # Save final dataset
        # ----------------------------------------------------

        save_dataset(df)

        print()
        print("=" * 60)
        print("DOWNLOAD FINISHED")
        print("=" * 60)

        print()
        print(
            "Total records saved:",
            len(df)
        )

        print(
            "Total columns:",
            len(df.columns)
        )

        print()
        print(
            "Saved to:"
        )

        print(
            OUTPUT_FILE
        )

    except Exception as error:

        print()
        print("=" * 60)
        print("ERROR")
        print("=" * 60)

        print(
            error
        )


if __name__ == "__main__":
    main()