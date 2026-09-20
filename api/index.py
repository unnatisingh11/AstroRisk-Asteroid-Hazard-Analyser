from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os


# ---------------------------------
# Paths
# ---------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "neo_data.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "astrorisk_rf.pkl"
)


# ---------------------------------
# Load dataset and model
# ---------------------------------

df = pd.read_csv(DATA_PATH)

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
features = model_data["features"]


# ---------------------------------
# FastAPI application
# ---------------------------------

app = FastAPI(
    title="AstroRisk API",
    description="API for asteroid PHA classification",
    version="1.0.0"
)


# ---------------------------------
# CORS
# ---------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------
# Response helper
# ---------------------------------

def asteroid_response(asteroid):

    input_data = asteroid[features].to_frame().T

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    return {
        "spkid": str(asteroid["spkid"]),
        "designation": str(asteroid["pdes"]),
        "name": str(asteroid["full_name"]),

        "pha_probability": round(
            float(probability) * 100,
            2
        ),

        "classification": (
            "PHA"
            if prediction == 1
            else "Non-PHA"
        ),

        "orbital_parameters": {
            "eccentricity": float(asteroid["e"]),
            "semi_major_axis": float(asteroid["a"]),
            "perihelion_distance": float(asteroid["q"]),
            "inclination": float(asteroid["i"]),
            "orbital_period": float(asteroid["per"])
        },

        "observation_information": {
            "data_arc": float(asteroid["data_arc"]),
            "observations": int(asteroid["n_obs_used"]),
            "condition_code": float(
                asteroid["condition_code"]
            )
        }
    }


# ---------------------------------
# Home endpoint
# ---------------------------------

@app.get("/")
def home():

    return {
        "message": "AstroRisk API is running",
        "version": "1.0.0"
    }


# ---------------------------------
# Get all asteroid names
# ---------------------------------

@app.get("/api/asteroids")
def get_asteroids():

    asteroids = []

    for _, asteroid in df.iterrows():

        asteroids.append({
            "spkid": str(asteroid["spkid"]),
            "designation": str(asteroid["pdes"]),
            "name": str(asteroid["full_name"])
        })

    return {
        "count": len(asteroids),
        "asteroids": asteroids
    }


# ---------------------------------
# Search asteroid
# ---------------------------------

@app.get("/api/asteroids/search")
def search_asteroids(q: str):

    query = q.strip().lower()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty."
        )

    matches = df[
        df["pdes"].astype(str).str.lower().str.contains(
            query,
            na=False
        )
        |
        df["spkid"].astype(str).str.lower().str.contains(
            query,
            na=False
        )
        |
        df["full_name"].astype(str).str.lower().str.contains(
            query,
            na=False
        )
    ]

    results = []

    for _, asteroid in matches.head(20).iterrows():

        results.append({
            "spkid": str(asteroid["spkid"]),
            "designation": str(asteroid["pdes"]),
            "name": str(asteroid["full_name"])
        })

    return {
        "count": len(results),
        "results": results
    }


# ---------------------------------
# Analyze asteroid
# ---------------------------------

@app.get("/api/asteroids/{spkid}")
def analyze_asteroid(spkid: str):

    matches = df[
        df["spkid"].astype(str) == str(spkid)
    ]

    if matches.empty:

        raise HTTPException(
            status_code=404,
            detail="Asteroid not found."
        )

    asteroid = matches.iloc[0]

    return asteroid_response(asteroid)