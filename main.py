import joblib
from fastapi import FastAPI
from recommender import hybrid

# Load model artifacts once when the application starts
artifacts = joblib.load("models/hybrid_recommender.pkl")

svd = artifacts["svd"]
cosine_sim = artifacts["cosine_sim"]
indices = artifacts["indices"]
indices_map = artifacts["indices_map"]
id_map = artifacts["id_map"]
smd = artifacts["smd"]

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Movie Recommendation API"}


@app.get("/recommend")
def recommend(userId: int, title: str):
    movies = hybrid(userId, title)
    return movies.to_dict(orient="records")