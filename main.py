"""Movie recommendation API."""

from fastapi import FastAPI

from recommender import hybrid, load_artifacts

app = FastAPI(title="Movie Recommendation API")

artifacts = load_artifacts()


@app.get("/")
def home():
    return {"message": "Movie Recommendation API"}


@app.get("/recommend")
def recommend(userId: int, title: str):
    """Return hybrid recommendations for a user and a movie they like."""
    movies = hybrid(artifacts, userId, title)
    return movies.to_dict(orient="records")
