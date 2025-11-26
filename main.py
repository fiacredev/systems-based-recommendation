"""Movie recommendation API."""

from fastapi import FastAPI

app = FastAPI(title="Movie Recommendation API")


@app.get("/")
def home():
    return {"message": "Movie Recommendation API"}
