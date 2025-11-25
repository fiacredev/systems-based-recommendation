"""Movie recommendation helpers."""

import joblib


def load_artifacts(path="models/hybrid_recommender.pkl"):
    """Load the artifacts produced by the notebook."""
    return joblib.load(path)


def similar_titles(artifacts, title, n=10):
    """Return the titles most similar to the given one."""
    indices = artifacts["indices"]
    cosine_sim = artifacts["cosine_sim"]

    idx = indices[title]
    sim_scores = sorted(
        enumerate(cosine_sim[int(idx)]), key=lambda x: x[1], reverse=True
    )
    movie_indices = [i[0] for i in sim_scores[1 : n + 1]]

    return artifacts["smd"].iloc[movie_indices]["title"]


def hybrid(artifacts, userId, title, n=10):
    """Rank the content based neighbours of a title with the collaborative model."""
    indices = artifacts["indices"]
    cosine_sim = artifacts["cosine_sim"]
    indices_map = artifacts["indices_map"]
    svd = artifacts["svd"]
    smd = artifacts["smd"]

    idx = indices[title]
    sim_scores = sorted(
        enumerate(cosine_sim[int(idx)]), key=lambda x: x[1], reverse=True
    )
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    movies = smd.iloc[movie_indices][
        ["title", "vote_count", "vote_average", "year", "id"]
    ].copy()

    movies["est"] = movies["id"].apply(
        lambda x: svd.predict(userId, indices_map.loc[x]["movieId"]).est
    )
    movies = movies.sort_values("est", ascending=False)

    return movies.head(n)
