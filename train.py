"""Rebuild the hybrid recommender artifact from the raw dataset.

This is the notebook pipeline written as a script, so the artifact can be rebuilt
without opening Jupyter:

    python train.py

It reads the dataset from Movie/, fits the SVD model on the ratings sample, rebuilds
the metadata similarity matrix and writes models/hybrid_recommender.pkl. The API
loads that file at startup, so run this once before serving anything.
"""

import os
from ast import literal_eval

import joblib
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader

MOVIE_DIR = "Movie"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "hybrid_recommender.pkl")

# These three rows have broken values and break everything downstream.
BROKEN_ROWS = [19730, 29503, 35587]


def load_metadata():
    """Read the movie metadata and pull out the fields we actually use."""
    md = pd.read_csv(os.path.join(MOVIE_DIR, "movies_metadata.csv"))

    md = md.drop(BROKEN_ROWS)
    md["id"] = md["id"].astype("int")
    md["genres"] = md["genres"].fillna("[]").apply(literal_eval)
    md["genres"] = md["genres"].apply(
        lambda x: [i["name"] for i in x] if isinstance(x, list) else []
    )
    md["year"] = pd.to_datetime(md["release_date"], errors="coerce").apply(
        lambda x: str(x).split("-")[0] if x != np.nan else np.nan
    )

    return md


def load_links():
    """Return the tmdb ids that have a movielens counterpart."""
    links_small = pd.read_csv(os.path.join(MOVIE_DIR, "links_small.csv"))
    return links_small[links_small["tmdbId"].notnull()]["tmdbId"].astype("int")


def build_soup(md, links_small):
    """Build the metadata soup used to describe every movie."""
    credits = pd.read_csv(os.path.join(MOVIE_DIR, "credits.csv"))
    keywords = pd.read_csv(os.path.join(MOVIE_DIR, "keywords.csv"))

    credits["id"] = credits["id"].astype("int")
    keywords["id"] = keywords["id"].astype("int")

    md = md.merge(credits, on="id")
    md = md.merge(keywords, on="id")
    smd = md[md["id"].isin(links_small)].copy()

    smd["crew"] = smd["crew"].apply(literal_eval)
    smd["keywords"] = smd["keywords"].apply(literal_eval)

    smd["director"] = smd["crew"].apply(find_director)
    smd["cast"] = smd["cast"].apply(lambda x: [i["name"] for i in x] if isinstance(x, list) else [])
    smd["cast"] = smd["cast"].apply(lambda x: x[:3])
    smd["keywords"] = smd["keywords"].apply(
        lambda x: [i["name"] for i in x] if isinstance(x, list) else []
    )

    smd["cast"] = smd["cast"].apply(lambda x: [clean_name(i) for i in x])
    smd["director"] = smd["director"].fillna("").apply(clean_name)
    smd["director"] = smd["director"].apply(lambda x: [x, x, x])

    counts = smd.apply(lambda x: pd.Series(x["keywords"]), axis=1).stack()
    counts = counts.value_counts()
    counts = counts[counts > 1]

    stemmer = SnowballStemmer("english")
    smd["keywords"] = smd["keywords"].apply(
        lambda x: [stemmer.stem(i) for i in x if i in counts]
    )
    smd["keywords"] = smd["keywords"].apply(
        lambda x: [clean_name(i) for i in x]
    )

    smd["soup"] = smd["keywords"] + smd["cast"] + smd["director"] + smd["genres"]
    smd["soup"] = smd["soup"].apply(lambda x: " ".join(x))

    return smd.reset_index()


def find_director(crew):
    """Pull the director out of the crew list."""
    for member in crew:
        if member["job"] == "Director":
            return member["name"]
    return np.nan


def clean_name(name):
    """Lowercase a name and drop the spaces."""
    return str(name).lower().replace(" ", "")


def build_similarity(smd):
    """Compute the cosine similarity between every pair of movies."""
    count = CountVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, stop_words="english")
    count_matrix = count.fit_transform(smd["soup"])
    return cosine_similarity(count_matrix, count_matrix)


def train_svd():
    """Fit the SVD model on the ratings sample."""
    ratings = pd.read_csv(os.path.join(MOVIE_DIR, "ratings_small.csv"))
    reader = Reader(rating_scale=(ratings["rating"].min(), ratings["rating"].max()))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

    trainset = data.build_full_trainset()
    svd = SVD()
    svd.fit(trainset)

    return svd


def build_id_maps(smd):
    """Map titles to movielens ids so the two models can be joined."""
    id_map = pd.read_csv(os.path.join(MOVIE_DIR, "links_small.csv"))[["movieId", "tmdbId"]]
    id_map["tmdbId"] = id_map["tmdbId"].apply(convert_int)
    id_map.columns = ["movieId", "id"]
    id_map = id_map.merge(smd[["title", "id"]], on="id").set_index("title")

    return id_map, id_map.set_index("id")


def convert_int(value):
    """Convert to int, or give up and return nan."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return np.nan


def main():
    print("Loading the dataset")
    md = load_metadata()
    links_small = load_links()

    print("Building the metadata soup")
    smd = build_soup(md, links_small)

    print("Computing the similarity matrix")
    cosine_sim = build_similarity(smd)
    indices = pd.Series(smd.index, index=smd["title"])

    print("Training the SVD model")
    svd = train_svd()

    print("Mapping the ids")
    id_map, indices_map = build_id_maps(smd)

    os.makedirs(MODEL_DIR, exist_ok=True)
    artifacts = {
        "svd": svd,
        "cosine_sim": cosine_sim,
        "indices": indices,
        "indices_map": indices_map,
        "id_map": id_map,
        "smd": smd,
    }
    joblib.dump(artifacts, MODEL_PATH)

    print(f"Saved {MODEL_PATH} with {len(smd)} movies")


if __name__ == "__main__":
    main()
