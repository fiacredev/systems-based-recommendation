"""Rebuild the metadata similarity matrix from the raw dataset.

    python train.py
"""

import os
from ast import literal_eval

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MOVIE_DIR = "Movie"
MODEL_PATH = os.path.join("models", "hybrid_recommender.pkl")


def load_metadata():
    """Read the movie metadata and pull out the fields we use."""
    md = pd.read_csv(os.path.join(MOVIE_DIR, "movies_metadata.csv"))
    md = md.drop([19730, 29503, 35587])
    md["id"] = md["id"].astype("int")
    md["genres"] = md["genres"].fillna("[]").apply(literal_eval)
    md["genres"] = md["genres"].apply(
        lambda x: [i["name"] for i in x] if isinstance(x, list) else []
    )
    return md


def main():
    print("Loading the dataset")
    md = load_metadata()

    print("Building the similarity matrix")
    count = CountVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, stop_words="english")
    count_matrix = count.fit_transform(md["overview"].fillna(""))
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    os.makedirs("models", exist_ok=True)
    joblib.dump({"cosine_sim": cosine_sim}, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
