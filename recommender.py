import joblib

artifacts = joblib.load("models/hybrid_recommender.pkl")

svd = artifacts["svd"]
cosine_sim = artifacts["cosine_sim"]
indices = artifacts["indices"]
indices_map = artifacts["indices_map"]
id_map = artifacts["id_map"]
smd = artifacts["smd"]


def hybrid(userId, title):
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]

    movie_indices = [i[0] for i in sim_scores]

    movies = smd.iloc[movie_indices][
        ["title", "vote_count", "vote_average", "year", "id"]
    ].copy()

    movies["est"] = movies["id"].apply(
        lambda x: svd.predict(userId, indices_map.loc[x]["movieId"]).est
    )

    movies = movies.sort_values("est", ascending=False)

    return movies.head(10)