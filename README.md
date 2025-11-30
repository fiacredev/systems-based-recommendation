# Hybrid Movie Recommender

A movie recommendation engine that mixes two very different ways of guessing what
someone might want to watch next: what a film looks like on paper, and what people who
rate a lot of movies actually thought of it.

The project started as a notebook where I was poking around the MovieLens and TMDB
metadata, and it slowly grew into something that serves recommendations over HTTP.

## What the recommender does

Two models do the work, and a third one decides how to combine them.

**Content based.** Every movie is described by its keywords, its three main actors, its
director and its genres. All of that is glued into one text blob, the blob is turned
into a vector, and the cosine similarity between every pair of movies gives a
neighbourhood of films that look alike. Show it The Dark Knight and it hands back the
Batman films, the Nolan films, and the crime dramas.

**Collaborative.** Ratings from the MovieLens users are fed to an SVD model from the
Surprise library. It learns what users with similar taste tend to rate highly, which
catches things metadata cannot: that a person who enjoys slow dramas also enjoys this
obscure French thriller that shares no keywords with anything else they rated.

**Hybrid.** The content based neighbours are the candidate pool, the SVD model ranks
them, and the top ten survive. Content based picks the neighbourhood, collaborative
decides the order inside it.

## Project layout

```
.
├── Movie/                     raw dataset files (see the fetch script below)
├── models/                    trained artifacts, rebuilt with train.py
├── scripts/fetch_data.sh      downloads the dataset
├── main.py                    FastAPI application
├── recommender.py             recommendation logic used by the API
├── train.py                   rebuilds models/hybrid_recommender.pkl
└── movie.ipynb                the notebook where the whole thing was built
```

The notebook is the source of truth for the pipeline. `train.py` is the same pipeline
written as a script so the artifact can be rebuilt without opening Jupyter.

## Getting the data

The dataset is the MovieLens data joined with TMDB metadata. It is published on Kaggle
as *Movie Dataset Collection*.

```bash
./scripts/fetch_data.sh
```

The script uses the Kaggle CLI if it is installed and authenticated. If you would rather
download the archive by hand, drop the zip in the project root and run the script again,
or set `DATASET_URL` to a direct link:

```bash
DATASET_URL=https://example.com/movie-dataset.zip ./scripts/fetch_data.sh
```

Everything lands in `Movie/`. The notebook reads from that folder, so keep the file
names as they come.

Two of the files are too big to keep in git: `credits.csv` and `ratings.csv`. The fetch
script pulls them along with the rest, and `.gitignore` keeps them out of the
repository.

## Running the notebook

```bash
pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4
jupyter lab
```

Then open `movie.ipynb` and run it from the top. It takes a while, mostly on the two
similarity matrices, and it writes the trained artifact to `models/` at the end.

## Rebuilding the model

If you would rather not open the notebook:

```bash
python train.py
```

It reads the dataset from `Movie/`, fits the SVD model, rebuilds the similarity matrix
and writes `models/hybrid_recommender.pkl`. The artifact is around 700 MB, which is
why it is not kept in git. Rebuild it instead of downloading it.

## Serving the API

```bash
uvicorn main:app --reload
```

Two endpoints:

| Method | Path         | What it does                                   |
| ------ | ------------ | ---------------------------------------------- |
| GET    | `/`          | Health check                                  |
| GET    | `/recommend` | Hybrid recommendations for a user and a title  |

`/recommend` takes a `userId` and a `title`, both as query parameters:

```bash
curl "http://127.0.0.1:8000/recommend?userId=1&title=Avatar"
```

The response is a list of ten movies with their title, release year, vote count, vote
average, TMDB id and the estimated rating the SVD model gave them:

```json
[
  {
    "title": "Terminator 2: Judgment Day",
    "vote_count": 4274.0,
    "vote_average": 7.7,
    "year": "1991",
    "id": 280,
    "est": 3.457096
  }
]
```

The title has to match exactly what is in the dataset, spelling and capitalisation
included. `/docs` serves the interactive Swagger page if you would rather click through
it.

## Notes and limits

A few things worth knowing before you trust the numbers.

The title lookup is exact. `Avatar` works, `avatar` does not, and `Avatar: The Way of
Water` is a different film with a different entry. A small fuzzy match would make the
API friendlier, but it is not in there yet.

The SVD model was trained on `ratings_small.csv`, about a hundred thousand ratings from
roughly a thousand users. It is enough to show the idea, not enough to be clever about
obscure films. The full ratings file improves the predictions noticeably.

The content based half has no notion of quality. It happily returns a documentary with
the same keywords as the film you asked about. The weighted rating filter that was in
the notebook is not part of the served version, and it should be.

Three rows in the metadata have broken values and get dropped early on. They are
removed by index in the notebook and by the same index list in `train.py`.

The notebook reads the metadata as `MOVIE/movies_metadata.csv` while the ratings come
from `Movie/ratings_small.csv`. That works on a case insensitive filesystem, which is
most desktops, and it fails on a case sensitive one. `train.py` uses `Movie/` for both
and has no such problem.

## Contact

Questions, ideas, or a bug I did not notice: open an issue on the repository, or write
to me at [fiacredeveloper@gmail.com](mailto:fiacredeveloper@gmail.com).

Built by [fiacredev](https://github.com/fiacredev).
