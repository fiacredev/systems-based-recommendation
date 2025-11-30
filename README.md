# Movie Project ML

A movie recommendation system built from the MovieLens ratings and the TMDB movie
metadata.

## The idea

Two models do the work here. A content based model finds movies that look alike, and a
collaborative filtering model ranks them based on what people actually rated.

## Project layout

```
.
├── Movie/       the dataset
├── models/      the trained artifacts
└── movie.ipynb  the notebook where it all happens
```

## Getting the data

The dataset is published on Kaggle as *Movie Dataset Collection*. Download it and drop
the files in `Movie/`.

## Running the notebook

```bash
pip install -r requirements.txt
jupyter lab
```

Open `movie.ipynb` and run it from the top.

## Contact

[fiacredeveloper@gmail.com](mailto:fiacredeveloper@gmail.com)
