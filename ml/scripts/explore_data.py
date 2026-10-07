import pandas as pd
from pathlib import Path

def explore_movielens():
    script_dir = Path(__file__).parent.resolve()
    raw_dir = script_dir.parent / "data" / "raw" / "ml-latest-small"
    processed_dir = script_dir.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("--- Loading Datasets ---")
    try:
        movies = pd.read_csv(raw_dir / "movies.csv")
        ratings = pd.read_csv(raw_dir / "ratings.csv")
        tags = pd.read_csv(raw_dir / "tags.csv")
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        print("Please run download_data.py first.")
        return

    print("\n--- Basic Statistics ---")
    num_movies = movies['movieId'].nunique()
    num_users = ratings['userId'].nunique()
    num_ratings = len(ratings)
    
    print(f"Number of movies: {num_movies:,}")
    print(f"Number of users: {num_users:,}")
    print(f"Number of ratings: {num_ratings:,}")

    print("\n--- Missing Values ---")
    print("Movies dataset:")
    print(movies.isnull().sum())
    print("\nRatings dataset:")
    print(ratings.isnull().sum())
    
    print("\n--- Rating Distribution ---")
    rating_dist = ratings['rating'].value_counts().sort_index()
    print(rating_dist)
    rating_dist.to_csv(processed_dir / "rating_distribution.csv")
    
    print("\n--- Genre Frequency ---")
    # genres are pipe-separated like "Adventure|Animation|Children|Comedy|Fantasy"
    all_genres = movies['genres'].str.split('|').explode()
    genre_freq = all_genres.value_counts()
    print(genre_freq.head(10))
    genre_freq.to_csv(processed_dir / "genre_frequency.csv")

    print("\n--- Top Rated Movies (Min 50 ratings) ---")
    movie_stats = ratings.groupby('movieId').agg(
        rating_count=('rating', 'count'),
        rating_mean=('rating', 'mean')
    )
    
    # Filter for minimum ratings
    min_ratings = 50
    top_movies = movie_stats[movie_stats['rating_count'] >= min_ratings]
    top_movies = top_movies.sort_values(by='rating_mean', ascending=False).head(10)
    
    # Join with movie titles
    top_movies_with_titles = top_movies.merge(movies[['movieId', 'title']], on='movieId')
    print(top_movies_with_titles[['title', 'rating_mean', 'rating_count']])
    
    # Save the processed aggregated stats
    movie_stats.to_csv(processed_dir / "movie_rating_stats.csv")
    print(f"\n[Success] Saved exploration summaries to {processed_dir.relative_to(script_dir.parent.parent)}")

if __name__ == "__main__":
    explore_movielens()
