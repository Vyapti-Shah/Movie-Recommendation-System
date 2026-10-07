import pandas as pd
from pathlib import Path
import sys

def build_interactions():
    script_dir = Path(__file__).parent.resolve()
    raw_dir = script_dir.parent / "data" / "raw" / "ml-latest-small"
    processed_dir = script_dir.parent / "data" / "processed"
    
    ratings_input = raw_dir / "ratings.csv"
    movies_input = processed_dir / "movies_processed.parquet"
    output_file = processed_dir / "ratings_processed.parquet"

    print("--- Starting Interactions Pipeline ---")
    if not ratings_input.exists():
        print(f"Error: {ratings_input} not found.")
        sys.exit(1)
    if not movies_input.exists():
        print(f"Error: {movies_input} not found. Run clean_movies.py first.")
        sys.exit(1)

    ratings = pd.read_csv(ratings_input)
    movies = pd.read_parquet(movies_input)
    initial_count = len(ratings)

    # 1. Drop missing critical fields
    ratings = ratings.dropna(subset=['userId', 'movieId', 'rating', 'timestamp'])

    # 2. Validate ranges (MovieLens is 0.5 to 5.0)
    out_of_bounds = ratings[(ratings['rating'] < 0.5) | (ratings['rating'] > 5.0)]
    if not out_of_bounds.empty:
        print(f"Warning: Dropping {len(out_of_bounds)} ratings out of valid bounds [0.5 - 5.0]")
        ratings = ratings[(ratings['rating'] >= 0.5) & (ratings['rating'] <= 5.0)]

    # 3. Convert timestamps properly
    # MovieLens timestamp is Unix seconds
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

    # 4. Remove duplicate interactions
    # A user shouldn't rate the same movie twice. If they do, keep the most recent rating.
    ratings = ratings.sort_values('timestamp').drop_duplicates(subset=['userId', 'movieId'], keep='last')

    # 5. Referential Integrity (Foreign Key validation)
    # Ensure all ratings correspond to a valid, cleaned movie ID
    valid_movie_ids = set(movies['movieId'])
    valid_ratings = ratings['movieId'].isin(valid_movie_ids)
    invalid_count = (~valid_ratings).sum()
    if invalid_count > 0:
        print(f"Warning: Dropping {invalid_count} ratings for non-existent movies.")
        ratings = ratings[valid_ratings]

    # --- Validations ---
    if ratings['rating'].isnull().any():
        print("Validation Error: Null ratings found after cleaning!")
        sys.exit(1)
        
    if ratings.duplicated(subset=['userId', 'movieId']).any():
        print("Validation Error: Duplicate user/movie interactions still exist!")
        sys.exit(1)

    final_count = len(ratings)
    print(f"Ratings cleaned: {initial_count} -> {final_count} (Removed {initial_count - final_count} duplicates/invalid)")

    # Save as Parquet
    ratings.to_parquet(output_file, index=False)
    print(f"Successfully saved to {output_file.relative_to(script_dir.parent.parent)}")

if __name__ == "__main__":
    build_interactions()
