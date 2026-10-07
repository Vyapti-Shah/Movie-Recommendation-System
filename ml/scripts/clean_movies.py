import pandas as pd
from pathlib import Path
import sys

def clean_movies():
    script_dir = Path(__file__).parent.resolve()
    raw_dir = script_dir.parent / "data" / "raw" / "ml-latest-small"
    processed_dir = script_dir.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = raw_dir / "movies.csv"
    output_file = processed_dir / "movies_processed.parquet"

    print("--- Starting Movie Cleaning Pipeline ---")
    if not input_file.exists():
        print(f"Error: {input_file} not found. Run download script first.")
        sys.exit(1)

    movies = pd.read_csv(input_file)
    initial_count = len(movies)
    
    # 1. Remove missing IDs or titles
    movies = movies.dropna(subset=['movieId', 'title'])
    
    # 2. Clean whitespace from titles
    movies['title'] = movies['title'].str.strip()

    # 3. Deduplicate by title
    # It's possible the dataset has the same movie listed twice under different IDs
    movies = movies.drop_duplicates(subset=['title'], keep='first')
    
    # 4. Normalize Genres
    # Convert "Action|Adventure" to Python list ["Action", "Adventure"]
    # Replace "(no genres listed)" with an empty list
    def process_genres(g_str):
        if pd.isna(g_str) or g_str == "(no genres listed)":
            return []
        return g_str.split('|')
        
    movies['genres'] = movies['genres'].apply(process_genres)

    # --- Validations ---
    # Check for duplicate IDs
    if movies['movieId'].duplicated().any():
        print("Validation Error: Duplicate movieIds found after cleaning!")
        sys.exit(1)
        
    # Check for null IDs
    if movies['movieId'].isnull().any():
        print("Validation Error: Null movieIds found!")
        sys.exit(1)

    final_count = len(movies)
    print(f"Movies cleaned: {initial_count} -> {final_count} (Removed {initial_count - final_count} duplicates/invalid)")

    # Save as Parquet
    # Parquet is columnar, compressed, and maintains datatypes natively (like lists for genres)
    movies.to_parquet(output_file, index=False)
    print(f"Successfully saved to {output_file.relative_to(script_dir.parent.parent)}")

if __name__ == "__main__":
    clean_movies()
