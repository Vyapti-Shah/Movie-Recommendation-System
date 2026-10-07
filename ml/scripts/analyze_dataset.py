import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def main():
    script_dir = Path(__file__).parent.resolve()
    processed_dir = script_dir.parent / "data" / "processed"
    eda_dir = script_dir.parent.parent / "docs" / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    
    movies_file = processed_dir / "movies_processed.parquet"
    ratings_file = processed_dir / "ratings_processed.parquet"

    print("--- Starting Exploratory Data Analysis ---")
    if not movies_file.exists() or not ratings_file.exists():
        print("Error: Processed data not found. Run Day 3 scripts first.")
        sys.exit(1)

    movies = pd.read_parquet(movies_file)
    ratings = pd.read_parquet(ratings_file)

    # 1. Rating Distribution
    plt.figure(figsize=(8, 5))
    rating_counts = ratings['rating'].value_counts().sort_index()
    plt.bar(rating_counts.index, rating_counts.values, width=0.4, color='skyblue', edgecolor='black')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(eda_dir / 'rating_distribution.png')
    plt.close()
    print("Generated rating_distribution.png")

    # 2. Ratings per Movie (Long-Tail Distribution)
    plt.figure(figsize=(10, 5))
    movie_counts = ratings['movieId'].value_counts().values
    plt.plot(movie_counts, color='red')
    plt.title('Ratings per Movie (Long-Tail Distribution)')
    plt.xlabel('Movies (Sorted by Popularity)')
    plt.ylabel('Number of Ratings')
    plt.yscale('log') # Log scale to better visualize the tail
    plt.grid(alpha=0.3)
    plt.savefig(eda_dir / 'ratings_per_movie.png')
    plt.close()
    print("Generated ratings_per_movie.png")

    # 3. Ratings per User
    plt.figure(figsize=(8, 5))
    user_counts = ratings['userId'].value_counts()
    plt.hist(user_counts.values, bins=50, color='purple', edgecolor='black')
    plt.title('Distribution of Ratings per User')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(eda_dir / 'ratings_per_user.png')
    plt.close()
    print("Generated ratings_per_user.png")

    # 4. Genre Frequency
    plt.figure(figsize=(10, 6))
    all_genres = movies['genres'].explode()
    # Filter out empty lists or nulls if any exist
    all_genres = all_genres[all_genres.notna()]
    genre_counts = all_genres.value_counts()
    plt.bar(genre_counts.index, genre_counts.values, color='coral', edgecolor='black')
    plt.title('Genre Frequency')
    plt.xlabel('Genre')
    plt.ylabel('Number of Movies')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(eda_dir / 'genre_frequency.png')
    plt.close()
    print("Generated genre_frequency.png")

    # 5. Rating Count vs Average Rating
    plt.figure(figsize=(8, 6))
    movie_stats = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        count=('rating', 'count')
    )
    plt.scatter(movie_stats['avg_rating'], movie_stats['count'], alpha=0.4, color='teal')
    plt.title('Average Rating vs. Rating Count')
    plt.xlabel('Average Rating')
    plt.ylabel('Number of Ratings')
    plt.grid(alpha=0.3)
    plt.savefig(eda_dir / 'rating_vs_popularity.png')
    plt.close()
    print("Generated rating_vs_popularity.png")

    print(f"\n[Success] EDA plots saved to {eda_dir.relative_to(script_dir.parent.parent)}")

if __name__ == "__main__":
    main()
