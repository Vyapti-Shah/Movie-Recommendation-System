import pandas as pd
from pathlib import Path
import sys

def build_movie_text():
    script_dir = Path(__file__).parent.resolve()
    processed_dir = script_dir.parent / "data" / "processed"
    raw_dir = script_dir.parent / "data" / "raw" / "ml-latest-small"
    
    movies_input = processed_dir / "movies_processed.parquet"
    tags_input = raw_dir / "tags.csv"
    output_file = processed_dir / "movie_documents.parquet"

    print("--- Starting Movie Text Engineering Pipeline ---")
    if not movies_input.exists():
        print(f"Error: {movies_input} not found. Run clean_movies.py first.")
        sys.exit(1)
    if not tags_input.exists():
        print(f"Error: {tags_input} not found. Run download_data.py first.")
        sys.exit(1)

    movies = pd.read_parquet(movies_input)
    tags_df = pd.read_csv(tags_input)

    # 1. Process Tags
    # Drop missing tags, normalize to lowercase, strip spaces to ensure uniformity
    tags_df = tags_df.dropna(subset=['tag'])
    tags_df['tag'] = tags_df['tag'].astype(str).str.lower().str.strip()
    
    # Group by movieId and keep unique tags only using set() to avoid duplicated words
    tags_grouped = tags_df.groupby('movieId')['tag'].apply(lambda x: list(set(x))).reset_index()

    # 2. Merge with Movies
    # Left merge so movies without tags are still kept
    movies_text = movies.merge(tags_grouped, on='movieId', how='left')
    
    # Fill NaN for movies with no tags with an empty list
    movies_text['tag'] = movies_text['tag'].apply(lambda d: d if isinstance(d, list) else [])

    # Optional: Load rich TMDB metadata
    tmdb_file = processed_dir.parent / "raw" / "tmdb_metadata.json"
    tmdb_data = {}
    if tmdb_file.exists():
        import json
        with open(tmdb_file, 'r', encoding='utf-8') as f:
            tmdb_data = json.load(f)
            
    # 3. Build Semantic Document
    # We construct a clean, structured string block for each movie
    def create_document(row):
        doc_parts = []
        
        # Title
        title = str(row['title']).strip()
        if title:
            doc_parts.append(f"Title: {title}")
            
        # Genres
        genres = row['genres']
        if hasattr(genres, '__len__') and len(genres) > 0:
            doc_parts.append(f"Genres: {', '.join(list(genres))}")
            
        # Tags
        tags = row['tag']
        if hasattr(tags, '__len__') and len(tags) > 0:
            doc_parts.append(f"Tags: {', '.join(list(tags))}")
            
        # TMDB Rich Metadata
        movie_id_str = str(row['movieId'])
        if movie_id_str in tmdb_data and "error" not in tmdb_data[movie_id_str]:
            meta = tmdb_data[movie_id_str]
            if meta.get("overview"):
                doc_parts.append(f"Overview: {meta['overview']}")
            if meta.get("director"):
                doc_parts.append(f"Director: {meta['director']}")
            if meta.get("cast") and len(meta["cast"]) > 0:
                doc_parts.append(f"Cast: {', '.join(meta['cast'])}")
            
        # Join with newlines
        return "\n".join(doc_parts)

    movies_text['movie_document'] = movies_text.apply(create_document, axis=1)

    # Rename 'tag' back to 'tags' for consistency in output metadata
    movies_text = movies_text.rename(columns={'tag': 'tags'})

    # 4. Save to Parquet
    # We save both the structured document AND the original metadata arrays
    movies_text.to_parquet(output_file, index=False)
    
    print(f"Engineered semantic documents for {len(movies_text)} movies.")
    
    print("\nExample documents:")
    print("-" * 30)
    # Find a movie that definitely has tags, like Pulp Fiction (296) or Inception (79132)
    # Using a generic sort to find one with tags
    sample = movies_text[movies_text['tags'].map(len) > 2].iloc[0]
    print(sample['movie_document'])
    print("-" * 30)
    
    print(f"\n[Success] Saved to {output_file.relative_to(script_dir.parent.parent)}")

if __name__ == "__main__":
    build_movie_text()
