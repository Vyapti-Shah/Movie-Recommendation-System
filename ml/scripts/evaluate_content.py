import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(project_root))

from ml.recommender.content_based import ContentRecommender

def evaluate():
    print("Loading recommender model...")
    recommender = ContentRecommender()
    
    test_movies = [
        "Inception",
        "Interstellar",
        "Toy Story (1995)",
        "Godfather, The",
        "Hangover, The"
    ]
    
    print("\n" + "="*70)
    print("CONTENT RECOMMENDER - MANUAL EVALUATION REPORT")
    print("="*70)
    
    for title_query in test_movies:
        # Simple string match to find the movie ID
        matches = recommender.movies_df[recommender.movies_df['title'].str.contains(title_query, case=False, regex=False)]
        
        if matches.empty:
            print(f"Movie matching '{title_query}' not found.")
            continue
            
        # Take the exact or first match
        target_movie = matches.iloc[0]
        target_id = target_movie['movieId']
        target_title = target_movie['title']
        
        # Format target genres natively
        target_genres_raw = target_movie['genres']
        if hasattr(target_genres_raw, '__len__') and not isinstance(target_genres_raw, str):
            target_genres = ", ".join(list(target_genres_raw))
        else:
            target_genres = str(target_genres_raw)
            
        print(f"\n[TARGET] {target_title}")
        print(f"Genres: {target_genres}")
        print("-" * 70)
        
        try:
            recs = recommender.get_similar_movies(target_id, k=5)
            for idx, row in recs.iterrows():
                print(f"  {row['similarity_score']:.4f} | {row['title']:<40} | {row['genres']}")
        except Exception as e:
            print(f"  Error generating recommendations: {e}")
            
    print("\n" + "="*70)

if __name__ == "__main__":
    evaluate()
