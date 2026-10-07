import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import logging

class ContentRecommender:
    def __init__(self, data_dir: str = None):
        """
        Initializes the content-based recommender by loading precomputed embeddings,
        movie IDs, and the movie metadata.
        """
        if data_dir is None:
            script_dir = Path(__file__).parent.resolve()
            self.data_dir = script_dir.parent / "data"
        else:
            self.data_dir = Path(data_dir)
            
        self.embeddings_dir = self.data_dir / "embeddings"
        self.processed_dir = self.data_dir / "processed"
        
        self.emb_file = self.embeddings_dir / "movie_embeddings.npy"
        self.ids_file = self.embeddings_dir / "movie_ids.npy"
        self.movies_file = self.processed_dir / "movies_processed.parquet"
        
        self._load_data()
        
    def _load_data(self):
        logging.info("Loading embeddings and movie metadata...")
        if not self.emb_file.exists() or not self.ids_file.exists():
            raise FileNotFoundError("Embeddings not found. Run generate_embeddings.py first.")
            
        self.embeddings = np.load(self.emb_file)
        self.movie_ids = np.load(self.ids_file)
        
        # Load metadata for display purposes (mapping ID -> Title/Genres)
        self.movies_df = pd.read_parquet(self.movies_file)
        
        # Create a mapping from movieId to matrix row index for O(1) fast lookup
        self.id_to_idx = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        logging.info(f"Loaded content data for {len(self.movie_ids)} movies.")

    def get_similar_movies(self, movie_id: int, k: int = 10) -> pd.DataFrame:
        """
        Returns the top K similar movies based on semantic cosine similarity.
        """
        if movie_id not in self.id_to_idx:
            raise ValueError(f"Movie ID {movie_id} not found in catalog.")
            
        # 1. Get the target movie's embedding
        target_idx = self.id_to_idx[movie_id]
        target_embedding = self.embeddings[target_idx].reshape(1, -1)
        
        # 2. Compute Cosine Similarity between the target and ALL other movies in the catalog.
        # Since we normalized embeddings during generation, this is incredibly fast.
        similarities = cosine_similarity(target_embedding, self.embeddings)[0]
        
        # 3. Get top K indices (ignoring the target movie itself)
        # argsort sorts ascending. We take the end (highest scores), reverse, and filter the target itself.
        top_indices = np.argsort(similarities)[-(k+1):][::-1]
        top_indices = [idx for idx in top_indices if idx != target_idx][:k]
        
        # 4. Map back to movie data
        recommendations = []
        for idx in top_indices:
            sim_movie_id = self.movie_ids[idx]
            score = similarities[idx]
            
            # Fetch metadata
            movie_data = self.movies_df[self.movies_df['movieId'] == sim_movie_id].iloc[0]
            
            recommendations.append({
                'movieId': sim_movie_id,
                'title': movie_data['title'],
                'genres': ", ".join(movie_data['genres']) if len(movie_data['genres']) > 0 else "None",
                'similarity_score': round(float(score), 4)
            })
            
        return pd.DataFrame(recommendations)

# Example usage if script is run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        recommender = ContentRecommender()
        
        # Let's test with Inception (movieId 79132)
        target_id = 79132 
        
        target_movie = recommender.movies_df[recommender.movies_df['movieId'] == target_id].iloc[0]
        print(f"\nTarget Movie: {target_movie['title']}")
        print(f"Genres: {target_movie['genres']}\n")
        
        print("--- Top 10 Semantic Recommendations ---")
        recs = recommender.get_similar_movies(target_id, k=10)
        print(recs[['title', 'similarity_score', 'genres']].to_string(index=False))
        
    except Exception as e:
        logging.error(f"Error: {e}")
