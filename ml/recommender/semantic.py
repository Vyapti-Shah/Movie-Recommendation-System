from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from pathlib import Path

class SemanticRecommender:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            script_dir = Path(__file__).parent.resolve()
            self.data_dir = script_dir.parent / "data"
        else:
            self.data_dir = Path(data_dir)
            
        self.qdrant_dir = self.data_dir / "qdrant"
        self.local_model_path = self.data_dir.parent / "models" / "all-MiniLM-L6-v2"
        
        # Connect to local Qdrant
        self.client = QdrantClient(path=str(self.qdrant_dir))
        self.collection_name = "movies"
        
        # Lazy load model (only needed for raw text searches, not ID lookups)
        self.model = None

    def _get_model(self):
        if self.model is None:
            if self.local_model_path.exists():
                self.model = SentenceTransformer(str(self.local_model_path))
            else:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.model

    def get_similar_movies(self, movie_id: int, k: int = 10):
        """
        Retrieves similar movies by fetching the vector from Qdrant directly,
        bypassing the embedding model entirely. Fast O(1) lookup + O(log N) search.
        """
        # Fetch the precomputed vector directly from Qdrant
        response = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[movie_id],
            with_vectors=True
        )
        
        if not response:
            raise ValueError(f"Movie ID {movie_id} not found in Qdrant collection.")
            
        target_vector = response[0].vector
        
        # Search Qdrant for similar vectors using HNSW algorithm
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=target_vector,
            limit=k + 1  # Fetch +1 to exclude itself
        )
        
        recommendations = []
        for result in search_result.points:
            if result.id == movie_id:
                continue # Skip the target movie itself
                
            recommendations.append({
                "movie_id": result.payload.get("movie_id"),
                "title": result.payload.get("title"),
                "genres": result.payload.get("genres"),
                "score": result.score
            })
            
        return recommendations[:k]
        
    def search_by_text(self, query: str, k: int = 10):
        """
        Encodes a free-text search query using the ML model and searches Qdrant.
        """
        model = self._get_model()
        # Normalization simplifies distance calculations natively in Qdrant
        query_vector = model.encode(query, normalize_embeddings=True).tolist()
        
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k
        )
        
        recommendations = []
        for result in search_result.points:
            recommendations.append({
                "movie_id": result.payload.get("movie_id"),
                "title": result.payload.get("title"),
                "genres": result.payload.get("genres"),
                "score": result.score
            })
            
        return recommendations

# Example usage
if __name__ == "__main__":
    recommender = SemanticRecommender()
    
    print("--- Testing ID Search (Inception) ---")
    recs = recommender.get_similar_movies(79132, k=5)
    for r in recs:
        print(f"{r['score']:.4f} | {r['title']} | {r['genres']}")
        
    print("\n--- Testing Free Text Search ('a movie about stealing secrets in dreams') ---")
    # This proves the model understands semantic concepts beyond just ID mappings
    recs = recommender.search_by_text("a movie about stealing secrets in dreams", k=5)
    for r in recs:
        print(f"{r['score']:.4f} | {r['title']} | {r['genres']}")
