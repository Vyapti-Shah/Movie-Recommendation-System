import numpy as np
import pandas as pd
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def build_index():
    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir.parent / "data"
    embeddings_dir = data_dir / "embeddings"
    qdrant_dir = data_dir / "qdrant"
    
    emb_file = embeddings_dir / "movie_embeddings.npy"
    ids_file = embeddings_dir / "movie_ids.npy"
    movies_file = data_dir / "processed" / "movies_processed.parquet"

    if not emb_file.exists():
        logging.error("Embeddings not found. Run generate_embeddings.py first.")
        return

    logging.info("Loading embeddings and metadata...")
    embeddings = np.load(emb_file)
    movie_ids = np.load(ids_file)
    movies_df = pd.read_parquet(movies_file)
    
    # 1. Initialize Local Qdrant
    logging.info("Initializing Qdrant Client...")
    # This creates a local SQLite/RocksDB database inside our data folder
    client = QdrantClient(path=str(qdrant_dir))
    
    collection_name = "movies"
    # Automatically extract the dimension (e.g., 384 for MiniLM)
    vector_size = embeddings.shape[1] 

    # 2. Make Indexing Idempotent
    if client.collection_exists(collection_name=collection_name):
        logging.info(f"Collection '{collection_name}' already exists. Recreating it to ensure idempotency...")
        client.delete_collection(collection_name=collection_name)
        
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

    # 3. Batch Upserts with Metadata Payloads
    logging.info("Preparing payloads and batching points...")
    points = []
    
    # Create a dictionary lookup for metadata
    movies_dict = movies_df.set_index('movieId').to_dict('index')

    # Load TMDB metadata if available to put in the payload
    tmdb_file = data_dir / "raw" / "tmdb_metadata.json"
    tmdb_data = {}
    if tmdb_file.exists():
        import json
        with open(tmdb_file, 'r', encoding='utf-8') as f:
            tmdb_data = json.load(f)

    for i, (movie_id, embedding) in enumerate(zip(movie_ids, embeddings)):
        meta = movies_dict.get(movie_id, {})
        title = meta.get('title', 'Unknown')
        
        # Convert genres safely to list
        genres = meta.get('genres', [])
        if hasattr(genres, '__len__') and not isinstance(genres, str):
            genres_list = list(genres)
        else:
            genres_list = []
            
        # Get TMDB data
        movie_id_str = str(movie_id)
        tmdb_meta = tmdb_data.get(movie_id_str, {})
            
        # The Payload is JSON data attached to the vector
        payload = {
            "movie_id": int(movie_id),
            "title": str(title),
            "genres": genres_list,
            "overview": tmdb_meta.get('overview', ''),
            "poster_path": tmdb_meta.get('poster_path', '')
        }
        
        # Qdrant point IDs can be UUIDs or Integers. movieId is perfect.
        points.append(
            PointStruct(
                id=int(movie_id), 
                vector=embedding.tolist(), 
                payload=payload
            )
        )

    # Upsert in batches of 500 to save memory and optimize network/disk IO
    batch_size = 500
    logging.info(f"Upserting {len(points)} points in batches of {batch_size}...")
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
        
    logging.info("Qdrant indexing complete!")

if __name__ == "__main__":
    build_index()
