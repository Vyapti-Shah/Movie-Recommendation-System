import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def generate_embeddings():
    script_dir = Path(__file__).parent.resolve()
    processed_dir = script_dir.parent / "data" / "processed"
    embeddings_dir = script_dir.parent / "data" / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = processed_dir / "movie_documents.parquet"
    emb_file = embeddings_dir / "movie_embeddings.npy"
    ids_file = embeddings_dir / "movie_ids.npy"

    logging.info("--- Starting Embedding Generation ---")
    if not input_file.exists():
        logging.error(f"{input_file} not found. Run Day 4 scripts first.")
        sys.exit(1)

    # Check for resumability
    if emb_file.exists() and ids_file.exists():
        logging.info(f"Embeddings already exist at {emb_file.name}. Skipping generation.")
        logging.info("Delete these files if you want to force regeneration.")
        return

    # 1. Load Data
    logging.info("Loading movie documents...")
    movies_df = pd.read_parquet(input_file)
    
    # 2. Extract texts and IDs
    movie_ids = movies_df['movieId'].values
    documents = movies_df['movie_document'].tolist()
    
    # 3. Load Model (Only once)
    model_name = 'all-MiniLM-L6-v2'
    local_model_path = script_dir.parent / "models" / model_name
    
    if local_model_path.exists():
        logging.info(f"Loading SentenceTransformer from local path: {local_model_path.name}...")
        model = SentenceTransformer(str(local_model_path))
    else:
        logging.info(f"Downloading pretrained SentenceTransformer model: {model_name}...")
        model = SentenceTransformer(model_name)
        logging.info(f"Saving model locally to {local_model_path.relative_to(script_dir.parent.parent)} for offline use...")
        local_model_path.mkdir(parents=True, exist_ok=True)
        model.save(str(local_model_path))
    
    # 4. Generate Embeddings (batching handled natively by sentence-transformers)
    logging.info(f"Generating embeddings for {len(documents)} movies...")
    logging.info("This might take a few minutes on CPU...")
    
    # normalize_embeddings=True standardizes vector length to 1.
    # This mathematically simplifies cosine similarity to a simple dot product later.
    embeddings = model.encode(documents, batch_size=8, show_progress_bar=True, normalize_embeddings=True)
    
    # 5. Save outputs
    logging.info(f"Saving embeddings and IDs to {embeddings_dir.relative_to(script_dir.parent.parent)}...")
    np.save(emb_file, embeddings)
    np.save(ids_file, movie_ids)
    
    logging.info("Embedding generation complete!")

if __name__ == "__main__":
    generate_embeddings()
