import pandas as pd
import requests
import json
import time
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def fetch_tmdb():
    API_KEY = os.getenv("TMDB_API_KEY")
    if not API_KEY:
        logging.error("TMDB_API_KEY environment variable is not set.")
        logging.error("Please get a free API key from https://www.themoviedb.org/settings/api")
        return

    script_dir = Path(__file__).parent.resolve()
    raw_dir = script_dir.parent / "data" / "raw" / "ml-latest-small"
    output_file = script_dir.parent / "data" / "raw" / "tmdb_metadata.json"
    
    links_file = raw_dir / "links.csv"
    if not links_file.exists():
        logging.error("links.csv not found.")
        return
        
    links = pd.read_csv(links_file)
    # Drop rows without tmdbId
    links = links.dropna(subset=['tmdbId'])
    
    # Load existing progress if any so we can resume if it crashes
    metadata = {}
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            logging.info(f"Loaded {len(metadata)} existing records.")

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    logging.info(f"Fetching TMDB metadata for {len(links)} movies...")
    session = requests.Session()
    
    count = 0
    for _, row in links.iterrows():
        movie_id = str(int(row['movieId']))
        tmdb_id = int(row['tmdbId'])
        
        if movie_id in metadata:
            continue
            
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?append_to_response=credits"
        
        try:
            # Support both v3 API Keys and v4 Bearer Tokens
            if len(API_KEY) < 50:
                response = session.get(f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={API_KEY}&append_to_response=credits", timeout=10)
            else:
                response = session.get(url, headers=headers, timeout=10)
                
            if response.status_code == 200:
                data = response.json()
                
                # Extract what we need
                overview = data.get('overview', '')
                poster_path = data.get('poster_path', '')
                
                # Extract top 5 cast members
                cast = []
                credits = data.get('credits', {})
                for cast_member in credits.get('cast', [])[:5]:
                    cast.append(cast_member.get('name'))
                    
                # Extract director
                director = None
                for crew_member in credits.get('crew', []):
                    if crew_member.get('job') == 'Director':
                        director = crew_member.get('name')
                        break
                        
                metadata[movie_id] = {
                    "overview": overview,
                    "poster_path": poster_path,
                    "cast": cast,
                    "director": director
                }
                
            elif response.status_code == 429: # Rate limit
                logging.warning("Rate limit hit. Sleeping for 5 seconds...")
                time.sleep(5)
                continue
            elif response.status_code == 404:
                metadata[movie_id] = {"error": "not_found"}
                
        except Exception as e:
            logging.error(f"Error fetching {tmdb_id}: {e}")
            
        count += 1
        if count % 100 == 0:
            logging.info(f"Processed {count} new movies...")
            # Save progress incrementally
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
                
        # TMDB allows ~40 requests per second. A small sleep keeps us safe.
        time.sleep(0.05)
        
    # Final save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Finished fetching TMDB data! Saved to {output_file.name}")

if __name__ == "__main__":
    fetch_tmdb()
