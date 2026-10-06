# ReelSense

A production-style movie recommendation system built with modern web and machine learning technologies. 

## 🏗️ Architecture Stack

- **Frontend**: React + TypeScript + Vite (To be built)
- **Backend**: FastAPI (Python 3.11/3.12)
- **Databases**: 
  - PostgreSQL (Relational data & users)
  - Redis (Caching & fast lookups)
  - Qdrant (Vector database for embeddings)
- **Machine Learning**: LightFM (Collaborative Filtering) & Sentence Transformers (Content-based filtering)
- **Dataset**: MovieLens
- **Infrastructure**: Docker & Docker Compose (To be configured)

## 📁 Project Structure

* **`backend/`**: FastAPI server, API routes, database models, and business logic.
* **`frontend/`**: React application and UI components.
* **`ml/`**: Machine learning pipelines.
  * **`data/raw/`**: Unprocessed MovieLens datasets.
  * **`data/processed/`**: Cleaned data ready for training.
  * **`data/embeddings/`**: Generated vector embeddings for Qdrant.
  * **`models/`**: Saved LightFM and embedding models.
  * **`scripts/`**: Data fetching, training, and evaluation scripts.
* **`tests/`**: Unit and integration tests for all components.
* **`docker/`**: Dockerfiles and container configurations (e.g., init scripts).
* **`docs/`**: API documentation, ML experiment logs, and architecture diagrams.

## 🚀 Getting Started

*(Instructions will be added as the project foundation is built out)*

## 📜 License

[MIT](LICENSE)
