import pytest
import sys
from pathlib import Path

# Ensure the root of the project is in the python path
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from ml.recommender.content_based import ContentRecommender

@pytest.fixture(scope="module")
def recommender():
    # Load once for all tests to speed up execution
    return ContentRecommender()

def test_valid_movie_recommendations(recommender):
    # Test Toy Story (movieId 1)
    recs = recommender.get_similar_movies(movie_id=1, k=10)
    assert not recs.empty, "Recommendations should not be empty"
    assert len(recs) == 10, "Should return exactly 10 recommendations"
    assert 'movieId' in recs.columns, "Should contain movieId column"
    assert 'title' in recs.columns, "Should contain title column"
    assert 'similarity_score' in recs.columns, "Should contain similarity_score column"

def test_invalid_movie_id(recommender):
    # Testing an ID that is extremely unlikely to exist
    with pytest.raises(ValueError):
        recommender.get_similar_movies(movie_id=-999, k=10)

def test_top_k_behavior(recommender):
    k = 5
    recs = recommender.get_similar_movies(movie_id=1, k=k)
    assert len(recs) == k, "Should respect the k parameter"

def test_self_exclusion(recommender):
    target_id = 1
    recs = recommender.get_similar_movies(movie_id=target_id, k=20)
    # The target movie should never be recommended to itself
    assert target_id not in recs['movieId'].values, "The source movie should be excluded from results"

def test_similarity_ordering(recommender):
    recs = recommender.get_similar_movies(movie_id=1, k=10)
    scores = recs['similarity_score'].tolist()
    # Check if scores are strictly descending
    assert scores == sorted(scores, reverse=True), "Recommendations must be ordered by similarity score descending"

def test_duplicate_recommendations(recommender):
    recs = recommender.get_similar_movies(movie_id=1, k=10)
    # Check that all recommended movie IDs are unique
    assert len(recs['movieId'].unique()) == len(recs), "There should be no duplicate movies in the recommendation list"
