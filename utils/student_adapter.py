"""
Student Implementation Adapter for Music Recommender Web API
=============================================================
This file bridges your notebook implementation with the Flask web application.

INSTRUCTIONS:
1. Copy your COMPLETE, TESTED implementations below
2. Do NOT modify the get_recommendations_for_api function
3. Test using: python -m utils.test_student_adapter
"""

import numpy as np
import pandas as pd
import os

# ============================================================================
# STUDENT IMPLEMENTATION SECTION
# Copy your complete, final implementations from the notebook below
# ============================================================================

class FeatureScaler:
    """
    TODO: Copy your complete FeatureScaler implementation here.
    Must include: __init__, fit, transform, fit_transform methods
    """
    def __init__(self):
        # Keep lightweight constructor so module import does not fail
        self._fitted = False
    
    def fit(self, X):
        """Learn the scaling parameters from data X."""
        raise NotImplementedError("FeatureScaler.fit is a stub. Please implement it in utils/student_adapter.py")
    
    def transform(self, X):
        """Apply the learned scaling to data X."""
        raise NotImplementedError("FeatureScaler.transform is a stub. Please implement it in utils/student_adapter.py")
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        raise NotImplementedError("FeatureScaler.fit_transform is a stub. Please implement it in utils/student_adapter.py")


class KNNRecommender:
    """
    TODO: Copy your complete KNNRecommender implementation here.
    Must include all methods and static distance functions.
    """
    
    def __init__(self, k=10):
        # Keep constructor minimal so initialization paths do not crash immediately
        self.k = k
        self._fitted = False
    
    @staticmethod
    def euclidean_distance(a, b):
        """Calculate Euclidean distance between vectors a and b."""
        raise NotImplementedError("KNNRecommender.euclidean_distance is a stub. Please implement it in utils/student_adapter.py")
    
    @staticmethod
    def cosine_distance(a, b):
        """Calculate Cosine distance between vectors a and b."""
        raise NotImplementedError("KNNRecommender.cosine_distance is a stub. Please implement it in utils/student_adapter.py")
    
    def fit(self, item_profile_df, feature_columns):
        """Prepare the recommender with track data."""
        raise NotImplementedError("KNNRecommender.fit is a stub. Please implement it in utils/student_adapter.py")
    
    def find_neighbors(self, track_id, n_neighbors=None, distance_metric='euclidean'):
        """Find k nearest neighbors for a track."""
        raise NotImplementedError("KNNRecommender.find_neighbors is a stub. Please implement it in utils/student_adapter.py")
    
    def recommend(self, track_id, n_recommendations=None, distance_metric='euclidean'):
        """Generate recommendations for a track."""
        raise NotImplementedError("KNNRecommender.recommend is a stub. Please implement it in utils/student_adapter.py")


# Optional: If you implemented HybridKNNRecommender, add it here
class HybridKNNRecommender(KNNRecommender):
    """
    OPTIONAL: If you completed the hybrid distance implementation, copy it here.
    """
    pass



# ============================================================================
# API ADAPTER SECTION - DO NOT MODIFY ANYTHING BELOW THIS LINE
# ============================================================================

# Cache for the recommender instance
_recommender_cache = None
_audio_features = ['energy', 'danceability', 'acousticness', 'valence', 
                   'tempo', 'instrumentalness', 'loudness', 'liveness', 'speechiness']


def get_recommendations_for_api(track_id, k=10, metric='cosine', use_hybrid=False):
    """
    Bridge function between student implementation and web API.
    DO NOT MODIFY THIS FUNCTION.
    """
    global _recommender_cache
    
    try:
        # Load data and initialize recommender if needed
        if _recommender_cache is None:
            # Find the data file
            possible_paths = [
                'data/mergedFile.csv',
                '../data/mergedFile.csv',
                os.path.join(os.path.dirname(__file__), '..', 'data', 'mergedFile.csv'),
                'data/item_profile.csv',
                '../data/item_profile.csv',
                os.path.join(os.path.dirname(__file__), '..', 'data', 'item_profile.csv')
            ]
            
            item_profile = None
            for path in possible_paths:
                if os.path.exists(path):
                    item_profile = pd.read_csv(path, dtype={'id': str})
                    break
            
            if item_profile is None:
                raise FileNotFoundError("Could not find mergedFile.csv or item_profile.csv")
            
            # Initialize the appropriate recommender
            if use_hybrid and 'HybridKNNRecommender' in globals():
                _recommender_cache = HybridKNNRecommender(k=k)
            else:
                _recommender_cache = KNNRecommender(k=k)
            
            # Use only features that exist in the loaded file
            available_feats = [f for f in _audio_features if f in item_profile.columns]
            if not available_feats:
                raise ValueError("No required audio feature columns found in data file")

            _recommender_cache.fit(item_profile, available_feats)
            print(f"Initialized {type(_recommender_cache).__name__} with {len(item_profile)} tracks")
        
        # Get recommendations
        recommendations = _recommender_cache.recommend(
            track_id, 
            n_recommendations=k, 
            distance_metric=metric
        )
        
        # Convert to API format
        result = {}
        for _, row in recommendations.iterrows():
            result[row['id']] = {
                'distance': float(row['distance']),
                'song': row.get('song', 'Unknown'),
                'artist': row.get('artist', 'Unknown'),
                'features': {feat: float(row.get(feat, 0)) for feat in _audio_features if feat in row}
            }
        
        return result
        
    except Exception as e:
        print(f"Error in student implementation: {e}")
        import traceback
        traceback.print_exc()
        return {}


def test_implementation():
    """
    Test function to verify your implementation works.
    Run this after copying your code above.
    """
    try:
        from utils.test_student_adapter import run_comprehensive_tests
        return run_comprehensive_tests()
    except ImportError:
        # Fallback if test file is not in expected location
        import subprocess
        import sys
        result = subprocess.run([sys.executable, '-m', 'utils.test_student_adapter'], 
                              capture_output=False)
        return result.returncode == 0


if __name__ == "__main__":
    print("To test your implementation, run:")
    print("  python -m utils.test_student_adapter")             

# ============================================================================
# AUTO-INITIALIZATION FOR API INTEGRATION
# ============================================================================

def initialize_for_api():
    """
    Initialize the recommender for API usage.
    This is called when api_helpers imports this module.
    """
    import os
    import pandas as pd
    
    # Try to find and load the data (prefer mergedFile.csv which has full feature set)
    possible_paths = [
        'data/mergedFile.csv',
        os.path.join(os.path.dirname(__file__), '..', 'data', 'mergedFile.csv'),
        'data/item_profile.csv',
        os.path.join(os.path.dirname(__file__), '..', 'data', 'item_profile.csv')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, dtype={'id': str})
                audio_features = ['energy', 'danceability', 'acousticness', 'valence', 
                                 'tempo', 'instrumentalness', 'loudness', 'liveness', 'speechiness']
                
                # Create and fit the recommender
                recommender = KNNRecommender(k=10)
                recommender.fit(df, audio_features)
                
                print(f"✅ Student recommender initialized with {len(df)} tracks")
                return recommender
                
            except Exception as e:
                print(f"Error loading data from {path}: {e}")
                continue
    
    print("⚠️ Could not initialize student recommender - data files not found")
    return None

# Export the initialized recommender for api_helpers to use (disabled to avoid duplicate init; api_helpers handles it)
# student_recommender_instance = initialize_for_api()