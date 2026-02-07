#!/usr/bin/env python
"""
Test script for verifying student implementation
Run this after completing utils/student_adapter.py:
    python -m utils.test_student_adapter
"""

import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_comprehensive_tests():
    """Run comprehensive tests on the student implementation."""
    
    print("="*60)
    print("MUSIC RECOMMENDER - STUDENT IMPLEMENTATION TEST SUITE")
    print("="*60)
    
    total_tests = 0
    passed_tests = 0
    
    # Test 1: Import student implementation
    print("\n[TEST 1] Importing student implementation...")
    try:
        from utils.student_adapter import FeatureScaler, KNNRecommender, get_recommendations_for_api
        print("✅ Successfully imported all required components")
        passed_tests += 1
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Make sure you've copied your implementations to utils/student_adapter.py")
        return
    finally:
        total_tests += 1
    
    # Test 2: FeatureScaler implementation
    print("\n[TEST 2] Testing FeatureScaler...")
    try:
        # Test initialization
        scaler = FeatureScaler()
        
        # Test with simple data
        test_data = np.array([[1, 100], [2, 200], [3, 300], [4, 400], [5, 500]], dtype=float)
        scaled_data = scaler.fit_transform(test_data)
        
        # Check mean is approximately 0
        mean_val = np.mean(scaled_data, axis=0)
        assert np.allclose(mean_val, [0, 0], atol=0.01), f"Mean not 0: {mean_val}"
        
        # Check std is approximately 1
        std_val = np.std(scaled_data, axis=0)
        assert np.allclose(std_val, [1, 1], atol=0.01), f"Std not 1: {std_val}"
        
        # Test transform separately
        new_data = np.array([[3, 300]], dtype=float)
        transformed = scaler.transform(new_data)
        assert transformed.shape == new_data.shape, "Transform changed shape"
        
        print("✅ FeatureScaler works correctly")
        passed_tests += 1
    except Exception as e:
        print(f"❌ FeatureScaler failed: {e}")
    finally:
        total_tests += 1
    
    # Test 3: Distance functions
    print("\n[TEST 3] Testing distance functions...")
    try:
        # Test vectors
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        zero = np.zeros(3)
        
        # Test Euclidean distance
        euclidean = KNNRecommender.euclidean_distance(a, b)
        expected_euclidean = np.sqrt(27)  # sqrt((4-1)^2 + (5-2)^2 + (6-3)^2)
        assert abs(euclidean - expected_euclidean) < 0.01, f"Euclidean: got {euclidean}, expected {expected_euclidean}"
        
        # Test Cosine distance
        cosine = KNNRecommender.cosine_distance(a, b)
        expected_cosine = 1 - (32 / (np.sqrt(14) * np.sqrt(77)))  # 1 - cos_similarity
        assert abs(cosine - expected_cosine) < 0.01, f"Cosine: got {cosine}, expected {expected_cosine}"
        
        # Test edge case: zero vector
        cosine_zero = KNNRecommender.cosine_distance(a, zero)
        assert cosine_zero == 1.0, f"Cosine with zero vector should be 1.0, got {cosine_zero}"
        
        print("✅ Distance functions work correctly")
        passed_tests += 1
    except Exception as e:
        print(f"❌ Distance functions failed: {e}")
    finally:
        total_tests += 1
    
    # Test 4: KNNRecommender initialization and fit
    print("\n[TEST 4] Testing KNNRecommender initialization...")
    try:
        # Create test data
        test_df = pd.DataFrame({
            'id': ['track1', 'track2', 'track3', 'track4', 'track5'],
            'energy': [0.5, 0.6, 0.7, 0.8, 0.9],
            'valence': [0.4, 0.5, 0.6, 0.7, 0.8],
            'tempo': [120, 125, 130, 135, 140]
        })
        
        features = ['energy', 'valence', 'tempo']
        
        # Initialize and fit
        recommender = KNNRecommender(k=2)
        recommender.fit(test_df, features)
        
        # Check that fit worked
        assert recommender.features_matrix is not None, "Features matrix not created"
        assert recommender.features_matrix.shape == (5, 3), f"Wrong shape: {recommender.features_matrix.shape}"
        assert len(recommender.track_id_to_index) == 5, "Track index mapping wrong"
        
        print("✅ KNNRecommender initialization works")
        passed_tests += 1
    except Exception as e:
        print(f"❌ KNNRecommender initialization failed: {e}")
    finally:
        total_tests += 1
    
    # Test 5: find_neighbors method
    print("\n[TEST 5] Testing find_neighbors...")
    try:
        # Use the recommender from previous test
        neighbors = recommender.find_neighbors('track1', n_neighbors=2, distance_metric='euclidean')
        
        # Check return format
        assert isinstance(neighbors, list), "find_neighbors should return a list"
        assert len(neighbors) == 2, f"Should return 2 neighbors, got {len(neighbors)}"
        assert all(isinstance(n, tuple) and len(n) == 2 for n in neighbors), "Should return list of (distance, id) tuples"
        
        # Check that distances are sorted
        distances = [d for d, _ in neighbors]
        assert distances == sorted(distances), "Neighbors not sorted by distance"
        
        # Check that query track is not in neighbors
        neighbor_ids = [tid for _, tid in neighbors]
        assert 'track1' not in neighbor_ids, "Query track should not be in neighbors"
        
        print("✅ find_neighbors works correctly")
        passed_tests += 1
    except Exception as e:
        print(f"❌ find_neighbors failed: {e}")
    finally:
        total_tests += 1
    
    # Test 6: recommend method
    print("\n[TEST 6] Testing recommend method...")
    try:
        recommendations = recommender.recommend('track1', n_recommendations=2, distance_metric='cosine')
        
        # Check return type
        assert isinstance(recommendations, pd.DataFrame), "recommend should return DataFrame"
        assert len(recommendations) == 2, f"Should return 2 recommendations, got {len(recommendations)}"
        assert 'distance' in recommendations.columns, "DataFrame should have 'distance' column"
        assert 'id' in recommendations.columns, "DataFrame should have 'id' column"
        
        # Check sorting
        distances = recommendations['distance'].values
        assert all(distances[i] <= distances[i+1] for i in range(len(distances)-1)), "Not sorted by distance"
        
        print("✅ recommend method works correctly")
        passed_tests += 1
    except Exception as e:
        print(f"❌ recommend method failed: {e}")
    finally:
        total_tests += 1
    
    # Test 7: API integration function
    print("\n[TEST 7] Testing API integration...")
    try:
        # This test requires the actual data file
        if not os.path.exists('data/item_profile.csv'):
            print("⚠️ Skipping API test - data/item_profile.csv not found")
            print("   This test requires the actual dataset")
        else:
            # Load a sample track ID
            df = pd.read_csv('data/item_profile.csv', dtype={'id': str})
            sample_id = df.iloc[0]['id']
            
            # Test the API function
            result = get_recommendations_for_api(sample_id, k=3, metric='cosine')
            
            assert isinstance(result, dict), "API should return a dictionary"
            assert len(result) == 3, f"Should return 3 recommendations, got {len(result)}"
            
            # Check format of each recommendation
            for track_id, info in result.items():
                assert 'distance' in info, "Each recommendation should have 'distance'"
                assert isinstance(info['distance'], (int, float)), "Distance should be numeric"
            
            print("✅ API integration works correctly")
            passed_tests += 1
    except Exception as e:
        print(f"❌ API integration failed: {e}")
    finally:
        total_tests += 1
    
    # Final summary
    print("\n" + "="*60)
    print(f"TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 CONGRATULATIONS! All tests passed!")
        print("Your implementation is ready for the web application.")
        print("\nNext steps:")
        print("1. Run: python app.py")
        print("2. Open: http://127.0.0.1:5002")
        print("3. Test your recommender in the web interface!")
    else:
        print(f"\n⚠️ {total_tests - passed_tests} test(s) failed.")
        print("Please review the errors above and fix your implementation.")
    
    print("="*60)
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)