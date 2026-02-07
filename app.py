"""
Main Flask Application and API endpoints.

This file sets up the Flask server, configures CORS, defines all the API routes
for searching, getting recommendations, and fetching audio, and handles errors.
It now imports helper functions from the `utils` directory.
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import traceback

# --- Refactored Imports ---
# Import helper modules from the 'utils' directory.
from utils import api_helpers as hp
from utils.config import get_config

# --- App Initialization ---
# The static_folder points to the directory where your HTML/CSS/JS files are.
# In a typical setup, your knn-music-recommender.html would go in a 'static' folder.
app = Flask(__name__, static_folder='static', static_url_path='')

# Set environment to development by default if not set
if 'FLASK_ENV' not in os.environ:
    os.environ['FLASK_ENV'] = 'development'

# Load configuration from utils/config.py
config_obj = get_config()
app.config.from_object(config_obj)

# Enable Cross-Origin Resource Sharing for API routes
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- Error Handling ---
class APIError(Exception):
    """Custom exception class for API-specific errors."""
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

@app.errorhandler(APIError)
def handle_api_error(error):
    """Handles custom API errors with a JSON response."""
    response = {'success': False, 'error': error.message}
    return jsonify(response), error.status_code

@app.errorhandler(404)
def not_found(error):
    """Handles 404 Not Found errors."""
    return jsonify({'success': False, 'error': 'Not Found'}), 404

@app.errorhandler(Exception)
def handle_generic_error(e):
    """Handles all other unexpected exceptions."""
    traceback.print_exc()
    return jsonify({'success': False, 'error': 'An unexpected internal server error occurred.'}), 500

# --- API Routes ---
API_PREFIX = '/api'

@app.route(f'{API_PREFIX}/featured')
def get_featured():
    """Endpoint to get a list of featured tracks with valid album art."""
    featured_tracks = hp.get_featured_tracks(limit=4)
    if not featured_tracks:
        # Fallback in case no valid tracks can be found
        return jsonify({'success': False, 'error': 'Could not retrieve featured tracks.'}), 500
    return jsonify({'success': True, 'data': featured_tracks})

@app.route(f'{API_PREFIX}/search')
def search_music():
    """Endpoint to search for music by song and/or artist name."""
    song_input = request.args.get('song_name', '')
    artist_input = request.args.get('artist_name', '')
    
    if not song_input and not artist_input:
        raise APIError("A search query ('song_name' or 'artist_name') is required.", 400)
        
    results = hp.get_data_from_query(song_input, artist_input)
    return jsonify({'success': True, 'data': results})

@app.route(f'{API_PREFIX}/recommend/track/<string:track_id>')
def recommend_by_track(track_id: str):
    """Endpoint to get recommendations based on a single seed track."""
    if not track_id:
        raise APIError("Track ID cannot be empty.", 400)
    
    try:
        k = int(request.args.get('k', 10))
        metric = request.args.get('metric', 'cosine')
        features_str = request.args.get('features', '')
        # Filter out empty strings that can result from splitting an empty string
        selected_features = [f for f in features_str.split(',') if f] or None
        
        result = hp.get_recommendations(
            way="fromTrackID",
            k=k,
            track_id=track_id,
            selected_features=selected_features,
            distance_metric=metric
        )
        return jsonify(result)
        
    except ValueError:
        raise APIError("Invalid parameter format. 'k' must be an integer.", 400)

@app.route(f'{API_PREFIX}/recommend/profile', methods=['POST'])
def recommend_from_profile():
    """Endpoint to get recommendations based on a custom feature profile."""
    data = request.get_json()
    if not data:
        raise APIError("Request body must be JSON.", 400)
        
    features = data.get('features')
    if not features:
        raise APIError("Missing 'features' object in request body.", 400)
    
    # Ensure features are in a consistent order for the model
    feature_order = ['energy', 'danceability', 'acousticness', 'valence', 'tempo', 'instrumentalness']
    feature_vector = [features.get(f, 0.5) for f in feature_order]

    result = hp.get_recommendations(
        way="fromProfile",
        k=int(data.get('k', 10)),
        query_vector=feature_vector,
        selected_features=data.get('selected_features'),
        distance_metric=data.get('distance_metric', 'cosine')
    )
    return jsonify(result)

@app.route(f'{API_PREFIX}/getmp3url')
def get_mp3_url_route():
    """Endpoint to get a playable audio stream URL from YouTube."""
    song_id = request.args.get('songid')
    if not song_id:
        raise APIError("'songid' parameter is required.", 400)
        
    track_info = hp.get_information_from_id(song_id)
    if not track_info:
        raise APIError(f"Track with ID '{song_id}' not found.", 404)
        
    audio_url = hp.get_mp3_url(track_info.get('song'), track_info.get('artist'))
    
    if audio_url:
        return jsonify({'success': True, 'url': audio_url})
    else:
        raise APIError("Could not find a playable audio stream for this track.", 404)

# --- Static File Serving ---
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """
    Serves the main HTML file for the frontend and other static assets.
    The frontend app is expected to be in the 'static' directory.
    """
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        # Serves the main entry point of the web app.
        return send_from_directory(app.static_folder, 'knn-music-recommender.html')

# --- Main Execution ---
if __name__ == "__main__":
    # Use debug=True only for development
    app.run(host='0.0.0.0', port=5002, debug=app.config.get('DEBUG', False))
