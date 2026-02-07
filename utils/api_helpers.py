"""
Helper functions for the Flask API.
This version uses the official Google API for YouTube search for maximum reliability
and ensures featured tracks have valid album art.
"""
import pandas as pd
import numpy as np
import requests
import hashlib
import os
from googleapiclient.discovery import build
from typing import Dict, List, Optional, Any
from urllib.parse import quote_plus

# --- Refactored Imports ---
# Import other local utility modules
from .config import get_config

# --- Configuration & Data Loading ---
CONFIG = get_config()
YOUTUBE_API_KEY = CONFIG.YOUTUBE_API_KEY
MERGED_FILE = pd.DataFrame()
ITEM_PROFILE = pd.DataFrame()

try:
    # Use the robust path from the config object
    merged_path = os.path.join(CONFIG.DATA_DIR, 'mergedFile.csv')
    profile_path = os.path.join(CONFIG.DATA_DIR, 'item_profile.csv')
    
    MERGED_FILE = pd.read_csv(merged_path, dtype={'id': str})
    ITEM_PROFILE = MERGED_FILE.copy() #pd.read_csv(profile_path, dtype={'id': str})
    
    # Pre-process for faster searching
    MERGED_FILE['song_lower'] = MERGED_FILE['song'].str.lower()
    MERGED_FILE['artist_lower'] = MERGED_FILE['artist'].str.lower()
except FileNotFoundError as e:
    print(f"FATAL: Data file not found - {e}. The application cannot run without it.")
    # Application will not function correctly, but this prevents a crash on import

# --- Student Implementation Support ---
STUDENT_IMPLEMENTATION_AVAILABLE = False
student_recommender = None

def check_student_implementation():
    """Check if student has completed their implementation."""
    global STUDENT_IMPLEMENTATION_AVAILABLE, student_recommender
    try:
        from .student_adapter import KNNRecommender as StudentKNN
        
        # Try to initialize with a small test
        test_df = ITEM_PROFILE.head(10) if not ITEM_PROFILE.empty else pd.DataFrame()
        if not test_df.empty:
            audio_features = ['energy', 'danceability', 'acousticness', 'valence', 
                            'tempo', 'instrumentalness', 'loudness', 'liveness', 'speechiness']
            
            # Test that student implementation has required methods
            test_recommender = StudentKNN(k=5)
            test_recommender.fit(test_df, audio_features)
            
            # If we get here, student implementation exists
            # Now create the real recommender with full data
            student_recommender = StudentKNN(k=10)
            student_recommender.fit(ITEM_PROFILE, audio_features)
            STUDENT_IMPLEMENTATION_AVAILABLE = True
            print("✅ Student k-NN implementation loaded successfully!")
            return True
            
    except Exception as e:
        print(f"⚠️ Student implementation not ready: {e}")
        print("Complete your implementation in utils/student_adapter.py")
        return False

# Run check on module load
check_student_implementation()

# --- Caching ---
# Simple in-memory caches to reduce redundant API calls
ALBUM_ART_CACHE = {}
YOUTUBE_LINK_CACHE = {}

# --- Album Art Fetching ---
def fetch_album_art_realtime(song_name: str, artist_name: str) -> str:
    """Fetches album art from the iTunes API, with caching."""
    if not song_name or not artist_name:
        return generate_placeholder_art("Unknown", "Track")
        
    cache_key = f"{artist_name.lower()}|||{song_name.lower()}"
    if cache_key in ALBUM_ART_CACHE:
        return ALBUM_ART_CACHE[cache_key]

    try:
        query = f"{song_name} {artist_name}"
        url = f"https://itunes.apple.com/search?term={quote_plus(query)}&media=music&entity=song&limit=1"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get('results'):
            # Get high-resolution artwork
            artwork_url = data['results'][0].get('artworkUrl100', '').replace('100x100', '600x600')
            if artwork_url:
                ALBUM_ART_CACHE[cache_key] = artwork_url
                return artwork_url
    except requests.RequestException as e:
        print(f"iTunes API request failed for '{song_name}': {e}")

    # Fallback if iTunes search fails or returns no art
    fallback_art = generate_placeholder_art(song_name, artist_name)
    ALBUM_ART_CACHE[cache_key] = fallback_art
    return fallback_art

def generate_placeholder_art(song_name: str, artist_name: str) -> str:
    """Generates a consistent, colored placeholder image URL from placehold.co."""
    text = f"{song_name} {artist_name}".lower()
    # Use a hash to create a consistent color for each track
    color = hashlib.md5(text.encode()).hexdigest()[:6]
    display_text = quote_plus(song_name[:20] if song_name else "Music")
    return f"https://placehold.co/600x600/{color}/ffffff?text={display_text}"

# --- Data Retrieval and Formatting ---
def get_information_from_id(track_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves all details for a specific track ID, fetching album art if missing."""
    if MERGED_FILE.empty: return None
    record = MERGED_FILE[MERGED_FILE["id"] == track_id]
    if record.empty: return None
    
    track_data = record.iloc[0].to_dict()
    # Convert numpy types to native Python types for JSON serialization
    for key, value in track_data.items():
        if isinstance(value, np.generic):
            track_data[key] = value.item()
            
    # If album art is missing or is a placeholder, try to fetch it
    if pd.isna(track_data.get('albumart')) or not track_data.get('albumart'):
         track_data['albumart'] = fetch_album_art_realtime(track_data.get('song'), track_data.get('artist'))
         
    return track_data

def get_featured_tracks(limit: int = 4) -> Dict[str, Dict[str, Any]]:
    """
    **OPTIMIZED LOGIC**
    Selects a random variety of tracks, ensuring they have valid album art.
    """
    if MERGED_FILE.empty: return {}
    
    featured_tracks = {}
    seen_artists = set()
    
    # Filter for tracks that are more likely to be valid (have an album name)
    pool = MERGED_FILE.dropna(subset=['album', 'artist_lower', 'song', 'artist']).copy()
    
    # Add a safety break to prevent infinite loops if no valid art can be found
    max_attempts = limit * 10 
    attempts = 0

    while len(featured_tracks) < limit and attempts < max_attempts and not pool.empty:
        attempts += 1
        
        # Sample one track to check
        sample_track = pool.sample(n=1).iloc[0]
        artist = sample_track['artist_lower']

        if artist in seen_artists:
            continue # Ensure artist variety

        # Check for valid album art
        track_info = get_information_from_id(sample_track['id'])
        if track_info and 'placehold.co' not in track_info.get('albumart', ''):
            featured_tracks[track_info['id']] = track_info
            seen_artists.add(artist)
            # Remove from pool to avoid re-sampling
            pool.drop(sample_track.name, inplace=True)

    return featured_tracks

# --- Search Functionality ---
def get_data_from_query(song_input: str, artist_input: str) -> Dict[str, Dict[str, Any]]:
    """Performs a fast search on the dataset using pre-lowercased columns."""
    if MERGED_FILE.empty: return {}
    
    song_q, artist_q = song_input.lower().strip(), artist_input.lower().strip()
    
    # Build a filter mask based on inputs
    mask = pd.Series(True, index=MERGED_FILE.index)
    if song_q:
        mask &= MERGED_FILE['song_lower'].str.contains(song_q, na=False)
    if artist_q:
        mask &= MERGED_FILE['artist_lower'].str.contains(artist_q, na=False)
        
    results_df = MERGED_FILE[mask]

    # Sort by popularity and get the top 10 results
    top_results = results_df.nlargest(10, 'popularity') if 'popularity' in results_df.columns else results_df.head(10)
    
    # Get full info for each result
    return {row['id']: get_information_from_id(row['id']) for _, row in top_results.iterrows()}

# --- Recommendation Engine Interface ---
def get_recommendations(**kwargs) -> Dict[str, Any]:
    """
    Get recommendations using student implementation if available.
    Shows clear feedback if not implemented.
    """
    
    # Always try student implementation first
    if kwargs.get('way') == 'fromTrackID':
        if STUDENT_IMPLEMENTATION_AVAILABLE:
            try:
                track_id = kwargs.get('track_id')
                k = kwargs.get('k', 10)
                metric = kwargs.get('distance_metric', 'cosine').lower()
                
                # Use student implementation
                recs_df = student_recommender.recommend(
                    track_id=track_id,
                    n_recommendations=k,
                    distance_metric=metric
                )
                
                # Convert to expected format
                response_data = {}
                for _, row in recs_df.iterrows():
                    track_info = get_information_from_id(row['id'])
                    if track_info:
                        response_data[row['id']] = track_info
                        # Convert distance to similarity score
                        response_data[row['id']]['similarity_score'] = 1 / (1 + row.get('distance', 0))
                
                if response_data:
                    return {'success': True, 'data': response_data}
                else:
                    return {
                        'success': False, 
                        'error': 'No recommendations found. Check your implementation.',
                        'data': {}
                    }
                    
            except Exception as e:
                print(f"Error in student implementation: {e}")
                return {
                    'success': False,
                    'error': f'Error in your k-NN implementation: {str(e)}',
                    'data': {}
                }
        else:
            # Student hasn't implemented yet - show helpful message
            return {
                'success': False,
                'error': 'k-NN not implemented yet. Complete your implementation in student_adapter.py',
                'data': {},
                'hint': 'Make sure you have implemented the KNNRecommender class with fit() and recommend() methods.'
            }
    
    # For other ways (fromProfile), also require student implementation
    return {
        'success': False,
        'error': 'This feature requires completing the k-NN implementation.',
        'data': {}
    }

# --- YouTube Audio Fetching (Using Official Google API) ---
def get_youtube_link(song_name: str, artist_name: str) -> Optional[str]:
    """Searches YouTube using the official Google API and returns the video URL."""
    cache_key = f"{artist_name.lower()}|||{song_name.lower()}"
    if cache_key in YOUTUBE_LINK_CACHE:
        return YOUTUBE_LINK_CACHE[cache_key]

    if not YOUTUBE_API_KEY or YOUTUBE_API_KEY == "YOUR_YOUTUBE_API_KEY_HERE":
        print("ERROR: YouTube API key is not configured in utils/config.py")
        return None
        
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        query = f"{song_name} {artist_name} official audio"
        request = youtube.search().list(
            q=query, 
            part='snippet', 
            type='video', 
            maxResults=1, 
            videoCategoryId='10' # Music category
        )
        response = request.execute()
        
        if response.get('items'):
            video_id = response['items'][0]['id']['videoId']
            url = f"https://www.youtube.com/watch?v={video_id}"
            YOUTUBE_LINK_CACHE[cache_key] = url
            return url
    except Exception as e:
        print(f"YouTube API Error: {e}")
        
    return None

def get_mp3_url(song_name: str, artist_name: str) -> Optional[str]:
    """Finds a YouTube video and extracts the direct audio stream URL using yt-dlp."""
    youtube_link = get_youtube_link(song_name, artist_name)
    if not youtube_link: return None
    
    # **FIX 2:** Added 'source_address' to force IPv4, which can resolve some
    # connection and signature extraction issues with YouTube's servers.
    # Prefer direct file streams (e.g., m4a/webm) over HLS (m3u8) to ensure
    # playback works in Chrome without HLS support. Fallback to HLS only if no
    # direct stream is available.
    ydl_opts = {
        # Prefer m4a, then any bestaudio with an actual audio codec
        'format': 'bestaudio[ext=m4a]/bestaudio[acodec!=none]/bestaudio',
        'quiet': True,
        'noplaylist': True,
        'source_address': '0.0.0.0',  # Force IPv4 connection
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
    }
    
    try:
        # yt-dlp is now imported on-demand to avoid a hard dependency if not used
        import yt_dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_link, download=False)
            formats = info.get('formats', [])

            # First, try to find a direct file stream (non-HLS) with an audio codec
            candidates = []
            for f in formats:
                if f.get('acodec') == 'none':
                    continue
                proto = (f.get('protocol') or '').lower()
                # Exclude HLS/DASH manifest-based protocols on the first pass
                if 'm3u8' in proto or 'http_dash_segments' in proto or 'dash' in proto:
                    continue
                url = f.get('url')
                if not url:
                    continue
                abr = f.get('abr') or 0
                ext = (f.get('ext') or '').lower()
                # Prefer m4a > webm > others by boosting their score
                score = abr
                if ext == 'm4a':
                    score += 1000
                elif ext == 'webm':
                    score += 500
                candidates.append((score, url))

            if candidates:
                # Return the highest-scoring direct stream
                candidates.sort(key=lambda x: x[0], reverse=True)
                return candidates[0][1]

            # Fallback: if no direct streams found, allow HLS (may work in Safari)
            for f in formats:
                if f.get('acodec') != 'none' and 'm3u8' in ((f.get('protocol') or '').lower()):
                    return f.get('url')
            
            return None
    except Exception as e:
        print(f"yt-dlp error for '{youtube_link}': {e}")
        
    return None
