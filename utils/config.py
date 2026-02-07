"""
Configuration settings for the Flask application.
All paths are now resolved from the project's root directory.
"""
import os
from dotenv import load_dotenv

class Config:
    """Base configuration class with shared settings."""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'a-secure-dev-secret-key')
    
    # --- IMPORTANT ---
    # The YouTube Data API v3 key is read from the environment.
    # Create a .env file at the project root or export it in your shell.
    # Example .env line: YOUTUBE_API_KEY=your_real_key

    # --- Path Resolution Update ---
    # `__file__` is in the `utils` directory. Go up one level to get the project root.
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Load environment variables from .env in the project root (if present)
    load_dotenv(os.path.join(BASE_DIR, '.env'))
    
    # Now read the API key from the environment (no hardcoded default)
    YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY')
    
    # These folders are expected to be at the root of the project
    STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    # CORS settings
    CORS_HEADERS = 'Content-Type'

class DevelopmentConfig(Config):
    """Configuration for local development."""
    DEBUG = True
    ENV = 'development'

class ProductionConfig(Config):
    """Configuration for production environments."""
    DEBUG = False
    ENV = 'production'
    
    def __init__(self):
        super().__init__()
        self.SECRET_KEY = os.environ.get('SECRET_KEY')
        if not self.SECRET_KEY:
            raise ValueError("No SECRET_KEY set for production application")
        if not self.YOUTUBE_API_KEY:
            raise ValueError("No YOUTUBE_API_KEY set for production application")
    
    def __init__(self):
        super().__init__()
        self.SECRET_KEY = os.environ.get('SECRET_KEY')
        if not self.SECRET_KEY:
            raise ValueError("No SECRET_KEY set for production application")
        if not self.YOUTUBE_API_KEY or self.YOUTUBE_API_KEY == "YOUR_YOUTUBE_API_KEY_HERE":
            raise ValueError("No YOUTUBE_API_KEY set for production application")

class TestingConfig(Config):
    """Configuration for running tests."""
    TESTING = True
    DEBUG = True

# Dictionary to map environment names to configuration classes
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name: str = None) -> Config:
    """Factory function to get a configuration object based on the environment."""
    if config_name is None:
        # Default to 'development' if FLASK_ENV is not set
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    config_class = config_map.get(config_name.lower(), config_map['default'])
    return config_class()
