"""
Configuration Module

This module loads and provides access to all configuration values from
environment variables. It uses a .env file for local development.

All configuration is centralized here for easy management.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# Google Cloud Configuration
# ============================================================================

# Google Cloud Project ID (required for all Google Cloud services)
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")

# Google Cloud Location/Region (default: us-central1)
# This determines which regional endpoint to use
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# ============================================================================
# Language Tutor Configuration
# ============================================================================

# Default language for the tutor (can be overridden by frontend selection)
# Options: Spanish, German, French, Telugu, etc.
TUTOR_LANGUAGE = os.getenv("TUTOR_LANGUAGE", "Spanish")

# ============================================================================
# API Server Configuration
# ============================================================================

# Host to bind the server to (0.0.0.0 = all interfaces)
API_HOST = os.getenv("API_HOST", "0.0.0.0")

# Port to run the server on
API_PORT = int(os.getenv("API_PORT", "8000"))
