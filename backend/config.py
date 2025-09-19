# app/config.py
import os
from dotenv import load_dotenv
load_dotenv()

VIDEO_SDK_API_KEY = os.getenv("VIDEO_SDK_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
ELEVENLABS_API = os.getenv("ELEVENLABS_API", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "FAQ_repo"
