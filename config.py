import os

# Database Configuration
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "your-password",  # Recommendation: Use environment variables for sensitive info
    "database": "child_safety"
}

# Paths Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "data", "embeddings", "faiss_index.bin")
IMAGE_STORAGE_PATH = os.path.join(BASE_DIR, "data", "images")
YOLO_FACE_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n-face.pt")

# Ensure directories exist
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
os.makedirs(IMAGE_STORAGE_PATH, exist_ok=True)

# Additional configuration parameters
EMBEDDING_DIM = 512  # Dimension of facial embeddings
SIMILARITY_THRESHOLD = 0.7  # Default similarity threshold for face matching
MAX_MATCHES = 5  # Maximum number of matches to return