import torch
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
import logging

class FaceEmbedding:
    def __init__(self, model_type='vggface2'):
        """
        Initialize face embedding model with device support and consistent preprocessing
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.model = InceptionResnetV1(pretrained=model_type).eval().to(self.device)
            logging.info(f"Face embedding model loaded on {self.device}")
        except Exception as e:
            logging.error(f"Model loading error: {e}")
            raise

    def extract_embedding(self, face):
        """
        Enhanced embedding extraction with robust preprocessing
        """
        try:
            # Validate input
            if face is None or face.size == 0:
                logging.error("Invalid face image")
                return None
            
            # Ensure face is the right format
            if face.dtype != np.float32:
                face = face.astype(np.float32)
            
            # Consistent preprocessing
            # Resize to exactly 160x160
            face_resized = cv2.resize(face, (160, 160))
            
            # Normalize between -1 and 1 (typical for face recognition models)
            face_normalized = (face_resized / 255.0 - 0.5) * 2.0
            
            # Convert to tensor with correct dimensions
            face_tensor = torch.tensor(face_normalized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model(face_tensor)
            
            # Move back to CPU and convert to numpy
            embedding_np = embedding.cpu().numpy().flatten()
            
            # L2 normalization
            embedding_np /= np.linalg.norm(embedding_np)
            
            return embedding_np
        
        except Exception as e:
            logging.error(f"Embedding extraction error: {e}")
            return None

def extract_embedding(face):
    """
    Convenience function with comprehensive error handling
    """
    try:
        # Validate input
        if face is None or face.size == 0:
            logging.error("Invalid face image for embedding")
            return None
        
        embedder = FaceEmbedding()
        return embedder.extract_embedding(face)
    except Exception as e:
        logging.error(f"Embedding extraction failed: {e}")
        return None