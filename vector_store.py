import faiss
import numpy as np
import os
from config import FAISS_INDEX_PATH
import logging
from database import search_open_cases

class VectorStore:
    def __init__(self, embedding_dim=512):
        """
        Initialize FAISS vector store with comprehensive error handling
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Ensure a consistent index type
        self.embedding_dim = embedding_dim
        self.index = None
        
        # Initialize the index
        self.create_or_load_index()

    def create_or_load_index(self):
        """
        Create a new index or load an existing one
        """
        try:
            # Create a new index if file doesn't exist
            if not os.path.exists(FAISS_INDEX_PATH):
                self.logger.info("Creating new FAISS index")
                # Use IndexFlatL2 for Euclidean distance (better for facial embeddings)
                base_index = faiss.IndexFlatL2(self.embedding_dim)
                self.index = faiss.IndexIDMap(base_index)
                self.save_index()
            else:
                # Load existing index
                self.logger.info("Loading existing FAISS index")
                self.index = faiss.read_index(FAISS_INDEX_PATH)
                
                # Verify index
                self.logger.info(f"Loaded index dimension: {self.index.d}")
                self.logger.info(f"Total vectors in index: {self.index.ntotal}")
        
        except Exception as e:
            self.logger.error(f"Error creating/loading index: {e}")
            # Fallback to creating a new index
            base_index = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIDMap(base_index)

    def add_embedding(self, embedding, embedding_id):
        """
        Add embedding to vector store with comprehensive checks
        """
        try:
            # Ensure embedding is correct shape and type
            embedding = np.array([embedding], dtype=np.float32)
            embedding_id = np.array([embedding_id], dtype=np.int64)
            
            # Verify embedding dimension
            if embedding.shape[1] != self.embedding_dim:
                self.logger.warning(f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {embedding.shape[1]}")
                return False
            
            # Normalize embedding for better similarity search
            embedding = embedding / np.linalg.norm(embedding, axis=1)[:, np.newaxis]
            
            # Add embedding
            self.index.add_with_ids(embedding, embedding_id)
            
            # Save updated index
            self.save_index()
            
            self.logger.info(f"Added embedding with ID {embedding_id[0]}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error adding embedding: {e}")
            return False

    def search_embeddings(self, embedding, top_k=5, similarity_threshold=0.7):
        """
        Enhanced search with improved similarity calculation
        """
        try:
            # Normalize query embedding
            embedding = np.array([embedding], dtype=np.float32)
            embedding = embedding / np.linalg.norm(embedding, axis=1)[:, np.newaxis]
            
            # Perform search
            D, I = self.index.search(embedding, top_k)
            
            # Detailed logging of search results
            self.logger.info("Search Results:")
            for dist, idx in zip(D[0], I[0]):
                # Convert distance to similarity (for L2 distance)
                similarity = 1 / (1 + dist)
                self.logger.info(f"Embedding ID: {idx}, Distance: {dist}, Similarity: {similarity}")
            
            # Filter matches based on similarity
            matches = []
            for sim, embedding_id in zip(D[0], I[0]):
                # Convert distance to similarity
                similarity = 1 / (1 + sim)
                if similarity > similarity_threshold and embedding_id != -1:
                    matches.append(embedding_id)
            
            # Return matches or -1 if no matches
            return matches if matches else [-1]
        
        except Exception as e:
            self.logger.error(f"Error searching embeddings: {e}")
            return [-1]

    def save_index(self, filename=FAISS_INDEX_PATH):
        """
        Save FAISS index with robust error handling
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Save index
            faiss.write_index(self.index, filename)
            self.logger.info(f"Index saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving index: {e}")

# Utility functions
def add_embedding_to_faiss(embedding, embedding_id):
    """
    Convenience function to add embedding with error handling
    """
    try:
        vector_store = VectorStore()
        return vector_store.add_embedding(embedding, embedding_id)
    except Exception as e:
        logging.error(f"Error adding embedding: {e}")
        return False

def search_faiss(embedding, top_k=5, similarity_threshold=0.7):
    """
    Convenience function to search embeddings
    """
    try:
        vector_store = VectorStore()
        return vector_store.search_embeddings(embedding, top_k, similarity_threshold)
    except Exception as e:
        logging.error(f"Error searching embeddings: {e}")
        return [-1]