import json
import os
import numpy as np
from typing import Dict, Tuple, Optional, Union, List
from sklearn.metrics.pairwise import cosine_similarity

class QueryCache:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.vectors_file = os.path.join(cache_dir, "query_vectors.npy")
        self.metadata_file = os.path.join(cache_dir, "query_metadata.json")
        self.cache: Dict[str, dict] = {}
        self.vectors = []
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialize or load existing cache."""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load metadata
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.cache = json.load(f)
        
        # Load vectors
        if os.path.exists(self.vectors_file):
            self.vectors = np.load(self.vectors_file)
    
    def _save_cache(self):
        """Save cache to disk."""
        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
        
        # Save vectors
        if isinstance(self.vectors, np.ndarray) and self.vectors.size > 0:
            np.save(self.vectors_file, self.vectors)
    
    def add_query(self, query: str, filtered_query: str, embedding: Union[np.ndarray, List]):
        """Add a query and its embedding to the cache."""
        query_id = str(len(self.cache))
        
        # Store metadata
        self.cache[query_id] = {
            "original_query": query,
            "filtered_query": filtered_query,
            "timestamp": str(np.datetime64('now'))
        }
        
        # Convert embedding to numpy array if it's a list and ensure it's 1D
        if isinstance(embedding, list):
            embedding_array = np.array(embedding).reshape(1, -1)
        else:
            embedding_array = np.array(embedding).reshape(1, -1)
            
        # Initialize vectors as numpy array if empty
        if not isinstance(self.vectors, np.ndarray) or len(self.vectors) == 0:
            self.vectors = embedding_array
        else:
            self.vectors = np.vstack([self.vectors, embedding_array])
        
        self._save_cache()
    
    def find_similar_query(self, query_embedding: Union[np.ndarray, List], similarity_threshold: float = 0.9) -> Optional[Tuple[dict, float]]:
        """Find most similar cached query if it exists."""
        if not self.vectors:
            return None
            
        # Convert query_embedding to numpy array if it's a list
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        
        # Convert vectors to numpy array if it's a list
        if isinstance(self.vectors, list):
            vectors_array = np.array(self.vectors)
        else:
            vectors_array = self.vectors
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding.reshape(1, -1), vectors_array)
        max_similarity = similarities.max()
        
        if max_similarity >= similarity_threshold:
            most_similar_idx = similarities.argmax()
            query_id = str(most_similar_idx)
            return self.cache[query_id], max_similarity
        
        return None
    
    def get_all_queries(self) -> Dict[str, dict]:
        """Get all cached queries."""
        return self.cache
