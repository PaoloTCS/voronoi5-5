import os
import numpy as np
from typing import List, Dict, Any, Optional, Union
import requests
import json
import time
from pathlib import Path

# Simulated embeddings for development if API key not available
USE_SIMULATED_EMBEDDINGS = True
EMBEDDING_DIMENSION = 384  # Default dimension for simulated embeddings

class EmbeddingService:
    """
    Service for generating text embeddings using AI models.
    
    This service can use different embedding models:
    1. OpenAI's text-embedding models
    2. Local models via sentence-transformers (if installed)
    3. Simulated embeddings for development (random vectors)
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "text-embedding-ada-002"):
        """
        Initialize the embedding service.
        
        Args:
            api_key: API key for embedding service (e.g. OpenAI)
            model_name: Name of the embedding model to use
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name
        self.cache = {}  # Simple cache for embeddings
        
        # Try to load sentence-transformers if available
        self.sentence_transformer = None
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_transformer = SentenceTransformer(model_name)
            print(f"Using sentence-transformers model: {model_name}")
        except ImportError:
            if not self.api_key and not USE_SIMULATED_EMBEDDINGS:
                print("Warning: Neither API key nor sentence-transformers available. Using simulated embeddings.")
    
    def _generate_simulated_embedding(self, text: str) -> np.ndarray:
        """
        Generate a simulated embedding vector for development purposes.
        Uses a hash of the text to create a deterministic but random-seeming vector.
        
        Args:
            text: Text to generate an embedding for
            
        Returns:
            A numpy array of the simulated embedding
        """
        # Use hash of text as seed for random generator to ensure deterministic behavior
        text_hash = hash(text) % (2**32)
        np.random.seed(text_hash)
        
        # Generate random vector
        embedding = np.random.randn(EMBEDDING_DIMENSION)
        
        # Normalize to unit length
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def _generate_openai_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding using the OpenAI API.
        
        Args:
            text: Text to generate an embedding for
            
        Returns:
            A numpy array of the embedding
        """
        if not self.api_key:
            raise ValueError("OpenAI API key is required for OpenAI embeddings")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "input": text,
            "model": self.model_name
        }
        
        # Try API call with backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://api.openai.com/v1/embeddings",
                    headers=headers,
                    data=json.dumps(data)
                )
                response.raise_for_status()
                result = response.json()
                embedding = np.array(result["data"][0]["embedding"])
                return embedding
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Error generating embedding: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise ValueError(f"Failed to generate embedding after {max_retries} attempts: {e}")
    
    def _generate_sentence_transformer_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding using the sentence-transformers library.
        
        Args:
            text: Text to generate an embedding for
            
        Returns:
            A numpy array of the embedding
        """
        if not self.sentence_transformer:
            raise ValueError("sentence-transformers is not available")
        
        embedding = self.sentence_transformer.encode(text, convert_to_numpy=True)
        return embedding
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get an embedding for the given text.
        
        Will try to use the available embedding methods in this order:
        1. Return from cache if available
        2. Use sentence-transformers if available
        3. Use OpenAI API if available
        4. Use simulated embeddings as fallback
        
        Args:
            text: Text to generate an embedding for
            
        Returns:
            A numpy array of the embedding
        """
        # Check cache first
        if text in self.cache:
            return self.cache[text]
        
        # Generate embedding based on available methods
        if self.sentence_transformer:
            embedding = self._generate_sentence_transformer_embedding(text)
        elif self.api_key:
            embedding = self._generate_openai_embedding(text)
        elif USE_SIMULATED_EMBEDDINGS:
            embedding = self._generate_simulated_embedding(text)
        else:
            raise ValueError("No embedding method available. Provide an API key or install sentence-transformers.")
        
        # Cache and return
        self.cache[text] = embedding
        return embedding
    
    def get_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of numpy arrays with embeddings
        """
        return [self.get_embedding(text) for text in texts]
    
    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity (float between -1 and 1)
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def get_most_similar(self, query: str, candidates: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Find the most similar texts to a query from a list of candidates.
        
        Args:
            query: Query text
            candidates: List of candidate texts to compare against
            top_k: Number of top results to return
            
        Returns:
            List of dicts with text and similarity score, sorted by similarity
        """
        query_emb = self.get_embedding(query)
        
        results = []
        for candidate in candidates:
            candidate_emb = self.get_embedding(candidate)
            similarity = np.dot(query_emb, candidate_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(candidate_emb))
            results.append({
                "text": candidate,
                "similarity": float(similarity)
            })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top-k
        return results[:top_k]
    
    def save_cache(self, filepath: str):
        """
        Save the embedding cache to a file.
        
        Args:
            filepath: Path to save the cache to
        """
        cache_dict = {k: v.tolist() for k, v in self.cache.items()}
        with open(filepath, 'w') as f:
            json.dump(cache_dict, f)
    
    def load_cache(self, filepath: str):
        """
        Load the embedding cache from a file.
        
        Args:
            filepath: Path to load the cache from
        """
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            cache_dict = json.load(f)
        
        self.cache = {k: np.array(v) for k, v in cache_dict.items()}
