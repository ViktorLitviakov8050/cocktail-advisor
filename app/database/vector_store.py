import faiss
import numpy as np
from typing import List, Dict
import pandas as pd

class VectorStore:
    def __init__(self):
        self.dimension = 384  # Default dimension for embeddings
        self.index = faiss.IndexFlatL2(self.dimension)
        self.cocktail_data: Dict[int, Dict] = {}
        self.favorite_ingredients: List[str] = []
        
    def add_cocktail(self, cocktail_id: int, embedding: np.ndarray, metadata: Dict):
        """Add a cocktail to the vector store"""
        self.index.add(np.array([embedding]))
        self.cocktail_data[cocktail_id] = metadata
        
    def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar cocktails"""
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return [self.cocktail_data[int(idx)] for idx in indices[0]]
    
    def add_favorite_ingredient(self, ingredient: str):
        """Store user's favorite ingredient"""
        if ingredient not in self.favorite_ingredients:
            self.favorite_ingredients.append(ingredient)
            
    def get_favorite_ingredients(self) -> List[str]:
        """Get user's favorite ingredients"""
        return self.favorite_ingredients 