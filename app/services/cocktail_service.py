from typing import List, Dict
import pandas as pd
from ..database.vector_store import VectorStore
from langchain.embeddings import OpenAIEmbeddings
import os

class CocktailService:
    def __init__(self):
        self.vector_store = VectorStore()
        self.embeddings = OpenAIEmbeddings()
        self.cocktails_df = self._load_cocktails_data()
        self._initialize_vector_store()
        
    def _load_cocktails_data(self) -> pd.DataFrame:
        """Load cocktails data from CSV file"""
        try:
            # Assuming the cocktails.csv is in the data directory
            df = pd.read_csv('data/cocktails.csv')
            return df
        except Exception as e:
            print(f"Error loading cocktails data: {e}")
            return pd.DataFrame()
            
    def _initialize_vector_store(self):
        """Initialize vector store with cocktail data"""
        try:
            for idx, row in self.cocktails_df.iterrows():
                # Create embedding for cocktail ingredients
                ingredients_text = " ".join(row['ingredients']) if isinstance(row['ingredients'], list) else str(row['ingredients'])
                embedding = self.embeddings.embed_query(ingredients_text)
                
                # Store cocktail data and embedding
                metadata = {
                    'name': row['name'],
                    'ingredients': row['ingredients'],
                    'instructions': row['instructions'] if 'instructions' in row else '',
                    'alcoholic': row['alcoholic'] if 'alcoholic' in row else True
                }
                self.vector_store.add_cocktail(idx, embedding, metadata)
        except Exception as e:
            print(f"Error initializing vector store: {e}")
    
    def get_favorite_ingredients(self) -> List[str]:
        """Get user's favorite ingredients from vector store"""
        return self.vector_store.get_favorite_ingredients()
        
    def add_favorite_ingredient(self, ingredient: str) -> Dict:
        """Add a favorite ingredient to vector store"""
        ingredient = ingredient.lower().strip()
        self.vector_store.add_favorite_ingredient(ingredient)
        return {"message": f"Added {ingredient} to favorites"}
        
    def search_cocktails_by_ingredient(self, ingredient: str, limit: int = 5) -> List[Dict]:
        """Search for cocktails containing specific ingredient"""
        ingredient = ingredient.lower().strip()
        matching_cocktails = []
        
        for cocktail in self.vector_store.cocktail_data.values():
            ingredients = [ing.lower() for ing in cocktail['ingredients']]
            if ingredient in ingredients:
                matching_cocktails.append(cocktail)
                if len(matching_cocktails) >= limit:
                    break
                    
        return matching_cocktails
        
    def get_similar_cocktails(self, cocktail_name: str, limit: int = 5) -> List[Dict]:
        """Find similar cocktails based on ingredients"""
        cocktail_name = cocktail_name.lower().strip()
        
        # Find the reference cocktail
        reference_cocktail = None
        for cocktail in self.vector_store.cocktail_data.values():
            if cocktail['name'].lower() == cocktail_name:
                reference_cocktail = cocktail
                break
                
        if not reference_cocktail:
            return []
            
        # Get embedding for reference cocktail
        ingredients_text = " ".join(reference_cocktail['ingredients'])
        query_embedding = self.embeddings.embed_query(ingredients_text)
        
        # Find similar cocktails
        return self.vector_store.search_similar(query_embedding, k=limit)
        
    def get_non_alcoholic_cocktails(self, limit: int = 5) -> List[Dict]:
        """Get non-alcoholic cocktails"""
        non_alcoholic = []
        
        for cocktail in self.vector_store.cocktail_data.values():
            if not cocktail['alcoholic']:
                non_alcoholic.append(cocktail)
                if len(non_alcoholic) >= limit:
                    break
                    
        return non_alcoholic 