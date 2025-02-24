from typing import List, Dict
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from ..utils.data_processor import process_cocktail_data, initialize_vector_store
import json
from langchain.schema import Document
from datetime import datetime

class CocktailService:
    _vector_store = None  # Class-level singleton

    def __init__(self):
        try:
            # Initialize OpenAI embeddings
            self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            
            # Initialize/load vector store only if not already created
            if CocktailService._vector_store is None:
                print("Creating vector store...")
                CocktailService._vector_store = self._initialize_vector_store()
            self.vector_store = CocktailService._vector_store
            
            # Load saved favorites
            self.favorites_file = "data/favorites.json"
            self.favorite_ingredients = self._load_favorites()
        except Exception as e:
            print(f"Error initializing CocktailService: {str(e)}")
            raise
        
    def _initialize_vector_store(self):
        """Initialize the vector store with cocktail data"""
        try:
            print("Creating vector store...")
            # Process cocktail data into Documents
            documents = process_cocktail_data("data/cocktails.csv")
            
            # Initialize vector store without metadata_config
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            # Add index optimization
            vector_store.index.nprobe = 3  # Increase search accuracy
            return vector_store
        
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            raise

    def search_cocktails(self, query: str, k: int = 5):
        """Search for cocktails based on query"""
        results = self.vector_store.similarity_search(query, k=k)
        return results

    def _load_favorites(self) -> set:
        """Load favorites from file"""
        try:
            if os.path.exists(self.favorites_file):
                with open(self.favorites_file, 'r') as f:
                    return set(json.load(f))
            return set()
        except Exception as e:
            print(f"Error loading favorites: {e}")
            return set()

    def _save_favorites(self):
        """Save favorites to file"""
        try:
            os.makedirs(os.path.dirname(self.favorites_file), exist_ok=True)
            with open(self.favorites_file, 'w') as f:
                json.dump(list(self.favorite_ingredients), f)
        except Exception as e:
            print(f"Error saving favorites: {e}")

    def add_favorite_ingredient(self, ingredient: str):
        """Add an ingredient to favorites and store in vector DB"""
        try:
            ingredient = ingredient.lower()
            self.favorite_ingredients.add(ingredient)
            self._save_favorites()

            # Create a document for the preference
            preference_doc = Document(
                page_content=f"User likes {ingredient} in cocktails",
                metadata={
                    'type': 'preference',
                    'ingredient': ingredient,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # Add to vector store
            self.vector_store.add_documents([preference_doc])
            
            return {"message": f"Added {ingredient} to favorites"}
        except Exception as e:
            print(f"Error adding favorite: {str(e)}")
            raise

    def get_favorite_ingredients(self):
        """Get user's favorite ingredients"""
        return list(self.favorite_ingredients)
        
    def search_cocktails_by_ingredient(self, ingredient: str, limit: int = 5) -> List[Dict]:
        """Search for cocktails containing specific ingredient"""
        try:
            # Convert ingredient to lowercase for case-insensitive search
            ingredient = ingredient.lower().strip()
            
            # Search using the ingredient as a query
            results = self.vector_store.similarity_search(
                f"cocktail with {ingredient}",
                k=limit
            )
            
            # Filter results to ensure they contain the ingredient
            matching_cocktails = []
            for result in results:
                if ingredient in result.metadata['ingredients'].lower():
                    matching_cocktails.append({
                        'name': result.metadata['name'],
                        'ingredients': result.metadata['ingredients'],
                        'category': result.metadata['category'],
                        'glass_type': result.metadata['glass_type'],
                        'alcoholic': result.metadata['alcoholic']
                    })
                    
            return matching_cocktails

        except Exception as e:
            print(f"Error searching by ingredient: {str(e)}")
            return []
        
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
        try:
            # Search using explicit query for non-alcoholic drinks
            results = self.vector_store.similarity_search(
                "non-alcoholic cocktails",
                k=limit * 2,  # Get more results to filter
                filter={"type": "cocktail"}
            )
            
            # Filter results to ensure they're non-alcoholic
            non_alcoholic = []
            for result in results:
                if result.metadata['alcoholic'].lower() == 'non alcoholic':
                    non_alcoholic.append({
                        'name': result.metadata['name'],
                        'ingredients': result.metadata['ingredients'],
                        'category': result.metadata['category'],
                        'glass_type': result.metadata['glass_type'],
                        'alcoholic': result.metadata['alcoholic']
                    })
                    if len(non_alcoholic) >= limit:
                        break
                    
            return non_alcoholic
        except Exception as e:
            print(f"Error getting non-alcoholic cocktails: {str(e)}")
            return []

    def search_with_preferences(self, query: str, k: int = 5):
        """Search cocktails considering user preferences"""
        try:
            # Get user's preferences
            preferences = list(self.favorite_ingredients)
            
            if preferences:
                # Enhance query with preferences
                enhanced_query = f"{query} with ingredients like {', '.join(preferences)}"
            else:
                enhanced_query = query
            
            # Search using enhanced query
            results = self.vector_store.similarity_search(
                enhanced_query,
                k=k,
                filter={"type": "cocktail"}  # Only return cocktail documents
            )
            
            return results
        except Exception as e:
            print(f"Error in preference-based search: {str(e)}")
            return []

    def remove_favorite_ingredient(self, ingredient: str):
        """Remove an ingredient from favorites"""
        try:
            ingredient = ingredient.lower()
            if ingredient in self.favorite_ingredients:
                self.favorite_ingredients.remove(ingredient)
                self._save_favorites()
                return {"message": f"Removed {ingredient} from favorites"}
            return {"message": f"{ingredient} was not in your favorites"}
        except Exception as e:
            print(f"Error removing favorite: {str(e)}")
            raise 