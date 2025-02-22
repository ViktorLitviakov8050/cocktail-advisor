import os
import sys
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.cocktail_service import CocktailService

def test_vector_store():
    print("\n=== Testing Vector Store ===")
    
    # Check if vector store exists
    vector_store_path = "data/vector_store"
    if os.path.exists(vector_store_path):
        print(f"✓ Vector store found at {vector_store_path}")
    else:
        print(f"✗ Vector store not found at {vector_store_path}")
        return

    # Initialize service
    service = CocktailService()
    
    # Test basic search
    print("\n=== Testing Search Functionality ===")
    query = "gin cocktail"
    results = service.search_cocktails(query, k=2)
    
    print(f"\nSearch results for '{query}':")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Name: {result.metadata.get('name')}")
        print(f"Category: {result.metadata.get('category')}")
        print(f"Ingredients: {result.metadata.get('ingredients')}")
    
    # Test ingredient search
    print("\n=== Testing Ingredient Search ===")
    ingredient = "vodka"
    results = service.search_cocktails_by_ingredient(ingredient, limit=2)
    
    print(f"\nCocktails containing '{ingredient}':")
    for i, result in enumerate(results, 1):
        print(f"\nCocktail {i}:")
        print(f"Name: {result.get('name')}")
        print(f"Ingredients: {result.get('ingredients')}")

if __name__ == "__main__":
    test_vector_store() 