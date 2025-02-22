import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.cocktail_service import CocktailService

def test_favorites_persistence():
    print("\n=== Testing Favorites Persistence ===")
    
    # Initialize service
    service = CocktailService()
    
    # Check initial favorites
    print("\nInitial favorites:")
    print(service.get_favorite_ingredients())
    
    # Add some favorites
    test_ingredients = ["vodka", "lime", "mint"]
    print("\nAdding ingredients:", test_ingredients)
    for ingredient in test_ingredients:
        service.add_favorite_ingredient(ingredient)
    
    # Show current favorites
    print("\nCurrent favorites:")
    print(service.get_favorite_ingredients())
    
    # Create new service instance to verify persistence
    print("\nCreating new service instance...")
    new_service = CocktailService()
    print("\nFavorites from new instance:")
    print(new_service.get_favorite_ingredients())

if __name__ == "__main__":
    test_favorites_persistence() 