import os
import sys
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.llm_service import LLMService
from app.services.cocktail_service import CocktailService

async def test_favorites_workflow():
    print("\n=== Testing Favorites Workflow ===")
    
    # Initialize services
    llm_service = LLMService()
    cocktail_service = CocktailService()
    
    # 1. Test manual addition of favorites
    print("\n1. Testing manual addition of favorites:")
    test_ingredients = ["vodka", "lime", "mint"]
    print(f"Adding ingredients: {test_ingredients}")
    
    for ingredient in test_ingredients:
        result = cocktail_service.add_favorite_ingredient(ingredient)
        print(f"Added {ingredient}: {result['message']}")
    
    # 2. Test loading saved favorites
    print("\n2. Current favorites from file:")
    favorites = cocktail_service.get_favorite_ingredients()
    print(f"Loaded favorites: {favorites}")
    
    # 3. Test LLM preference detection
    print("\n3. Testing LLM preference detection:")
    test_messages = [
        "I like gin and tonic",
        "My favorite ingredient is rum",
        "I love cocktails with mint"
    ]
    
    for message in test_messages:
        print(f"\nUser message: '{message}'")
        response = await llm_service.process_message(message)
        print(f"LLM Response: {response}")
        
        # Check if preferences were saved
        updated_favorites = cocktail_service.get_favorite_ingredients()
        print(f"Updated favorites: {updated_favorites}")
    
    # 4. Test persistence
    print("\n4. Testing persistence:")
    print("Creating new service instance...")
    new_service = CocktailService()
    persisted_favorites = new_service.get_favorite_ingredients()
    print(f"Favorites from new instance: {persisted_favorites}")
    
    # 5. Check favorites.json file
    print("\n5. Checking favorites.json file:")
    favorites_file = "data/favorites.json"
    if os.path.exists(favorites_file):
        with open(favorites_file, 'r') as f:
            import json
            file_content = json.load(f)
            print(f"File contents: {file_content}")
    else:
        print("favorites.json file not found!")

if __name__ == "__main__":
    asyncio.run(test_favorites_workflow()) 