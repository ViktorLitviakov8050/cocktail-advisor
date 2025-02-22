import os
import sys
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.llm_service import LLMService
from app.services.cocktail_service import CocktailService

async def test_rag_system():
    print("\n=== Testing RAG System with User Preferences ===")
    
    # Initialize services
    llm_service = LLMService()
    cocktail_service = CocktailService()
    
    # Test preference detection
    response = await llm_service.process_message("I like vodka")
    
    # Test cocktail recommendations
    response = await llm_service.process_message("Suggest a cocktail")
    
    # Test scenario 1: Adding preferences
    print("\n1. Testing preference detection and storage:")
    messages = [
        "I really like vodka and citrus flavors",
        "My favorite drinks are sweet and fruity",
        "I prefer strong cocktails with rum"
    ]
    
    for message in messages:
        print(f"\nUser message: '{message}'")
        response = await llm_service.process_message(message)
        print(f"LLM Response: {response}")
        
        # Show stored preferences
        print("Current preferences:", cocktail_service.get_favorite_ingredients())
    
    # Test scenario 2: Preference-influenced recommendations
    print("\n2. Testing preference-based recommendations:")
    query = "Suggest me a cocktail"
    print(f"\nUser query: '{query}'")
    
    # Get recommendations
    results = cocktail_service.search_with_preferences(query)
    print("\nRecommended cocktails:")
    for result in results:
        print(f"- {result.metadata['name']} ({result.metadata['ingredients']})")
    
    # Test scenario 3: RAG with preferences
    print("\n3. Testing RAG with preference history:")
    response = await llm_service.process_message("What cocktail should I try?")
    print(f"\nFinal LLM response with preferences: {response}")

if __name__ == "__main__":
    asyncio.run(test_rag_system()) 