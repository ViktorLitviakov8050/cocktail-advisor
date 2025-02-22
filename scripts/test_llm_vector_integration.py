import os
import sys
import asyncio
import pandas as pd
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.llm_service import LLMService
from app.services.cocktail_service import CocktailService

async def test_llm_vector_integration():
    print("\n=== Testing LLM-Vector Store Integration ===")
    
    # First, let's check the CSV data
    csv_path = "data/cocktails.csv"
    df = pd.read_csv(csv_path)
    print(f"\nTotal cocktails in CSV: {len(df)}")
    print("\nSample from CSV:")
    print(df[['name', 'ingredients', 'alcoholic']].head(2))
    
    # Initialize services
    cocktail_service = CocktailService()
    llm_service = LLMService()
    
    # Test cases with different types of queries
    test_queries = [
        "I want to make a cocktail with gin and tonic",
        "What's a good non-alcoholic drink?",
        "Suggest me something with vodka and fruit juice",
        "I need a classic cocktail recipe"
    ]
    
    for query in test_queries:
        print(f"\n\nTesting query: '{query}'")
        print("=" * 50)
        
        try:
            # First show vector store results
            print("\nVector Store Results:")
            vector_results = cocktail_service.search_cocktails(query, k=2)
            for i, result in enumerate(vector_results, 1):
                print(f"\nMatch {i}:")
                print(f"Name: {result.metadata['name']}")
                print(f"Ingredients: {result.metadata['ingredients']}")
                print(f"Category: {result.metadata['category']}")
            
            # Then show LLM response using these results
            print("\nLLM Response:")
            response = await llm_service.process_message(query)
            print(response)
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    # Run async test
    asyncio.run(test_llm_vector_integration()) 