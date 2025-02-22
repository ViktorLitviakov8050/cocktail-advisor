from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from typing import List, Dict
import os
from app.services.cocktail_service import CocktailService

class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY")  # Explicitly set API key
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question"
        )
        self.cocktail_service = CocktailService()  # Add this
        self.vector_store = self.cocktail_service.vector_store  # Add direct access to vector store
        
    async def process_message(self, message: str) -> str:
        """Process user message and return response"""
        try:
            # Check if user is sharing preferences
            if self._is_sharing_preferences(message):
                # Store preferences and confirm
                return self._handle_preferences(message)
            
            # Otherwise, generate cocktail response
            response = await self._generate_response(message)
            return response
        except Exception as e:
            # Log the error and return a friendly message
            print(f"Error processing message: {str(e)}")
            return "I apologize, but I encountered an error processing your message. Please try again."
        
    def _is_sharing_preferences(self, message: str) -> bool:
        """Detect if user is sharing preferences"""
        # Implement preference detection logic
        preference_keywords = ["i like", "my favorite", "i love", "i prefer"]
        return any(keyword in message.lower() for keyword in preference_keywords)
        
    def _handle_preferences(self, message: str) -> str:
        """Handle user preferences"""
        # Extract ingredients from preference statements
        preference_words = ["like", "love", "prefer", "favorite", "enjoy"]
        message_lower = message.lower()
        
        # Find mentioned ingredients
        ingredients = []
        if any(word in message_lower for word in preference_words):
            # Remove preference words to isolate ingredients
            for word in preference_words:
                message_lower = message_lower.replace(f"i {word}", "")
                message_lower = message_lower.replace(f"my {word}", "")
            
            # Common ingredients to look for
            common_ingredients = ["vodka", "gin", "rum", "tequila", "whiskey", 
                                "lime", "lemon", "mint", "tonic", "juice",
                                "sweet", "sour", "bitter", "fruity", "strong"]
            
            for ingredient in common_ingredients:
                if ingredient in message_lower:
                    # Add to favorites through cocktail service
                    self.cocktail_service.add_favorite_ingredient(ingredient)
                    ingredients.append(ingredient)
        
        if ingredients:
            return f"I've noted that you like {', '.join(ingredients)}! I'll keep these preferences in mind."
        return "I understand you're sharing preferences, but I'm not sure which ingredients you like."
        
    async def _generate_response(self, message: str) -> str:
        """Generate response using LLM"""
        try:
            # Extract any new preferences from the message
            self._handle_preferences(message)
            
            # Search with preferences
            cocktail_results = self.cocktail_service.search_with_preferences(message, k=3)
            
            # Get user's preference history
            preference_results = self.vector_store.similarity_search(
                message,
                k=2,
                filter={"type": "preference"}
            )
            
            # Format all information
            cocktail_info = "\n".join([
                f"Cocktail: {result.metadata['name']}\n"
                f"Ingredients: {result.metadata['ingredients']}\n"
                f"Category: {result.metadata['category']}\n"
                for result in cocktail_results
            ])
            
            preference_info = "\n".join([
                doc.page_content for doc in preference_results
            ])
            
            # Create enhanced prompt
            prompt = f"""Based on the user's message: '{message}'
            
            User's preference history:
            {preference_info}
            
            Relevant cocktails:
            {cocktail_info}
            
            Please provide a helpful response about these cocktails, considering the user's preferences and history.
            If suggesting cocktails, prioritize those that match the user's taste preferences."""
            
            # Generate response
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm.agenerate([messages])
            
            return response.generations[0][0].text
            
        except Exception as e:
            # Log the specific error
            print(f"Error generating LLM response: {str(e)}")
            raise  # Re-raise to be handled by process_message 