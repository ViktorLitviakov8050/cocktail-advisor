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
        
    def _format_cocktail_results(self, results) -> str:
        """Format cocktail results into a readable string for the LLM"""
        formatted_results = []
        for result in results:
            cocktail_info = f"""
            Cocktail: {result.metadata.get('name', 'Unknown')}
            Category: {result.metadata.get('category', 'Unknown')}
            Ingredients: {result.metadata.get('ingredients', 'Unknown')}
            Instructions: {result.page_content.split('Instructions:')[-1].strip()}
            """
            formatted_results.append(cocktail_info)
            
        return "\n".join(formatted_results)
        
    async def process_message(self, message: str) -> str:
        """Process user message and return response"""
        try:
            # Check for basic greetings
            message_lower = message.lower().strip()
            greetings = {
                'hi': 'Hi! How can I help you with cocktails today?',
                'hello': 'Hello! I\'m your cocktail advisor. What would you like to know?',
                'hey': 'Hey there! Ready to explore some cocktails?',
                'greetings': 'Greetings! What kind of cocktail are you interested in?'
            }
            
            if message_lower in greetings:
                return greetings[message_lower]
            
            # Handle help-related queries
            help_phrases = ['help', 'what can you do', "i don't know", 'how can you help']
            if any(phrase in message_lower for phrase in help_phrases):
                return """I can help you with:
1. Finding cocktails with specific ingredients (e.g., "Show me cocktails with vodka")
2. Recommending cocktails based on your preferences (e.g., "I like sweet drinks")
3. Providing cocktail recipes (e.g., "How do I make a Mojito?")
4. Finding non-alcoholic options (e.g., "Show me mocktails")
5. Suggesting similar cocktails (e.g., "What's similar to a Margarita?")

What would you like to know about?"""
            
            # Check if user is sharing preferences
            if self._is_sharing_preferences(message):
                return self._handle_preferences(message)
            
            # For other queries, use the cocktail-specific response
            response = await self._generate_response(message)
            return response
            
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            return "I apologize, but I encountered an error. Please try again or ask for help to see what I can do."
        
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
        try:
            # Query classification
            query_type = self._classify_query(message)
            
            if query_type == 'popular_cocktails':
                return """Here are some of the most popular cocktails:

1. Margarita - A refreshing mix of tequila, lime juice, and triple sec
2. Mojito - Classic Cuban cocktail with rum, mint, lime, and soda
3. Old Fashioned - A timeless whiskey cocktail with bitters and sugar
4. Moscow Mule - Vodka, ginger beer, and lime in a copper mug
5. Martini - The elegant combination of gin or vodka with vermouth

Would you like to know how to make any of these? Or would you prefer something different?"""
            
            elif query_type == 'general_info':
                return """Let me tell you about cocktails in a fun way! 🍸

Cocktails are like liquid art - they can be:
• Sweet and fruity (like Piña Coladas)
• Strong and sophisticated (like Manhattans)
• Light and refreshing (like Gin & Tonics)
• Creamy and indulgent (like White Russians)

What kind of flavors interest you? I can suggest some cocktails based on your taste!"""
            
            else:
                # Use enhanced search for specific cocktail queries
                cocktail_results = self.cocktail_service.search_with_preferences(message)
                if cocktail_results:
                    prompt = self._build_enhanced_prompt(message, cocktail_results, query_type)
                    return await self._get_llm_response(prompt)
                else:
                    return "I'm not sure about that specific cocktail. Would you like me to suggest some popular ones instead?"
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I'm having trouble understanding that. Could you try rephrasing your question?"

    def _classify_query(self, message: str) -> str:
        """Classify the type of query for better response targeting"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['popular', 'best', 'famous', 'most']):
            return 'popular_cocktails'
        elif any(word in message_lower for word in ['what is', 'tell me about', 'explain']):
            return 'general_info'
        else:
            return 'specific_query'

    def _build_enhanced_prompt(self, message: str, results, query_type: str) -> str:
        """Build context-aware prompts based on query type"""
        base_context = self._format_cocktail_results(results)
        preferences = self.cocktail_service.get_favorite_ingredients()
        
        prompts = {
            'recommendation': f"""Based on the user's preferences ({', '.join(preferences)}), 
                recommend cocktails from: {base_context}. 
                Explain why each cocktail matches their taste.""",
            
            'ingredient_query': f"""Analyze these cocktails: {base_context}
                Focus on ingredient combinations and proportions.
                Provide detailed information about how ingredients work together.""",
            
            'general_query': f"""Using this cocktail information: {base_context}
                Provide a comprehensive answer about cocktail preparation, history, or techniques."""
        }
        
        return prompts[query_type]

    async def _get_llm_response(self, prompt: str) -> str:
        """Get LLM response with quality checks"""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm.agenerate([messages])
            
            # Quality checks
            response_text = response.generations[0][0].text
            if self._validate_response(response_text):
                return response_text
            else:
                # Retry with more specific prompt
                return await self._generate_fallback_response(prompt)
            
        except Exception as e:
            print(f"Error in LLM response: {str(e)}")
            return "I apologize, but I encountered an error. Please try rephrasing your question."

    def _validate_response(self, response: str) -> bool:
        """Validate response quality"""
        min_length = 50
        required_elements = ['cocktail', 'ingredient']
        
        return (
            len(response) >= min_length and
            any(element in response.lower() for element in required_elements)
        ) 