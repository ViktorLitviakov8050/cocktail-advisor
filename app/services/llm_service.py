from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from typing import List, Dict
import os
from app.services.cocktail_service import CocktailService

class LLMService:
    def __init__(self):
        # You can switch between models by changing model_name:
        # - "gpt-3.5-turbo-0125" (latest GPT-3.5, better than old 3.5)
        # - "gpt-4-0125-preview" (latest GPT-4, most capable)
        # - "gpt-4-turbo-preview" (latest GPT-4 Turbo)
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo-0125", 
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question"
        )
        self.cocktail_service = CocktailService()
        self.vector_store = self.cocktail_service.vector_store
        
    def _format_cocktail_results(self, results) -> str:
        """Format cocktail results into a readable string"""
        formatted_results = []
        for result in results:
            if isinstance(result, dict):
                name = result.get("name", "Unknown")
                ingredients = result.get("ingredients", "Unknown")
            else:
                name = result.metadata.get("name", "Unknown")
                ingredients = result.metadata.get("ingredients", "Unknown")
            
            formatted_results.append(f"- {name}: {ingredients}")
        
        return "\n".join(formatted_results)
        
    async def process_message(self, message: str) -> str:
        """Process user message and return response"""
        try:
            # First, let's understand the message context and intent using LLM
            understanding = await self._understand_message(message)
            
            # Use the understanding to generate appropriate response
            return await self._generate_contextual_response(message, understanding)
            
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            return "I apologize, but I encountered an error. Please try again or ask for help to see what I can do."

    async def _understand_message(self, message: str) -> dict:
        """Use LLM to deeply understand the message context and intent"""
        prompt = f"""As an AI assistant who specializes in cocktails but can discuss any topic, analyze this message deeply.
        Message: "{message}"

        Provide a detailed analysis in JSON format with the following structure:
        {{
            "intent": {{
                "primary": string (e.g., "greeting", "cocktail_request", "preference_management", "general_chat", "help_request"),
                "secondary": string (e.g., "add_favorite", "remove_favorite", "get_recipe", "find_similar", "casual_conversation"),
                "requires_cocktail_context": boolean
            }},
            "preferences": {{
                "action": string ("add", "remove", "list", "none"),
                "ingredients": [string],
                "show_current_favorites": boolean
            }},
            "cocktail_search": {{
                "type": string ("by_ingredient", "by_name", "by_similarity", "by_category", "none"),
                "filters": {{
                    "count": number or null,
                    "is_alcoholic": boolean or null,
                    "ingredients": [string],
                    "similar_to": string or null,
                    "category": string or null,
                    "other_constraints": [string]
                }}
            }},
            "conversation": {{
                "topic": string,
                "requires_clarification": boolean,
                "sentiment": string,
                "is_follow_up": boolean
            }},
            "required_actions": [string]
        }}"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm.agenerate([messages])
            
            # Parse JSON response
            response_text = response.generations[0][0].text.strip()
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                import json
                return json.loads(json_str)
            
            return {"intent": {"primary": "general_chat"}}
        except Exception as e:
            print(f"Error in message understanding: {str(e)}")
            return {"intent": {"primary": "general_chat"}}

    async def _generate_contextual_response(self, message: str, understanding: dict) -> str:
        """Generate response based on message understanding"""
        try:
            # Get current favorites if needed
            favorites = []
            if any(action in understanding.get("required_actions", []) for action in ["list_favorites", "add_favorite", "remove_favorite"]):
                favorites = self.cocktail_service.get_favorite_ingredients()

            # Handle preference management
            preferences = understanding.get("preferences", {})
            if preferences.get("action") in ["add", "remove"]:
                ingredients = preferences.get("ingredients", [])
                if preferences["action"] == "add":
                    for ingredient in ingredients:
                        self.cocktail_service.add_favorite_ingredient(ingredient)
                else:  # remove
                    for ingredient in ingredients:
                        self.cocktail_service.remove_favorite_ingredient(ingredient)
                favorites = self.cocktail_service.get_favorite_ingredients()

            # Build context for LLM response
            cocktail_search = understanding.get("cocktail_search", {})
            conversation = understanding.get("conversation", {})
            
            prompt = f"""As an AI assistant specializing in cocktails but capable of general conversation, respond to: "{message}"

            Context:
            - User's Intent: {understanding.get("intent", {})}
            - Conversation Topic: {conversation.get("topic")}
            - Current Favorites: {favorites if favorites else "None"}
            - Search Parameters: {cocktail_search}
            
            If cocktail-related:
            1. Use the vector store to find relevant cocktails
            2. Include ingredients and measurements
            3. Consider user preferences
            4. Provide clear instructions if needed
            
            If general conversation:
            1. Be natural and engaging
            2. Use cocktail analogies if appropriate
            3. Show personality while staying professional"""

            # Get cocktail results if needed
            results = []
            if cocktail_search.get("type") != "none":
                filters = cocktail_search.get("filters", {})
                count = filters.get("count", 5)
                
                try:
                    if filters.get("is_alcoholic") is False:
                        results = self.cocktail_service.get_non_alcoholic_cocktails(limit=count)
                    elif filters.get("similar_to"):
                        results = self.cocktail_service.get_similar_cocktails(filters["similar_to"], limit=count)
                    elif filters.get("ingredients"):
                        for ingredient in filters["ingredients"]:
                            ingredient_results = self.cocktail_service.search_cocktails_by_ingredient(ingredient, limit=count)
                            if ingredient_results:
                                results.extend(ingredient_results)
                        # Remove duplicates and limit results
                        seen = set()
                        unique_results = []
                        for r in results:
                            name = r.get("name", "") if isinstance(r, dict) else r.metadata.get("name", "")
                            if name not in seen:
                                seen.add(name)
                                unique_results.append(r)
                        results = unique_results[:count]
                    else:
                        results = self.cocktail_service.search_with_preferences(message, k=count)
                except Exception as e:
                    print(f"Error in cocktail search: {str(e)}")
                    # Continue with empty results

            if results:
                formatted_results = self._format_cocktail_results(results)
                prompt += f"\n\nAvailable cocktails:\n{formatted_results}"

            messages = [{"role": "user", "content": prompt}]
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].text.strip()
            
        except Exception as e:
            print(f"Error generating contextual response: {str(e)}")
            return "I apologize, but I encountered an error. Could you try rephrasing your request?"

    def _handle_help_request(self) -> str:
        """Handle help-related queries"""
        return """I can help you with:
1. Finding cocktails with specific ingredients (e.g., "Show me cocktails with vodka")
2. Recommending cocktails based on your preferences (e.g., "I like sweet drinks")
3. Providing cocktail recipes (e.g., "How do I make a Mojito?")
4. Finding non-alcoholic options (e.g., "Show me mocktails")
5. Suggesting similar cocktails (e.g., "What's similar to a Margarita?")

What would you like to know about?"""

    async def _generate_response(self, message: str) -> str:
        """Generate response using LLM"""
        try:
            # Let LLM analyze the query and generate appropriate response
            prompt = f"""As a cocktail expert and conversational AI, analyze and respond to this message: "{message}"
            
            Consider:
            1. The type of information being requested
            2. Any specific constraints or preferences mentioned
            3. Whether it's about cocktails, ingredients, or general conversation
            
            If it's a cocktail-related query, format your response as a list of cocktails with their ingredients.
            If it's a general question, provide a natural, conversational response.
            If you're unsure, ask for clarification.
            
            Current context: You are a knowledgeable AI that specializes in cocktails but can engage in any topic of conversation.
            """
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].text.strip()
                
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error. Could you try rephrasing your request?"

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

    async def _is_sharing_preferences(self, message: str) -> bool:
        """Detect if user is sharing preferences using LLM"""
        result = await self._analyze_message_intent(message)
        return result["is_preference"]
        
    async def _handle_preferences(self, message: str) -> str:
        """Handle user preferences using LLM analysis"""
        result = await self._analyze_message_intent(message)
        
        if result["is_preference"] and result["ingredients"]:
            # Add each detected ingredient to favorites
            for ingredient in result["ingredients"]:
                self.cocktail_service.add_favorite_ingredient(ingredient)
            
            # Get updated list of favorites
            favorites = self.cocktail_service.get_favorite_ingredients()
            
            if len(result["ingredients"]) == 1:
                return f"I've added {result['ingredients'][0]} to your favorite ingredients. Your current favorites are: {', '.join(favorites)}"
            else:
                return f"I've added {', '.join(result['ingredients'])} to your favorite ingredients. Your current favorites are: {', '.join(favorites)}"
        
        return "I understand you're sharing preferences, but I'm not sure which ingredients you like. Could you please be more specific?" 