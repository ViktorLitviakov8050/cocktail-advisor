from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from typing import List, Dict
import os

class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY")  # Explicitly set API key
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
    async def process_message(self, message: str) -> str:
        """Process user message and return response"""
        try:
            # Detect if user is sharing preferences
            if self._is_sharing_preferences(message):
                return self._handle_preferences(message)
                
            # Otherwise process as regular query
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
        # Extract and store preferences
        # Return confirmation message
        return "I've noted your preferences! I'll keep them in mind for future recommendations."
        
    async def _generate_response(self, message: str) -> str:
        """Generate response using LLM"""
        try:
            # Create message list with proper format for ChatOpenAI
            messages = [{"role": "user", "content": message}]
            
            # Use acall instead of agenerate for ChatOpenAI
            response = await self.llm.acall(messages)
            
            # ChatOpenAI returns a dictionary with 'content' key
            if not isinstance(response, dict) or 'content' not in response:
                raise ValueError("Unexpected response format from LLM")
                
            return response['content']
            
        except Exception as e:
            # Log the specific error
            print(f"Error generating LLM response: {str(e)}")
            raise  # Re-raise to be handled by process_message 