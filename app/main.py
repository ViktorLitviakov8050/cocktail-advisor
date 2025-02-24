from fastapi import FastAPI, Request, Body, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel

from app.services.llm_service import LLMService
from .services.cocktail_service import CocktailService
from .models.schemas import ChatResponse

# Load environment variables
load_dotenv()

app = FastAPI(title="Cocktail Advisor Chat")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Initialize services
llm_service = LLMService()
cocktail_service = CocktailService()

class Message(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request
    })

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        message_text = body.get("text", "")
            
        if not message_text or not message_text.strip():
            raise HTTPException(status_code=400, detail="Message text cannot be empty")
            
        response = await llm_service.process_message(message_text)
        if not response:
            return {"response": "I apologize, but I couldn't generate a proper response. Could you try rephrasing your question?"}
        return {"response": response}
    except ValueError as ve:
        print(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your message: {str(e)}"
        )

@app.get("/")
async def root():
    return {"message": "Welcome to the Cocktail Recommendation System"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", 
                host="0.0.0.0", 
                port=8000, 
                reload=True,    # Enable auto-reload
                debug=True)     # Enable debug mode 