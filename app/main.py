from fastapi import FastAPI, Request, Form, Body
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from typing import Optional
from dotenv import load_dotenv

from app.services.llm_service import LLMService
from .services.cocktail_service import CocktailService
from .models.schemas import ChatResponse

# Load environment variables
load_dotenv()

app = FastAPI(title="Cocktail Advisor Chat")

# Mount static files - make sure this comes BEFORE templates initialization
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Initialize services
llm_service = LLMService()
cocktail_service = CocktailService()

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    # Force reload environment variables
    load_dotenv(override=True)  # Add override=True
    
    # Get the flag from environment variable, default to false if not set
    enable_favorites = os.getenv("ENABLE_FAVORITES", "false").lower() == "true"
    print(f"ENABLE_FAVORITES is set to: {enable_favorites}")  # Add debug print
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "enable_favorites": enable_favorites
    })

@app.post("/chat")
async def chat(message: str = Form(...)) -> ChatResponse:
    # Send message to LLM service
    response = await llm_service.process_message(message)
    return ChatResponse(message=response)

@app.get("/favorites")
async def get_favorites():
    return cocktail_service.get_favorite_ingredients()

@app.post("/favorites/ingredients")
async def add_favorite_ingredient(ingredient: str = Body(..., embed=True)):
    """
    Expects JSON body: {"ingredient": "vodka"}
    """
    return cocktail_service.add_favorite_ingredient(ingredient)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", 
                host="0.0.0.0", 
                port=8000, 
                reload=True,    # Enable auto-reload
                debug=True)     # Enable debug mode 