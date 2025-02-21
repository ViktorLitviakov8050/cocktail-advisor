from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import Optional

from app.services.llm_service import LLMService
from .services.cocktail_service import CocktailService
from .models.schemas import ChatResponse

app = FastAPI(title="Cocktail Advisor Chat")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Initialize services
llm_service = LLMService()
cocktail_service = CocktailService()

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat")
async def chat(message: str = Form(...)) -> ChatResponse:
    response = await llm_service.process_message(message)
    return ChatResponse(message=response)

@app.get("/favorites")
async def get_favorites():
    return cocktail_service.get_favorite_ingredients()

@app.post("/favorites/ingredients")
async def add_favorite_ingredient(ingredient: str):
    return cocktail_service.add_favorite_ingredient(ingredient) 