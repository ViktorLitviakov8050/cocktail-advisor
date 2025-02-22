# Cocktail Advisor Chat

A Python-based chat application that uses RAG (Retrieval-Augmented Generation) to provide cocktail recommendations and answer cocktail-related questions.

## Implementation Versions

The application can be run in three different ways, each on its own branch:

1. **Standard Version** (`main` branch)
   - Basic implementation with FAISS vector store
   - Run with: `uvicorn app.main:app --reload`
   - Best for development and testing

2. **Docker Version** (`feature/docker-implementation` branch)
   - Containerized application with persistent vector store
   - Run with: `docker compose up --build`
   - Best for deployment and data persistence

3. **Favorites-Optional Version** (`feature/disable-favorites` branch)
   - Configurable favorites system
   - Run with: `ENABLE_FAVORITES=false uvicorn app.main:app --reload`
   - Best for testing different UI configurations

To switch between versions:
```bash
# Switch to desired branch
git checkout main                          # Standard version
git checkout feature/docker-implementation # Docker version
git checkout feature/disable-favorites     # Favorites-optional version
```

## Features

- Chat interface for cocktail-related queries
- Integration with OpenAI GPT for natural language processing
- Vector database (FAISS) for storing and retrieving cocktail information
- User preference tracking for personalized recommendations
- RAG system for enhanced responses

## Prerequisites

- Python 3.8+
- OpenAI API key

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ViktorLitviakov8050/cocktail-advisor.git
cd cocktail-advisor
```
