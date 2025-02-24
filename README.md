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
- Optional favorites system for tracking preferred ingredients

## Available Implementations

The project offers different implementations across various branches:

- `main` - Standard version with basic implementation
- `feature/docker-implementation` - Dockerized version of the application
- More implementations coming soon...

## Prerequisites

- Python 3.8+
- OpenAI API key

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ViktorLitviakov8050/cocktail-advisor.git
cd cocktail-advisor
```

2. Choose your preferred implementation by checking out the appropriate branch:
```bash
# For standard version
git checkout main

# For Docker implementation
git checkout feature/docker-implementation
```

3. Set up environment variables in `.env`:
```env
OPENAI_API_KEY=your_key_here
ENABLE_FAVORITES=true  # or false to disable favorites
```

## Running the Application

You can control the favorites functionality when starting the app:

```bash
# Run with favorites (if ENABLE_FAVORITES=true in .env)
uvicorn app.main:app --reload

# Override .env and run without favorites
ENABLE_FAVORITES=false uvicorn app.main:app --reload
```

## RAG System Flow

The application uses a Retrieval-Augmented Generation (RAG) system that works as follows:

1. **Initial Setup**:
   - Cocktail data is processed from CSV into documents
   - Documents are stored in FAISS vector database
   - Each document contains cocktail information and metadata
   - User preferences are stored both in JSON and vector store

2. **Message Processing Flow**:
   ```mermaid
   graph TD
       A[User Message] --> B[LLMService]
       B --> C{Is Preference?}
       C -->|Yes| D[Store in Vector DB]
       C -->|No| E[Search Cocktails]
       D --> F[Update Favorites]
       E --> G[Get User Preferences]
       G --> H[Generate LLM Response]
       F --> H
   ```

3. **RAG Components**:
   - Vector Store: FAISS database storing cocktail data and preferences
   - Preference Detection: Identifies when users share ingredient preferences
   - Enhanced Search: Combines user query with stored preferences
   - Context-Aware Responses: LLM receives both cocktail data and user preferences

4. **Data Flow**:
   - User preferences are detected from messages
   - Preferences enhance cocktail search queries
   - Search results are combined with preference history
   - LLM generates personalized recommendations

Questions for stakeholders:
- Should be implemented favorite ingredients section on UI side or no? (probably it's a good option to change time by time your favorite ingredients)
- Should favorites persist between sessions or reset on each startup?
- Should we limit the number of favorite ingredients a user can save?
- Should favorite ingredients influence the ranking of cocktail recommendations?
- Should we add the ability to group favorite ingredients by category (e.g., spirits, mixers, garnishes)?

## Docker Implementation

Run the application using Docker:

```bash
# Build and start the container
docker-compose up --build

# Stop the container
docker-compose down
```

Environment variables can be configured in docker-compose.yml or .env file.

