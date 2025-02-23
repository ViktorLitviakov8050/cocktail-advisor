import pandas as pd
from typing import List
from langchain.docstore.document import Document
import os

def process_cocktail_data(csv_path: str) -> List[Document]:
    """
    Process cocktails CSV file into documents suitable for vector storage.
    """
    try:
        # Check if file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")

        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Map CSV columns
        df['glass_type'] = df['glassType']
        df['ingredients'] = df['ingredients'].apply(eval)  # Convert string list to actual list
        df['ingredientMeasures'] = df['ingredientMeasures'].apply(eval)
        
        documents = []
        
        # Process each cocktail into a Document
        for idx, row in df.iterrows():
            try:
                # Combine ingredients with their measures
                ingredients_with_measures = []
                for ing, measure in zip(row['ingredients'], row['ingredientMeasures']):
                    if pd.notna(measure):
                        ingredients_with_measures.append(f"{measure} {ing}")
                    else:
                        ingredients_with_measures.append(ing)
                
                ingredients_text = ", ".join(ingredients_with_measures)
                
                # Create formatted content
                content = f"""
                Cocktail Name: {str(row['name']).strip()}
                Ingredients: {ingredients_text}
                Instructions: {str(row['instructions']).strip()}
                Glass Type: {str(row['glass_type']).strip()}
                Category: {str(row['category']).strip()}
                Alcoholic: {str(row['alcoholic']).strip()}
                """
                
                # Create metadata for searching
                metadata = {
                    'name': str(row['name']).strip(),
                    'category': str(row['category']).strip(),
                    'glass_type': str(row['glass_type']).strip(),
                    'alcoholic': str(row['alcoholic']).strip(),
                    'ingredients': ingredients_text,
                    'source': 'cocktails_database',
                    'type': 'cocktail'
                }
                
                # Create Document object for vector store
                doc = Document(
                    page_content=content.strip(),
                    metadata=metadata
                )
                
                documents.append(doc)
                
            except Exception as row_error:
                print(f"Warning: Error processing row {idx}: {str(row_error)}")
                continue
            
        if not documents:
            raise ValueError("No documents were successfully processed")
            
        print(f"Successfully processed {len(documents)} cocktail recipes")
        return documents
        
    except Exception as e:
        print(f"Error processing cocktail data: {str(e)}")
        raise

def initialize_vector_store(documents: List[Document], vector_store_class, **kwargs):
    """
    Initialize vector store with processed documents.
    """
    try:
        if not documents:
            raise ValueError("No documents provided for vector store initialization")

        if not kwargs.get('embedding'):
            raise ValueError("No embedding model provided")

        vector_store = vector_store_class.from_documents(
            documents=documents,
            **kwargs
        )
        
        print("Successfully initialized vector store")
        return vector_store
        
    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
        raise