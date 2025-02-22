import faiss
import numpy as np
import os

# Check if the index file exists
index_path = "data/vector_store/index.faiss"
if not os.path.exists(index_path):
    print(f"Error: Index file not found at {index_path}")
    exit()

try:
    # Load the index
    index = faiss.read_index(index_path)

    # Print basic information about the index
    print("=== FAISS Index Information ===")
    print(f"Total vectors stored: {index.ntotal}")
    print(f"Vector dimension: {index.d}")
    print(f"Index type: {type(index)}")

    # If the index is not empty, show a sample vector
    if index.ntotal > 0:
        # Reconstruct the first vector (if possible)
        try:
            vector = index.reconstruct(0)  # Get the first vector
            print("\n=== Sample Vector ===")
            print(f"First vector (truncated): {vector[:10]}...")  # Show first 10 dimensions
        except Exception as e:
            print(f"\nCannot reconstruct vectors: {e}")
    
    # If you want to see more details about the index structure
    print("\n=== Index Parameters ===")
    print(f"Is index trained: {index.is_trained}")
    if hasattr(index, 'nprobe'):
        print(f"Number of probes: {index.nprobe}")

    # You can also check if it's an IVF (Inverted File) index
    if isinstance(index, faiss.IndexIVF):
        print(f"\n=== IVF Specific Info ===")
        print(f"Number of centroids: {index.nlist}")
        print(f"Quantizer type: {type(index.quantizer)}")

except Exception as e:
    print(f"Error loading or inspecting index: {e}")