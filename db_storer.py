from utils.embedder import Embedder
from utils.chroma_db_manager import ChromaDBManager
import pandas as pd
import warnings
import torch

# Suppress all warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Persistence path
persist_path = "chroma_db_storage"

def load_data(file_path):
    df = pd.read_csv(file_path)
    recipes = df.to_dict(orient='records')
    return recipes

def main():
    # Load and preprocess the data
    file_path = 'datasets/IndianFoodDatasetCSV.csv'
    recipes = load_data(file_path)

    # Initialize the embedder and ChromaDB manager
    print("Initializing Embedder...⌛")
    embedder = Embedder(model_name='sentence-transformers/all-MiniLM-L6-v2')
    print("Starting ChromaDB Manager...⌛")
    chroma_db_manager = ChromaDBManager(path=persist_path, collection_name="recipes")

    # Combine ingredients and instructions into a single text for each recipe
    combined_texts = [
        f"{recipe['Ingredients']}. {recipe['Instructions']}"
        for recipe in recipes
    ]

    # Generate embeddings
    print("Generating Embeddings for ChromaDB...⌛")
    embeddings = embedder.encode(combined_texts)

    # Convert embeddings from a list of tensors to a list of lists (required by chromadb)
    if isinstance(embeddings, list) and isinstance(embeddings[0], torch.Tensor):
        embeddings = [e.tolist() for e in embeddings]

    # Store embeddings in ChromaDB
    print("Storing Embeddings in ChromaDB...⌛")
    chroma_db_manager.add_embeddings(embeddings, recipes)

    print(f"Embeddings stored in {persist_path} ✅")

if __name__ == "__main__":
    main()
