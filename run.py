import argparse
from utils.embedder import Embedder
from utils.chroma_db_manager import ChromaDBManager
import warnings
# Suppress all warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

print("Querying...⌛")

def main(question):
    # Initialize the embedder and ChromaDB manager
    embedder = Embedder(model_name='sentence-transformers/all-MiniLM-L6-v2')
    chroma_db_manager = ChromaDBManager(path="chroma_db_storage", collection_name="quora_questions")
    query_embedding = embedder.encode_single(question)
    similar_questions = chroma_db_manager.query_embeddings(query_embedding)

    # Display the results
    print("Similar/Related Questions")
    for i, question in enumerate(similar_questions, 1):
        print(f"{i}: {question}")
    print("Done ✅")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Query ChromaDB with a question.')
    parser.add_argument('question', type=str, help='The input to query for similar question/s.')

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function with the provided question
    main(args.question)