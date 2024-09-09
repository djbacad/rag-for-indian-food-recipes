import argparse
from utils.embedder import Embedder
from utils.chroma_db_manager import ChromaDBManager
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
# Suppress all warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

print("Querying and Generating...⌛")

def main(question):
    # Initialize the embedder and ChromaDB manager
    embedder = Embedder(model_name='sentence-transformers/all-MiniLM-L6-v2')
    chroma_db_manager = ChromaDBManager(path="chroma_db_storage", collection_name="recipes")

    # Embed the input query
    query_embedding = embedder.encode_single(question)

    # Retrieve similar/related documents (recipes, etc.)
    context = chroma_db_manager.query_embeddings(query_embedding)

    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

    # Prepare input text for the model
    input_text = f"Question: {question}. Context: {context}"
    input_ids = tokenizer(input_text, max_length=512, truncation=True, return_tensors='pt').input_ids.to("cuda")

   # Generate a response
    print("Generating Response...⌛")
    outputs = model.generate(input_ids, max_new_tokens=2000, min_new_tokens=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(generated_text)
    # Display the generated response
    print("Generated Answer:")
    print(generated_response[0]['generated_text'])
    print("Done ✅")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Query ChromaDB and generate a response using RAG.')
    parser.add_argument('question', type=str, help='The input query for which to retrieve related content and generate an answer.')

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function with the provided question
    main(args.question)
