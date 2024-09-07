from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Class to generate embeddings
class Embedder:
    def __init__(self, model_name, device='cuda'):
        self.model = SentenceTransformer(model_name, device=device)

    # Method for single input
    def encode_single(self, text):
        return self.model.encode([text], convert_to_tensor=True)
 
    def encode(self, texts):
        embeddings = []
        for text in tqdm(texts, desc="", unit="text"):
            embedding = self.model.encode(text, convert_to_tensor=True)
            embeddings.append(embedding)
        return embeddings