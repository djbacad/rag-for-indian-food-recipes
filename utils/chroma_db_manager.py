import chromadb
import uuid

class ChromaDBManager:
    def __init__(self, collection_name, path):
        # self.client = chromadb.Client()
        # We'll use the persistentclient for testing and dev
        # This creates a persistent instance of Chroma that saves to disk
        self.client = chromadb.PersistentClient(path=path)
        # self.collection = self.client.create_collection(name=collection_name)
        # We use get_or_create so we can access the chromadb object after persistence
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_embeddings(self, embeddings, texts):
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]  # Generate unique IDs for each text
        metadatas = [{"question": text} for text in texts]
        # Adding documents and IDs to the collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query_embeddings(self, query_embedding, n_results=5):
        results = self.collection.query(query_embeddings=query_embedding.tolist(), n_results=n_results)
        return [result['question'] for result in results['metadatas'][0]]