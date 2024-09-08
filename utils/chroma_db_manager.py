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

    def add_embeddings(self, embeddings, recipes):
        ids = [str(uuid.uuid4()) for _ in range(len(recipes))]  # Generate unique IDs for each text
        metadatas = [
            {
                "TranslatedRecipeName": recipe['TranslatedRecipeName'],
                "TranslatedIngredients": recipe['TranslatedIngredients'],
                "TranslatedInstructions": recipe['TranslatedInstructions'],
                "Course": recipe['Course'],
                "TotalTimeInMins": recipe['TotalTimeInMins'],
                "URL": recipe['URL']
            }
            for recipe in recipes
        ]
        # Adding documents and IDs to the collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query_embeddings(self, query_embedding, n_results=1):
        # Retrieve the most relevant documents based on the query_embedding
        results = self.collection.query(query_embeddings=query_embedding.tolist(), n_results=n_results)

        metadatas = results.get('metadatas', [[]])[0]
        if not metadatas:
            return "No relevant recipes found."
       
        # Prepare text from retrieved metadata
        context_parts = []
        for metadata in metadatas:
            recipe_name = metadata.get('TranslatedRecipeName', 'Unknown Recipe')
            course = metadata.get('Course', 'Unknown Course')
            ingredients = metadata.get('TranslatedIngredients', 'No Ingredients')
            instructions = metadata.get('TranslatedInstructions', 'No Instructions')
            url = metadata.get('URL', 'No URL')

            context_part = (f"Recipe: {recipe_name}\n"
                            f"Course: {course}\n"
                            f"Ingredients: {ingredients}\n"
                            f"Instructions: {instructions}\n"
                            f"URL: {url}\n")
            
            context_parts.append(context_part)

        # Combine all context parts into one string
        return " ".join(context_parts)  # Join parts with a space or other delimiter as needed