Retrieval-Augmented Generation (RAG) for Indian Food Recipes
============================================================

### Sample Query:
![image](https://github.com/user-attachments/assets/5e52b930-d8f5-4ee8-8c95-bf3e78116fce)

This project demonstrates a Retrieval-Augmented Generation (RAG) system designed to provide detailed responses about Indian food recipes. It combines a retrieval mechanism with a generative model to offer rich, contextually accurate answers based on user queries. Notably, this implementation achieves RAG functionality without using LangChain, focusing on direct integration with ChromaDB and the google/flan-t5-small model for a streamlined solution.

### Project Highlights:
- Retrieval Mechanism: Uses ChromaDB to store and retrieve recipe information. Recipes are embedded and queried based on user input to find the most relevant data.
- Generative Model: Employs the google/flan-t5-small model from Hugging Face's Transformers library to generate detailed responses. This model processes the retrieved recipe data to create contextually rich answers.
- No LangChain: Implements RAG without LangChain, relying instead on a direct approach with ChromaDB and the selected generative model to achieve its functionality.

### Files Overview:
- embedder.py: Manages the embedding of query data using a pre-trained model to generate dense vector representations.
- chroma_db_manager.py: Handles storing and querying of recipe embeddings in ChromaDB. It retrieves relevant recipes based on user queries and prepares data for the generative model.
- run.py: Executes the RAG process by querying ChromaDB for related recipes and generating responses using the google/flan-t5-small model.
