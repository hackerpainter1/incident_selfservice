import numpy as np
from openai import OpenAI
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class DocumentRetriever:
    def __init__(self, api_key: str):
        """
        Initialize the DocumentRetriever with OpenAI API credentials.
        
        Args:
            api_key (str): Your OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)
        self.document_embeddings = {}
        self.documents = {}
        
    def process_documents(self, documents: Dict[str, str]) -> None:
        """
        Process and store documents with their embeddings.
        
        Args:
            documents (Dict[str, str]): Dictionary of document_id to document text
        """
        self.documents = documents
        for doc_id, content in documents.items():
            # Get embeddings for each document using OpenAI's embedding model
            embedding = self._get_embedding(content)
            self.document_embeddings[doc_id] = embedding
            
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embeddings for a piece of text using OpenAI's embedding model.
        
        Args:
            text (str): Text to get embeddings for
            
        Returns:
            List[float]: Embedding vector
        """
        response = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    
    def find_similar_documents(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Find the most similar documents to a query string.
        
        Args:
            query (str): Query text to find similar documents for
            top_k (int): Number of similar documents to return
            
        Returns:
            List[Tuple[str, float]]: List of (document_id, similarity_score) tuples
        """
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Calculate similarities between query and all documents
        similarities = []
        for doc_id, doc_embedding in self.document_embeddings.items():
            # Convert embeddings to numpy arrays for cosine similarity calculation
            query_array = np.array(query_embedding).reshape(1, -1)
            doc_array = np.array(doc_embedding).reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(query_array, doc_array)[0][0]
            similarities.append((doc_id, similarity))
        
        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

    def get_document_content(self, doc_id: str) -> str:
        """
        Get the content of a document by its ID.
        
        Args:
            doc_id (str): ID of the document to retrieve
            
        Returns:
            str: Document content
        """
        return self.documents.get(doc_id, "Document not found")

# Example usage
def main():
    # Initialize with your OpenAI API key
    retriever = DocumentRetriever("your-api-key-here")
    
    # Sample documents
    documents = {
        "doc1": "Machine learning is a subset of artificial intelligence that focuses on data and algorithms.",
        "doc2": "Natural language processing helps computers understand and generate human language.",
        "doc3": "Deep learning is a type of machine learning that uses neural networks with multiple layers.",
        "doc4": "Computer vision enables machines to understand and process visual information from the world."
    }
    
    # Process documents
    retriever.process_documents(documents)
    
    # Example query
    query = "How do computers understand human language?"
    similar_docs = retriever.find_similar_documents(query, top_k=2)
    
    print(f"Query: {query}\n")
    print("Most similar documents:")
    for doc_id, similarity in similar_docs:
        print(f"\nDocument ID: {doc_id}")
        print(f"Similarity Score: {similarity:.4f}")
        print(f"Content: {retriever.get_document_content(doc_id)}")

if __name__ == "__main__":
    main()
