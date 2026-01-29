import numpy as np
import faiss
import pickle
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import os


class EmbeddingStore:
    """
    Manages embeddings using FAISS vector database for efficient similarity search.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding store with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.documents = []  # Store document texts
        self.metadata = []   # Store metadata for each document
        self.doc_ids = []    # Store document IDs
        
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Array of embeddings
        """
        embeddings = self.model.encode(texts)
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        return embeddings.astype('float32')
    
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None, ids: List[str] = None):
        """
        Add texts to the embedding store.
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of document IDs
        """
        if not texts:
            return
            
        embeddings = self.encode_texts(texts)
        
        # Add embeddings to the FAISS index
        self.index.add(embeddings)
        
        # Store documents and metadata
        self.documents.extend(texts)
        
        if metadatas is None:
            metadatas = [{}] * len(texts)
        self.metadata.extend(metadatas)
        
        if ids is None:
            ids = [f"doc_{len(self.documents)-i-1}" for i in range(len(texts), 0, -1)]
        self.doc_ids.extend(ids)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform similarity search against the embedding store.
        
        Args:
            query: Query text to search for
            k: Number of results to return
            
        Returns:
            List of dictionaries containing matched documents and metadata
        """
        query_embedding = self.encode_texts([query])
        
        # Perform similarity search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.documents):  # Valid index check
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(score),
                    'id': self.doc_ids[idx] if idx < len(self.doc_ids) else f"doc_{idx}"
                })
        
        return results
    
    def save(self, path: str):
        """
        Save the embedding store to disk.
        
        Args:
            path: Path to save the embedding store
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.index")
        
        # Save documents, metadata, and doc_ids
        data = {
            'documents': self.documents,
            'metadata': self.metadata,
            'doc_ids': self.doc_ids
        }
        
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """
        Load the embedding store from disk.
        
        Args:
            path: Path to load the embedding store from
        """
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.index")
        
        # Load documents, metadata, and doc_ids
        with open(f"{path}.pkl", 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data['documents']
        self.metadata = data['metadata']
        self.doc_ids = data['doc_ids']
        
        # Reinitialize the model dimension based on loaded index
        self.dimension = self.index.d
    
    def clear(self):
        """Clear all stored embeddings."""
        # Create a new empty index with the same dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.metadata = []
        self.doc_ids = []
    
    def get_count(self) -> int:
        """
        Get the number of documents in the store.
        
        Returns:
            Number of documents in the store
        """
        return self.index.ntotal


class LocalEmbeddingStore(EmbeddingStore):
    """
    Extension of EmbeddingStore that provides local file persistence.
    """
    
    def __init__(self, persist_path: str = "./vector_store", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the local embedding store.
        
        Args:
            persist_path: Path to persist the vector store
            model_name: Name of the sentence transformer model to use
        """
        super().__init__(model_name)
        self.persist_path = persist_path
        
        # Try to load existing store if it exists
        if os.path.exists(f"{persist_path}.index") and os.path.exists(f"{persist_path}.pkl"):
            try:
                self.load(persist_path)
                print(f"Loaded existing vector store from {persist_path}")
            except Exception as e:
                print(f"Failed to load existing vector store: {e}")
                # Initialize fresh store if loading fails
                self.__init__(model_name)
    
    def save(self):
        """Save the embedding store to the configured path."""
        super().save(self.persist_path)
        print(f"Saved vector store to {self.persist_path}")
    
    def add_documents_with_metadata(self, texts: List[str], file_paths: List[str], chunk_indices: List[int]):
        """
        Add documents with rich metadata including file paths and chunk indices.
        
        Args:
            texts: List of text chunks to add
            file_paths: Corresponding file paths for each text
            chunk_indices: Chunk indices for each text
        """
        metadatas = []
        for fp, ci in zip(file_paths, chunk_indices):
            metadatas.append({
                'source_file': fp,
                'chunk_index': ci,
                'length': len(texts[ci]) if ci < len(texts) else 0
            })
        
        self.add_texts(texts, metadatas)