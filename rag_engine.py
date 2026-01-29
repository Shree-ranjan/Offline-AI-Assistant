from typing import List, Dict, Any, Optional
from document_processor import DocumentProcessor
from embedding_store import LocalEmbeddingStore
import re


class RAGEngine:
    """
    Implements Retrieval-Augmented Generation (RAG) system that combines document retrieval
    with language model generation to answer questions based on provided documents.
    """
    
    def __init__(self, persist_path: str = "./vector_store"):
        """
        Initialize the RAG engine.
        
        Args:
            persist_path: Path to persist the vector store
        """
        self.document_processor = DocumentProcessor()
        self.embedding_store = LocalEmbeddingStore(persist_path=persist_path)
        self.retrieved_docs = []
    
    def ingest_documents(self, file_paths: List[str], chunk_size: int = 1000, overlap: int = 100):
        """
        Ingest documents into the RAG system by processing, chunking, and storing embeddings.
        
        Args:
            file_paths: List of file paths to ingest
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
        """
        # Process documents
        processed_results = self.document_processor.process_documents(file_paths)
        
        all_texts = []
        all_metadatas = []
        all_doc_ids = []
        
        for file_path, result in processed_results.items():
            if 'error' not in result:
                chunks = result['chunks']
                for i, chunk in enumerate(chunks):
                    if chunk.strip():  # Only add non-empty chunks
                        all_texts.append(chunk)
                        all_metadatas.append({
                            'source_file': file_path,
                            'chunk_index': i,
                            'total_chunks': result['num_chunks']
                        })
                        all_doc_ids.append(f"{file_path}_chunk_{i}")
        
        # Add to embedding store
        if all_texts:
            self.embedding_store.add_texts(all_texts, all_metadatas, all_doc_ids)
            print(f"Ingested {len(all_texts)} chunks from {len(file_paths)} documents")
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 5, min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant documents for a given query.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of relevant documents with metadata
        """
        results = self.embedding_store.similarity_search(query, k=top_k)
        
        # Filter by minimum similarity
        filtered_results = [r for r in results if r['score'] >= min_similarity]
        
        # Store for potential later use
        self.retrieved_docs = filtered_results
        
        return filtered_results
    
    def format_context(self, retrieved_docs: List[Dict[str, Any]], max_length: int = 3000) -> str:
        """
        Format retrieved documents into a context string for the LLM.
        
        Args:
            retrieved_docs: List of retrieved documents
            max_length: Maximum length of context string
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        total_length = 0
        for doc in retrieved_docs:
            doc_text = f"\n--- Source: {doc['metadata'].get('source_file', 'Unknown')} ---\n{doc['document']}\n"
            
            if total_length + len(doc_text) > max_length:
                # Truncate the last document if needed
                remaining_chars = max_length - total_length
                if remaining_chars > 0:
                    truncated_doc = doc_text[:remaining_chars]
                    context_parts.append(truncated_doc)
                break
            else:
                context_parts.append(doc_text)
                total_length += len(doc_text)
        
        return "".join(context_parts)
    
    def generate_response(self, query: str, context: str, llm_connector) -> str:
        """
        Generate a response using the LLM with the provided context.
        
        Args:
            query: Original user query
            context: Retrieved context to augment the query
            llm_connector: LLM connector instance
            
        Returns:
            Generated response from the LLM
        """
        # Create a prompt that incorporates the context
        prompt = self.create_rag_prompt(query, context)
        
        # Generate response using the LLM
        response = llm_connector.generate(prompt)
        
        return response
    
    def create_rag_prompt(self, query: str, context: str) -> str:
        """
        Create a RAG-specific prompt that combines the query with the context.
        
        Args:
            query: Original user query
            context: Retrieved context
            
        Returns:
            Formatted prompt for the LLM
        """
        prompt = f"""Based on the following context, please answer the question. If the context doesn't contain the information needed to answer the question, say so.

Context:
{context}

Question: {query}

Answer: """
        return prompt
    
    def query(self, query: str, llm_connector, top_k: int = 5, min_similarity: float = 0.3) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve relevant documents and generate response.
        
        Args:
            query: User query
            llm_connector: LLM connector instance
            top_k: Number of top documents to retrieve
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary containing response and metadata
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_relevant_documents(
            query, 
            top_k=top_k, 
            min_similarity=min_similarity
        )
        
        # Format context from retrieved documents
        context = ""
        if retrieved_docs:
            context = self.format_context(retrieved_docs)
        else:
            context = "No relevant documents found in the knowledge base."
        
        # Generate response
        response = self.generate_response(query, context, llm_connector)
        
        return {
            'response': response,
            'retrieved_docs': retrieved_docs,
            'context_used': context,
            'query': query
        }
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the documents in the RAG system.
        
        Returns:
            Dictionary containing document statistics
        """
        stats = {
            'total_documents': len(set([doc['metadata']['source_file'] for doc in self.retrieved_docs])),
            'total_chunks': self.embedding_store.get_count(),
            'source_files': list(set([doc['metadata']['source_file'] for doc in self.retrieved_docs]))
        }
        return stats
    
    def clear_knowledge_base(self):
        """Clear all documents from the knowledge base."""
        self.embedding_store.clear()
        self.retrieved_docs = []
        print("Knowledge base cleared")


class AdvancedRAGEngine(RAGEngine):
    """
    Extended RAG engine with advanced features like conversation memory and reasoning.
    """
    
    def __init__(self, persist_path: str = "./vector_store"):
        super().__init__(persist_path)
        self.conversation_history = []
        self.max_history = 10  # Maximum number of conversation turns to retain
    
    def add_to_conversation_history(self, user_query: str, ai_response: str):
        """
        Add a query-response pair to the conversation history.
        
        Args:
            user_query: User's query
            ai_response: AI's response
        """
        self.conversation_history.append({
            'user_query': user_query,
            'ai_response': ai_response,
            'turn': len(self.conversation_history)
        })
        
        # Limit history size
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_conversation_context(self) -> str:
        """
        Get recent conversation history as context for the current query.
        
        Returns:
            Formatted conversation history
        """
        if not self.conversation_history:
            return ""
        
        context_parts = ["Recent conversation history:"]
        for entry in self.conversation_history[-3:]:  # Last 3 exchanges
            context_parts.append(f"User: {entry['user_query']}")
            context_parts.append(f"Assistant: {entry['ai_response']}")
        
        return "\n".join(context_parts)
    
    def query_with_memory(self, query: str, llm_connector, top_k: int = 5, min_similarity: float = 0.3) -> Dict[str, Any]:
        """
        Enhanced query method that incorporates conversation memory.
        
        Args:
            query: User query
            llm_connector: LLM connector instance
            top_k: Number of top documents to retrieve
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary containing response and metadata
        """
        # Get conversation context
        conversation_context = self.get_conversation_context()
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_relevant_documents(
            query, 
            top_k=top_k, 
            min_similarity=min_similarity
        )
        
        # Format context from retrieved documents
        knowledge_context = ""
        if retrieved_docs:
            knowledge_context = self.format_context(retrieved_docs, max_length=2000)
        else:
            knowledge_context = "No relevant documents found in the knowledge base."
        
        # Create enhanced prompt with both knowledge and conversation context
        prompt = self.create_enhanced_rag_prompt(query, knowledge_context, conversation_context)
        
        # Generate response
        response = llm_connector.generate(prompt)
        
        # Add to conversation history
        self.add_to_conversation_history(query, response)
        
        return {
            'response': response,
            'retrieved_docs': retrieved_docs,
            'knowledge_context': knowledge_context,
            'conversation_context': conversation_context,
            'query': query
        }
    
    def create_enhanced_rag_prompt(self, query: str, knowledge_context: str, conversation_context: str) -> str:
        """
        Create an enhanced RAG prompt that includes both knowledge and conversation context.
        
        Args:
            query: Original user query
            knowledge_context: Retrieved knowledge context
            conversation_context: Recent conversation history
            
        Returns:
            Formatted enhanced prompt for the LLM
        """
        parts = []
        
        if conversation_context:
            parts.append(f"Conversation History:\n{conversation_context}\n")
        
        parts.append(f"Reference Information:\n{knowledge_context}\n")
        parts.append(f"Current Question: {query}\n")
        parts.append("Based on the reference information and conversation history, please provide a helpful response:")
        
        return "\n".join(parts)