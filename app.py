import streamlit as st
import os
from pathlib import Path
import tempfile
from document_processor import DocumentProcessor
from embedding_store import LocalEmbeddingStore
from rag_engine import AdvancedRAGEngine
from llm_connector import AdvancedLLMConnector


# Initialize session state
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = AdvancedRAGEngine(persist_path="./vector_store")
    
if 'llm_connector' not in st.session_state:
    st.session_state.llm_connector = AdvancedLLMConnector(model_name="llama3")
    
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []


def main():
    st.set_page_config(
        page_title="Offline AI Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Offline AI Assistant")
    st.markdown("""
    This is a completely offline AI assistant that can chat like ChatGPT, 
    process documents (PDFs/notes/code), and answer questions using your local files.
    """)
    
    # Sidebar for settings and file uploads
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        available_models = st.session_state.llm_connector.list_available_models()
        
        # Safely extract model names, handling potential variations in response structure
        if available_models:
            try:
                # Handle different possible structures of model objects
                if isinstance(available_models[0], dict):
                    if 'name' in available_models[0]:
                        model_names = [model['name'].split(':')[0] for model in available_models]
                    elif 'model' in available_models[0]:
                        model_names = [model['model'].split(':')[0] for model in available_models]
                    else:
                        # If neither 'name' nor 'model' key exists, use default models
                        model_names = ['llama3', 'mistral', 'phi3']
                else:
                    # If the response is not a list of dicts, use default models
                    model_names = ['llama3', 'mistral', 'phi3']
            except:
                # Fallback to default models if any error occurs
                model_names = ['llama3', 'mistral', 'phi3']
        else:
            model_names = ['llama3', 'mistral', 'phi3']
        
        # Ensure we have at least the default models available
        default_models = ['llama3', 'mistral', 'phi3']
        for model in default_models:
            if model not in model_names:
                model_names.append(model)
        
        selected_model = st.selectbox(
            "Select LLM Model",
            options=model_names,
            index=0 if 'llama3' in model_names else 0
        )
        
        if selected_model != st.session_state.llm_connector.model_name:
            try:
                st.session_state.llm_connector.set_model(selected_model)
                st.success(f"Model changed to {selected_model}")
            except ValueError as ve:
                st.error(f"Model error: {str(ve)}")
            except KeyError as ke:
                st.warning(f"Model API structure difference, but model set to {selected_model}")
            except Exception as e:
                st.error(f"Unexpected error changing model: {str(e)}")
        
        # Temperature setting
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        st.session_state.llm_connector.temperature = temperature
        
        # Upload documents section
        st.header("📚 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT, PY, etc.)",
            type=['pdf', 'txt', 'py', 'js', 'html', 'css', 'md', 'json', 'xml', 'csv'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # Save uploaded files temporarily and process them
            temp_files = []
            for uploaded_file in uploaded_files:
                # Only process new files
                if uploaded_file.name not in [f.name for f in st.session_state.uploaded_files]:
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}", mode='wb') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_files.append(tmp_file.name)
                    
                    st.session_state.uploaded_files.append(uploaded_file)
            
            if temp_files:
                with st.spinner("Processing documents..."):
                    try:
                        st.session_state.rag_engine.ingest_documents(temp_files)
                        st.success(f"Successfully processed {len(temp_files)} document(s)!")
                        
                        # Clean up temporary files
                        for temp_file in temp_files:
                            os.unlink(temp_file)
                            
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
                        # Clean up temporary files even if there's an error
                        for temp_file in temp_files:
                            os.unlink(temp_file)
        
        # Display uploaded files
        if st.session_state.uploaded_files:
            st.subheader("Uploaded Files")
            for file in st.session_state.uploaded_files:
                st.write(f"📄 {file.name}")
        
        # Knowledge base stats
        st.subheader("📊 Knowledge Base Stats")
        stats = st.session_state.rag_engine.get_document_stats()
        st.write(f"Documents: {len(set(stats.get('source_files', [])))}")
        st.write(f"Chunks: {st.session_state.rag_engine.embedding_store.get_count()}")
        
        # Clear knowledge base button
        if st.button("🗑️ Clear Knowledge Base"):
            st.session_state.rag_engine.clear_knowledge_base()
            st.session_state.uploaded_files = []
            st.success("Knowledge base cleared!")
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (role, message) in enumerate(st.session_state.chat_history):
                with st.chat_message(role):
                    st.write(message)
        
        # User input
        user_input = st.chat_input("Ask a question about your documents or anything else...")
        
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append(("user", user_input))
            
            # Display user message
            with chat_container:
                with st.chat_message("user"):
                    st.write(user_input)
            
            # Generate and display AI response
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            # Use RAG if there are documents in the knowledge base
                            if st.session_state.rag_engine.embedding_store.get_count() > 0:
                                response_data = st.session_state.rag_engine.query_with_memory(
                                    user_input, 
                                    st.session_state.llm_connector
                                )
                                response = response_data['response']
                            else:
                                # Just use the LLM directly if no documents
                                response = st.session_state.llm_connector.chat(user_input)
                            
                            st.write(response)
                            
                            # Add AI response to chat history
                            st.session_state.chat_history.append(("assistant", response))
                            
                        except Exception as e:
                            error_msg = f"Sorry, I encountered an error: {str(e)}"
                            st.write(error_msg)
                            st.session_state.chat_history.append(("assistant", error_msg))
    
    with col2:
        st.header("💡 How to Use")
        st.markdown("""
        1. **Upload Documents** - Add PDFs, text files, or code
        2. **Ask Questions** - Query about your documents or general topics
        3. **Adjust Settings** - Change models and temperature
        4. **Clear Memory** - Reset the knowledge base when needed
        """)
        
        st.header("🔍 Features")
        st.markdown("""
        - 📄 **Document Processing**: Supports PDF, TXT, PY, JS, HTML, and more
        - 🧠 **RAG System**: Retrieves relevant info from your docs
        - 💬 **Conversational**: Maintains context in conversations
        - 🌐 **Offline**: Works completely without internet
        - 🔍 **Smart Search**: Finds relevant information automatically
        """)


if __name__ == "__main__":
    main()