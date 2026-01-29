# Offline AI Assistant

A completely offline AI assistant that works like ChatGPT but runs locally on your machine. It can chat with you, read PDFs/notes/code, and answer questions using your local files - all without internet connectivity or paid APIs.

## Features

- 💬 **Chat Interface**: Conversational AI experience similar to ChatGPT
- 📄 **Document Processing**: Read and analyze PDFs, text files, code files, and more
- 🔍 **RAG System**: Retrieval-Augmented Generation for answering questions from your documents
- 🧠 **Local Models**: Uses open-source models running completely offline
- 💾 **Persistent Storage**: Vector database stores embeddings for fast retrieval
- 🔄 **Conversation Memory**: Maintains context across conversation turns
- ⚙️ **Configurable**: Multiple model options and adjustable parameters

## Tech Stack

- **LLM**: LLaMA 3, Mistral, Phi-3 (or any model supported by Ollama)
- **Runner**: [Ollama](https://ollama.ai/) for local inference
- **Vector Database**: FAISS for efficient similarity search
- **Embeddings**: sentence-transformers for text encoding
- **UI**: Streamlit for the web interface
- **Document Processing**: PyMuPDF, PyPDF2 for PDF handling

## Prerequisites

1. **Python 3.8+**
2. **Ollama** - Download from [ollama.ai](https://ollama.ai/)
3. **At least 8GB RAM** (16GB+ recommended for better performance with larger models)
4. **Storage space** for models (varies by model size, typically 4-13GB)

## Installation

1. Clone or download this repository:
```bash
git clone <repository-url>
cd offline-ai-assistant
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install and run Ollama:
   - Download from [ollama.ai](https://ollama.ai/)
   - Follow installation instructions for your platform
   - Start the Ollama service

4. Pull a model (e.g., Llama 3):
```bash
ollama pull llama3
```
Alternatively, you can use mistral or phi3:
```bash
ollama pull mistral
# or
ollama pull phi3
```

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Open your browser to the provided URL (usually `http://localhost:8501`)

3. In the sidebar, select your preferred model and adjust settings

4. Upload documents using the file uploader (supports PDF, TXT, PY, JS, HTML, CSS, MD, JSON, XML, CSV)

5. Start chatting in the main interface!

## How It Works

1. **Document Ingestion**: Uploaded documents are processed, chunked, and converted to embeddings
2. **Vector Storage**: Embeddings are stored in a FAISS vector database for fast similarity search
3. **Query Processing**: When you ask a question, it's converted to an embedding
4. **Retrieval**: The system finds the most relevant document chunks using vector similarity
5. **Generation**: The LLM generates a response based on your question and the retrieved context

## Configuration

### Models
The assistant supports various open-source models:
- Llama 3 (`llama3`)
- Mistral (`mistral`)
- Phi-3 (`phi3`)
- Gemma (`gemma`)
- And many others supported by Ollama

To use a different model, either:
- Select from the dropdown in the app sidebar
- Or pull the model with: `ollama pull <model-name>`

### Parameters
- **Temperature**: Controls randomness (0.0 = deterministic, 1.0 = creative)
- **Chunk Size**: Size of text chunks for document processing
- **Top-K**: Number of documents to retrieve for each query

## Privacy & Security

- ✅ **Completely Offline**: All processing happens on your machine
- ✅ **No Data Transmission**: Your documents never leave your computer
- ✅ **Local Storage**: Embeddings stored locally on your device
- ✅ **Open Source**: All components are open source and auditable

## Troubleshooting

### Common Issues

1. **"Model not found" Error**: Make sure you've pulled the model with `ollama pull <model-name>`

2. **Out of Memory / Insufficient Memory Errors**: This is a common issue when running large models. Solutions:
   - Use smaller models like `phi3:mini` or `gemma:2b`
   - Close other applications to free up RAM
   - Increase virtual memory/page file size in Windows
   - Consider using CPU-only models which tend to be smaller

3. **Ollama Not Responding**: Ensure Ollama service is running

4. **Slow Responses**: Performance depends on your hardware; consider a smaller model for slower machines

### Performance Tips

- Use SSD storage for faster vector database access
- More RAM allows for larger context windows
- GPU acceleration can be enabled in Ollama if you have compatible hardware

## Customization

You can customize the assistant by modifying:
- `document_processor.py`: Add support for new file types
- `rag_engine.py`: Adjust retrieval algorithms
- `llm_connector.py`: Modify model parameters
- `app.py`: Change UI elements and layout

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests for:
- Bug fixes
- New features
- Performance improvements
- Additional file format support

## License

This project is open source and available under the MIT License.

---

Enjoy your private, offline AI assistant! All your data stays on your machine, and you have full control over the models and functionality.