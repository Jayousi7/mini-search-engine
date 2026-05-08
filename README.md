# Mini Search Engine + RAG

A hybrid search system combining classical Information Retrieval (IR) with AI-powered Retrieval-Augmented Generation (RAG). It retrieves relevant documents locally using TF-IDF and cosine similarity, then generates intelligent answers using an LLM.

## How It Works

### 1. Data Ingestion & Preprocessing
- **Source:** Cranfield 1400 Dataset (aeronautical engineering documents)
- **Pipeline:** Tokenization → Normalization → Stop-word removal → Stemming
- **Output:** Indexed documents ready for retrieval

### 2. Retrieval Engine
- **Inverted Index:** Maps terms to document IDs for fast lookup
- **TF-IDF Scoring:** Ranks documents by relevance to the query
- **Cosine Similarity:** Measures semantic similarity between queries and documents

### 3. RAG Answer Generation
- Top-ranked documents are used as context
- LLaMA-3.1 model generates human-readable answers based on retrieved content

### 4. Web Interface
- **Backend:** Flask REST API for search and evaluation
- **Frontend:** Interactive dashboard with real-time search results
## Performance Metrics

| Metric | Score |
| :--- | :--- |
| **MAP (Mean Average Precision)** | **0.2722** |
| **NDCG (Normalized Discounted Cumulative Gain)** | **0.5108** |

## Architecture

- **Retrieval:** 100% local - runs on your machine using TF-IDF and cosine similarity
- **AI Generation:** Hybrid - uses Groq API (LLaMA-3.1 model) for fast answer synthesis
- **Cross-Platform:** Works on Windows, Linux, and Mac through standardized paths

## Deployment

The project is fully containerized using Docker for easy setup and deployment.

### Quick Start
- create a .env file that containes your api key GROQ_API_KEY=
- ```bash 
docker-compose up --build
```

Then navigate to `http://localhost:5000` in your browser.

## Project Structure

- `parser.py` - Text preprocessing and tokenization
- `engine.py` - TF-IDF search engine with ranking
- `rag.py` - RAG integration with LLM
- `server.py` - Flask backend
- `templates/index.html` - Web interface
- `cran/` - Cranfield dataset

## Requirements

- Python 3.11+
- GROQ API key (for AI answer generation)