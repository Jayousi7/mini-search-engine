# Mini Search Engine + RAG

## 1. System Architecture

The project is structured into a modular pipeline following standard Information Retrieval (IR) principles:

### A. Data Ingestion & Preprocessing (`parser.py`)
*   **Source:** Cranfield 1400 Dataset (Aeronautical engineering abstracts).
*   **Pipeline:** 
    *   **Tokenization:** Segmenting text into individual words.
    *   **Normalization:** Lowercasing and punctuation removal.
    *   **Stop-word Removal:** Filtering out high-frequency words (e.g., "and", "the").
    *   **Stemming:** Reducing words to their base roots using the **Porter Stemmer**.
*   **Result:** The parser stores two versions of the data: **Stemmed Tokens** for mathematical retrieval and **Raw Text** for AI generation and UI display.

### B. Core Retrieval Engine (`engine.py`)
*   **Inverted Index:** A dictionary-based structure mapping terms to document IDs.
*   **TF-IDF Weighting:** 
    *   **Term Frequency (TF):** Logarithmically scaled as $1 + \log_{10}(count)$.
    *   **Inverse Document Frequency (IDF):** Calculated as $\log_{10}(N / df_t)$.
*   **Cosine Similarity:** The engine ranks documents by calculating the cosine of the angle between the Query vector ($Q$) and the Document vector ($D$).

> **Cosine Similarity Formula:**
> $$\text{Score}(Q, D) = \frac{\sum_{i=1}^{n} w_{i,Q} \cdot w_{i,D}}{\sqrt{\sum_{i=1}^{n} w_{i,Q}^2} \cdot \sqrt{\sum_{i=1}^{n} w_{i,D}^2}}$$

### C. System Evaluation (`evaluate_sys.py`)
*   **Mean Average Precision (MAP):** **0.2146**
    *   Measures average precision across all queries, rewarding relevant documents at the top.
*   **Normalized Discounted Cumulative Gain (NDCG):** **0.3302**
    *   Measures ranking quality based on graded relevance.

### D. RAG Extension (`rag.py`)
*   **Augmentation:** Top-ranked documents are used as context for the AI.
*   **Generation:** Uses the **LLaMA-3.1-8B-Instant** model for fast, efficient natural language synthesis.

### E. Web Interface (`server.py` & `templates/`)
*   **Backend:** A Flask server providing a REST API for search and evaluation.
*   **Frontend:** A premium HTML/CSS dashboard with real-time AJAX search.

## 2. Deployment & Containerization
The project is fully containerized using **Docker**.
*   **Dockerfile:** Configures the Python 3.11 environment and installs all dependencies (NLTK, Flask, Groq).
*   **Docker Compose:** Orchestrates the deployment, allowing the system to be started with a single command: `docker-compose up --build`.

## 3. Local vs. Cloud Execution
*   **Retrieval:** 100% Local (Index and math run on your hardware).
*   **AI Answer:** Hybrid (Context is retrieved locally, generation via Groq API).
*   **Cross-Platform:** The system uses standardized paths (`/`) to ensure compatibility between Windows and Linux/Docker environments.
