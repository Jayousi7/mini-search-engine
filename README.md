# Mini Search Engine + RAG
A modular Information Retrieval (IR) pipeline featuring a custom-built search engine and a Retrieval-Augmented Generation (RAG) extension. This project combines local, math-based document retrieval with cloud-based LLM synthesis for intelligent, context-aware answers.
## System Architecture
The project is structured following standard Information Retrieval principles, broken down into clean, modular components:
### A. Data Ingestion & Preprocessing (parser.py)
 * **Source:** Cranfield 1400 Dataset (Aeronautical engineering abstracts).
 * **Pipeline - Tokenization:** Segmenting text into individual words.
 * **Pipeline - Normalization:** Lowercasing and punctuation removal.
 * **Pipeline - Stop-word Removal:** Filtering out high-frequency words (e.g., "and", "the").
 * **Pipeline - Stemming:** Reducing words to their base roots using the **Porter Stemmer**.
 * **Result:** The parser stores two versions of the data: **Stemmed Tokens** for mathematical retrieval and **Raw Text** for AI generation and UI display.
### B. Core Retrieval Engine (engine.py)
 * **Inverted Index:** A dictionary-based structure mapping terms to document IDs.
 * **TF-IDF Weighting - Term Frequency (TF):** Logarithmically scaled as 1+\log_{10}(count).
 * **TF-IDF Weighting - Inverse Document Frequency (IDF):** Calculated as \log_{10}(N/df_t).
 * **Cosine Similarity:** The engine ranks documents by calculating the cosine of the angle between the Query vector (Q) and the Document vector (D).
### C. RAG Extension (rag.py)
 * **Augmentation:** Top-ranked documents retrieved by the core engine are used as factual context.
 * **Generation:** Uses the **LLaMA-3.1-8B-Instant** model for fast, efficient natural language synthesis.
### D. Web Interface (server.py & templates/)
 * **Backend:** A **Flask** server providing a REST API for search operations and system evaluation.
 * **Frontend:** A premium HTML/CSS dashboard featuring real-time AJAX search.
## System Evaluation (evaluate_sys.py)
The retrieval engine's accuracy has been benchmarked using standard IR metrics:

| Metric | Score | Description |
| :--- | :--- | :--- |
| **MAP** | **0.2146** | **Mean Average Precision**: Measures average precision across all queries, rewarding highly relevant documents placed at the top of the search results. |
| **NDCG** | **0.3302** | **Normalized Discounted Cumulative Gain**: Measures overall ranking quality based on graded relevance. |

## Local vs. Cloud Execution
 * **Retrieval (100% Local):** The inverted index and all mathematical similarity computations run securely on your local hardware.
 * **AI Answer (Hybrid):** Context is retrieved locally, while natural language generation is handled rapidly via the Groq API.
 * **Cross-Platform:** The system uses standardized paths (/) to ensure seamless compatibility between Windows and Linux/Docker environments.
## Deployment & Containerization
The project is fully containerized using **Docker** for a frictionless, "works on my machine" setup.
 * **Dockerfile:** Configures the Python 3.11 environment and installs all necessary dependencies (NLTK, Flask, Groq).
 * **Docker Compose:** Orchestrates the deployment, linking the necessary services and ports.
### Quick Start
You can spin up the entire system with a single command:
> docker-compose up --build
>