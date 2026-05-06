from groq import Groq
from parser import Parser
from engine import SearchEngine
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is not set! Please create a .env file and add your key.")
client = Groq(api_key=api_key)

def generate_rag_answer(query: str, search_engine: SearchEngine, k: int = 3):
    search_results = search_engine.search(query, k=k)
    if not search_results:
        return "No relevant documents found to answer your query."
    context_blocks = []
    for rank, (doc_id, score) in enumerate(search_results, start=1):
        doc_text = search_engine.raw_docs.get(doc_id, "")
        context_blocks.append(f"Document {doc_id}:\n{doc_text}\n")
    context = "\n".join(context_blocks)
    prompt = f"""You are a helpful research assistant. Use the provided documents to answer the user's query.
Note: The documents might be technical or preprocessed. Try to infer the meaning.
If the documents don't have the exact answer, try to summarize what is related in them, or explain what information is missing.

Retrieved Documents:
{context}

User Query: {query}
"""
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.1-8b-instant",
    )
    return chat_completion.choices[0].message.content

if __name__ == '__main__':
    doc_pars = Parser(r'cran/cran.all.1400')
    doc_pars.parse()
    SE = SearchEngine(doc_pars.tokens, doc_pars.raw_texts)
    SE.build_idx()
    SE.compute_idf()
    SE.compute_tfidf()
    SE.compute_doc_lengths()
    sample_query = "aircraft engine efficiency"
    try:
        rag_answer = generate_rag_answer(sample_query, SE, k=3)
        print(rag_answer)
    except Exception as e:
        print(f"Error: {e}")
