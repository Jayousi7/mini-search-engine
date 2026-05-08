from flask import Flask, render_template, request, jsonify
from parser import Parser
from engine import SearchEngine
from rag import generate_rag_answer
import os
import math
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv('.env')

app = Flask(__name__)

doc_pars = Parser(r'cran/cran.all.1400')
doc_pars.parse()
SE = SearchEngine(doc_pars.tokens, doc_pars.raw_texts)
SE.build_idx()
SE.compute_idf()
SE.compute_tfidf()
SE.compute_doc_lengths()

gt = defaultdict(dict)
with open(r'cran/cranqrel','r') as file:
    for line in file:
        info = line.split()
        query_id = int(info[0]); doc_id = int(info[1]); relevance_score = int(info[2])
        gt[query_id][doc_id] = {1:4, 2:3, 3:2, 4:1, -1:0}.get(relevance_score, 0)

query_pars = Parser(r'cran/cran.qry')
query_pars.parse()
all_aps, all_ndcgs = [], []
for q_id, (_, query) in list(enumerate(query_pars.tokens.items(), start=1))[:50]:
    total_true_relevant = sum(1 for rel_score in gt[q_id].values() if rel_score > 0 )
    if total_true_relevant == 0: continue
    search_result = SE.search(query, k=10)
    hits = 0; sum_precisions = 0; dcg = 0.0
    for rank, (d_id, score) in enumerate(search_result, start=1):
        if d_id in gt[q_id] and gt[q_id][d_id] > 0:
            hits += 1; sum_precisions += hits/rank
        dcg += gt[q_id].get(d_id, 0) / math.log2(rank + 1)
    ideal_rels = sorted(gt[q_id].values(), reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels[:10]))
    all_aps.append(sum_precisions / total_true_relevant)
    all_ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

SYSTEM_METRICS = {
    "map": round(sum(all_aps)/len(all_aps), 4) if all_aps else 0.0,
    "ndcg": round(sum(all_ndcgs)/len(all_ndcgs), 4) if all_ndcgs else 0.0
}

@app.route('/')
def index():
    return render_template('index.html', metrics=SYSTEM_METRICS)

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    results = SE.search(query, k=10)
    retrieved_docs = []
    for d_id, score in results:
        full_text = SE.raw_docs.get(d_id, "")
        retrieved_docs.append({
            "id": d_id,
            "score": round(score, 4),
            "text": full_text
        })
    try:
        ai_answer = generate_rag_answer(query, SE, k=3)
    except Exception as e:
        ai_answer = f"Error generating answer: {str(e)}"
    return jsonify({
        "results": retrieved_docs,
        "ai_answer": ai_answer
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
