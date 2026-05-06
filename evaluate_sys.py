import math
from collections import defaultdict
from parser import Parser
from engine import SearchEngine

def map_relevance(score):
    relevance_map = {1:4, 2:3, 3:2, 4:1, -1:0}
    return relevance_map[score]

def evaluate():
    gt = defaultdict(dict)
    with open(r'cran/cranqrel','r') as file:
        for line in file:
            info = line.split()
            query_id = int(info[0])
            doc_id = int(info[1])
            relevance_score = int(info[2])
            gt[query_id][doc_id] = map_relevance(relevance_score)

    query_pars = Parser(r'cran/cran.qry')
    query_pars.parse()
    doc_pars = Parser(r'cran/cran.all.1400')
    doc_pars.parse()
    SE = SearchEngine(doc_pars.tokens, doc_pars.raw_texts)
    SE.build_idx()
    SE.compute_idf()
    SE.compute_tfidf()
    SE.compute_doc_lengths()
    all_aps = []
    all_ndcgs = []

    for q_id,(_,query) in enumerate(query_pars.tokens.items(),start = 1):
        total_true_relevant = sum(1 for rel_score in gt[q_id].values() if rel_score > 0 )
        if total_true_relevant == 0:
            all_aps.append(0.0)
            all_ndcgs.append(0.0)
            continue
        hits = 0 
        sum_precisions = 0 
        dcg = 0.0
        search_result = SE.search(query,k=10)
        for rank , (d_id,score) in enumerate(search_result,start=1):
            if d_id in gt[q_id] and gt[q_id][d_id]>0:
                hits+=1
                sum_precisions +=hits/rank
            rel = gt[q_id].get(d_id, 0)
            dcg += rel / math.log2(rank + 1)
        ideal_rels = sorted(gt[q_id].values(), reverse=True)
        idcg = 0.0
        for rank, rel in enumerate(ideal_rels[:10], start=1):
            idcg += rel / math.log2(rank + 1)
        all_aps.append(sum_precisions / total_true_relevant)
        all_ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        
    print(f"Final MAP Score: {sum(all_aps)/len(all_aps):.4f}")
    print(f"Final NDCG Score: {sum(all_ndcgs)/len(all_ndcgs):.4f}")
          
if __name__ == '__main__':
    evaluate()
