from collections import defaultdict
from parser import Parser
from engine import SearchEngine
def map_relevance(score):
    map = {
        1:4,
        2:3,
        3:2,
        4:1,
       -1:0}
    
    return map[score]
def evaluate():
    gt = defaultdict(dict)
    
    with open(r'cran\cranqrel','r') as file:
        for line in file:
            info = line.split()
            query_id = int(info[0])
            doc_id = int(info[1])
            relevance_score = int(info[2])
            
            gt[query_id][doc_id] = map_relevance(relevance_score)

    query_pars = Parser(r'cran\cran.qry')
    query_pars.parse()
    
    doc_pars = Parser(r'cran\cran.all.1400')
    doc_pars.parse()
    
    SE = SearchEngine(doc_pars.tokens)
    SE.build_idx()
    SE.compute_idf()
    SE.compute_tfidf()
    SE.compute_doc_lengths()
    all_aps = list()

    for q_id,(_,query) in enumerate(query_pars.tokens.items(),start = 1):
        
        total_true_relevant = sum(1 for rel_score in gt[q_id].values() if rel_score > 0 )
        
        if total_true_relevant == 0:
            all_aps.append(0.0)
            continue
        hits = 0 
        sum_precisions = 0 
        search_result = SE.search(query,k=10)
        for rank , (d_id,score) in enumerate(search_result,start=1):
            if d_id in gt[q_id] and gt[q_id][d_id]>0:
                hits+=1
                sum_precisions +=hits/rank
        
        AP = sum_precisions / total_true_relevant
        all_aps.append(AP)
        
    final_map_score = sum(all_aps)/len(all_aps)
    print(f"Final MAP Score: {final_map_score}")      
          
if __name__ == '__main__':
    evaluate()
        
        
            
