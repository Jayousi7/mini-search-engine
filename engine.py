import math
from collections import defaultdict, Counter
from parser import Parser
class SearchEngine:
    def __init__(self, parsed_docs: dict[int, list[str]], raw_docs: dict[int, str]):
        self.docs = parsed_docs
        self.raw_docs = raw_docs
        self.total_docs = len(self.docs.keys())
        self.inverted_index = defaultdict(dict)
        self.idf = {}
        self.doc_vectors = defaultdict(dict)
    def build_idx(self):
        for doc_id,tokens in self.docs.items():
            term_frequencies = Counter(tokens)
            for word,count in term_frequencies.items():
                self.inverted_index[word][doc_id] = 1 + math.log10(count)
    def compute_idf(self):
        for word,doc_dict in self.inverted_index.items():
            doc_freq = len(doc_dict)
            self.idf[word] = math.log10(self.total_docs/doc_freq)
    def compute_tfidf(self):
        for word,doc_dict in self.inverted_index.items():
            for doc_id,tf in doc_dict.items():
                self.doc_vectors[doc_id][word] = tf * self.idf.get(word, 0)
    def compute_doc_lengths(self):
        self.doc_lengths = {}
        for doc_id ,vector in self.doc_vectors.items():
            sum_suares = sum(weight** 2 for weight in vector.values())
            self.doc_lengths[doc_id] = math.sqrt(sum_suares)
    def search(self,query:str,k:int = 10):
        if isinstance(query,str):
            query = Parser('')._preprocoess_pipeline(query)
        query_tf = Counter(query)
        query_sum_squares = 0.0
        query_vec = {}
        for word,count in query_tf.items():
            if word in self.idf:
                log_tf = 1 + math.log10(count)
                weight = log_tf * self.idf[word]
                query_vec[word] = weight
                query_sum_squares += weight ** 2 
        query_len = math.sqrt(query_sum_squares)
        if query_len == 0:
            return []
        scores = defaultdict(float)
        for word, query_weight in query_vec.items():
            for doc_id in self.inverted_index[word].keys():
                doc_weight = self.doc_vectors[doc_id][word]
                scores[doc_id] += query_weight * doc_weight
        for doc_id in scores:
            scores[doc_id] = scores[doc_id] /(self.doc_lengths[doc_id] * query_len)
        ranked_docs = sorted(scores.items(),key = lambda item : item[1],reverse = True )
        return ranked_docs[:k]