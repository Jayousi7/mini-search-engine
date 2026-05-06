import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

class Parser:
    def __init__ (self, datapath: str ):
        self.data = datapath
        self.tokens = dict()
        self.raw_texts = dict()
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def parse(self):
        title = False
        words = False 
        raw_text = ''
        id = None 
        with open(self.data,'r') as file:
            for line in file:
                line = line.strip() 
                if not line:
                    continue
                if line.startswith('.I'):
                    if id is not None:
                        self.raw_texts[id] = raw_text.strip()
                        processed_tokens = self._preprocoess_pipeline(raw_text)
                        self.tokens[id] = processed_tokens
                    id =  int(line.split()[1]) 
                    raw_text = ''
                    words = False
                    title = False
                    continue
                elif line.startswith('.T'):
                    title = True 
                    words = False
                    continue
                elif line.startswith('.W'):
                    words = True 
                    title = False
                    continue
                elif line.startswith('.A') or line.startswith('.B'):
                    title = False
                    words = False 
                    continue
                if title or words :
                    raw_text += line + ' '
                else: continue
            if id is not None:
                self.raw_texts[id] = raw_text.strip()
                processed_tokens = self._preprocoess_pipeline(raw_text)
                self.tokens[id] = processed_tokens
        
    def tokenize(self, text:str) ->list[str]:
        return word_tokenize(text,language='english',)
    
    def normalize(self, text:str)->str:
        text = text.lower()
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def remove_stop_words(self, text:list[str])->list[str]:
        text = [word for word in text if word not in self.stop_words]
        return text
    
    def stem(self, text:list[str])->list[str] :
        return [self.stemmer.stem(word) for word in text]
    
    def _preprocoess_pipeline(self,text:str)->list[str]:
        normalized_text = self.normalize(text)
        tokens = self.tokenize(normalized_text)
        clean_tokens = self.remove_stop_words(tokens)
        stemmed_tokens = self.stem(clean_tokens)
        return stemmed_tokens
    
    def __getitem__(self, doc_id: int) -> list[str]:
        return self.tokens.get(doc_id, [])
    
if __name__ == '__main__':
    P = Parser(r'cran\cran.all.1400')
    P.parse()
    print(P.tokens[1])