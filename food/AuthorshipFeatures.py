import Tokenize

def typeTokenRatio(text, lengthFilter=None):
    tokens = Tokenize.byWord(text)
    if lengthFilter != None:
        tokens = [token for token in tokens if len(token) >= lengthFilter]    
    types = set(tokens)
    return {"type/token" : len(types)/len(tokens)}

#for Naive Bayes
def typeTokenRatioBucketed(text, lengthFilter=None):
    tokens = Tokenize.byWord(text)
    if lengthFilter != None:
        tokens = [token for token in tokens if len(token) >= lengthFilter]    
    types = set(tokens)
    return {"type/token" :"HIGH" if len(types)/len(tokens) > .5 else "MEDIUM" if len(types)/len(tokens) > .2 else "LOW"}

def vocabSize(text, lengthFilter=None):
    tokens = Tokenize.byWord(text)
    if lengthFilter != None:
        tokens = [token for token in tokens if len(token) >= lengthFilter]    
    types = set(tokens)
    return {"vocabSize" :"HIGH" if len(types) > 50 else "MEDIUM" if len(types) > 20 else "LOW"}

#for Naive Bayes
def vocabSizeBucketed(text, lengthFilter=None):
    tokens = Tokenize.byWord(text)
    if lengthFilter != None:
        tokens = [token for token in tokens if len(token) >= lengthFilter]    
    types = set(tokens)
    return {"vocabSize" :len(types)}

#Need POS tagger for better features
