import Tokenize

def typeTokenRatio(text, lengthFilter=None):
    tokens = Tokenize.byWord(text)
    if lengthFilter != None:
        tokens = [token for token in tokens if len(token) >= lengthFilter]    
    types = set(tokens)
    return types/tokens

#for Naive Bayes
def typeTokenRatioBucketed(text, lengthFilter=None):
    tokens = Tokenize.byWord(text)
    if lengthFilter != None:
        tokens = [token for token in tokens if len(token) >= lengthFilter]    
    types = set(tokens)
    return "HIGH" if types/tokens > .5 else "MEDIUM" if types/tokens > .2 else "LOW"

def vocabSize(text, lengthFilter=None):
    tokens = Tokenize.byWord(text)
    if lengthFilter != None:
        tokens = [token for token in tokens if len(token) >= lengthFilter]    
    types = set(tokens)
    return "HIGH" if len(types) > 50 else "MEDIUM" if types/tokens > 20 else "LOW"

#for Naive Bayes
def vocabSizeBucketed(text, lengthFilter=None):
    tokens = Tokenize.byWord(text)
    if lengthFilter != None:
        tokens = [token for token in tokens if len(token) >= lengthFilter]    
    types = set(tokens)
    return len(types)

#Need POS tagger for better features
