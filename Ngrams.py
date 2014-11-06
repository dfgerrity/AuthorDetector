from nltk import FreqDist

def getNgrams(tokens, n):
    ngrams = []
    for i in range(len(tokens)-(n)):
        gram = []
        for j in range(i, i+n):
            gram.append(tokens[j])
        ngrams.append(" ".join(gram))
    #print(ngrams)
    return ngrams

def gramsAsLists(grams):    
    return [g.split() for g in grams]

def getNgramFreqDist(tokens, n):  
    grams = getNgrams(tokens, n)
    return FreqDist(grams)

