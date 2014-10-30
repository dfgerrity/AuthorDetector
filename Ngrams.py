from nltk import FreqDist

def getNgrams(tokens, n):
    ngrams
    for i in range(len(tokens)-(n-1)):
        gram = []
        for j in range(i, i+n-1):
            gram.append(token[j])
        ngrams.append(gram)
    return ngrams

def getNgramFreqDist(tokens, n):
    return FreqDist(getNgrams(tokens, n))

