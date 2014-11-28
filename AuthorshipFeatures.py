import Tokenize
import Ngrams
import TaggingTools
from nltk import LancasterStemmer

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

#for Naive Bayes
def vocabSizeBucketed(text, lengthFilter=None):
    tokens = Tokenize.byWord(text)
    if lengthFilter != None:
        tokens = [token for token in tokens if len(token) >= lengthFilter]    
    types = set(tokens)
    return {"vocabSize" :"HIGH" if len(types) > 50 else "MEDIUM" if len(types) > 20 else "LOW"}

def vocabSize(text, lengthFilter=None):
    tokens = Tokenize.byWord(text)
    if lengthFilter != None:
        tokens = [token for token in tokens if len(token) >= lengthFilter]    
    types = set(tokens)
    return {"vocabSize" :len(types)}

def avgWordLength(text):
    tokens = Tokenize.byWord(text)
    sum = 0
    count = 0
    tokens = list(set(tokens))
    for token in tokens:
        if token.isalpha():
            sum += len(token)
            count +=1
    return {"AVG word Length" : int(sum/count)}

# Bucketed version
def avgWordLengthBucketed(text):
    numericValue = avgWordLength(text)["AVG word Length"]
    bucketLabel = "Long" if numericValue > 5 else "Medium" if numericValue > 3 else "Short"
    return {"AVG word Length" : bucketLabel}

def topMCharacterNgrams(text, m ,n):
    fd = Ngrams.getNgramFreqDist(text,n)
    topM = sorted([item for item in fd.items()],key=lambda x:x[1], reverse=True)[:m]
    vector = {}
    for i in range(len(topM)):
        vector["#"+str(i)+" "+str(n)+"gram"] = topM[i][0]
    return vector

def topMWordNgrams(text, m ,n, stem=False):
    words=[]
    if stem:
        words = Tokenize.byWordStem(text)
    else:
        words = Tokenize.byWordAlphaOnly(text)
    fd = Ngrams.getNgramFreqDist(words,n)
    topM = sorted([item for item in fd.items()],key=lambda x:x[1], reverse=True)[:m]
    vector = {}
    for i in range(len(topM)):
        vector["#"+str(i)+" "+str(n)+"gram"] = topM[i][0]
    return vector

def textLength(text):
    return {"text Length" : len(Tokenize.byWord(text))}

#Need POS tagger for better features
#We Have a POS tagger!

def topMPOSNgrams(text, m ,n):
    POStags = [tag for word, tag in TaggingTools.tagPOS(text)]
    fd = Ngrams.getNgramFreqDist(POStags,n)
    topM = sorted([item for item in fd.items()],key=lambda x:x[1], reverse=True)[:m]
    vector = {}
    for i in range(len(topM)):
        vector["#"+str(i)+" "+str(n)+"gram"] = topM[i][0]
    return vector

#Testing
testext = "ababababababab abx abz aby abt bcbcbcbcbcb dgdgdgd rtrt ff dd dd eg dd eg tt ww xxx www www www"
def testtopMCharacterNgrams():
    print(topMCharacterNgrams(testext,5,2))
    print(topMCharacterNgrams(testext,5,3))

testwords = "We should probably do some sanity checks on the data (e.g. average # of sentences per answers, avg # of words per answers, avg # number of answers per author"
def testtopMWordNgrams():
    print(topMWordNgrams(testext,3,2))
    print(topMWordNgrams(testwords,3,2))
    print(topMWordNgrams(testwords,3,3))

def testAvgWordLength():
    print(avgWordLengthBucketed(testwords))
    print(avgWordLength(testwords))


#testtopMCharacterNgrams()
#testtopMWordNgrams()
#testAvgWordLength()

