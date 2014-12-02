import Tokenize
import Ngrams
import TaggingTools
from nltk import LancasterStemmer
from textblob_aptagger import PerceptronTagger

def typeTokenRatio(lengthFilter=None):
    def feature(text):
        text = " ".join(text)
        tokens = Tokenize.byWord(text)
        if lengthFilter != None:
            tokens = [token for token in tokens if len(token) >= lengthFilter]    
        types = set(tokens)
        return {"type/token" : int(100*len(types)/len(tokens))}
    return feature

#for Naive Bayes
def typeTokenRatioBucketed(lengthFilter=None):
    def feature(text):
        text = " ".join(text)
        tokens = Tokenize.byWord(text)
        if lengthFilter != None:
            tokens = [token for token in tokens if len(token) >= lengthFilter]    
        types = set(tokens)
        return {"type/tokenb" :"HIGH" if len(types)/len(tokens) > .5 else "MEDIUM" if len(types)/len(tokens) > .2 else "LOW"}
    return feature
#for Naive Bayes
def vocabSizeBucketed(lengthFilter=None):
    def feature(text):
        text = " ".join(text)
        tokens = Tokenize.byWord(text)
        if lengthFilter != None:
            tokens = [token for token in tokens if len(token) >= lengthFilter]    
        types = set(tokens)
        return {"vocabSize" :"HIGH" if len(types) > 50 else "MEDIUM" if len(types) > 20 else "LOW"}
    return feature

def vocabSize(lengthFilter=None):
    def feature(text):
        text = " ".join(text)
        tokens = Tokenize.byWord(text)
        if lengthFilter != None:
            tokens = [token for token in tokens if len(token) >= lengthFilter]    
        types = set(tokens)
        return {"vocabSizeb" :len(types)}
    return feature

def wordLengthDist(text):
    text = " ".join(text)
    words = Tokenize.byWordAlphaOnly(text)
    vector = {}
    total = 0
    for i in range(1,11):
        vector["%ofwords"+str(i)+"long"] = 0
    count = 0
    words = list(set(words))
    for word in words:
        if len(word) < 10:
            vector["%ofwords"+str(len(word))+"long"] += 1 
        else:
            vector["%ofwords"+str(10)+"long"] += 1
        total +=1
    for i in range(1,11):
        vector["%ofwords"+str(i)+"long"] = int(100*vector["%ofwords"+str(i)+"long"]/total)
    return vector

def avgWordLength(text):
    text = " ".join(text)
    tokens = Tokenize.byWordAlphaOnly(text)
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
    text = " ".join(text)
    numericValue = avgWordLength(text)["AVG word Length"]
    bucketLabel = "Long" if numericValue > 5 else "Medium" if numericValue > 3 else "Short"
    return {"AVG word Length b" : bucketLabel}

def topMCharacterNgrams(m ,n):
    def feature(text):
        text = " ".join(text)
        tokens = Tokenize.byWord(text)
        fd = Ngrams.getNgramFreqDist(text,n)
        topM = sorted([item for item in fd.items()],key=lambda x:x[1], reverse=True)[:m]
        vector = {}
        for i in range(len(topM)):
            vector["char#"+str(i)+" "+str(n)+"gramC"] = topM[i][0]
        return vector
    return feature

def topMWordNgrams(m ,n, stem=False):
    def feature(text):
        text = " ".join(text)
        tokens = Tokenize.byWord(text)
        words=[]
        if stem:
            words = Tokenize.byWordStem(text)
        else:
            words = Tokenize.byWordAlphaOnly(text)
        fd = Ngrams.getNgramFreqDist(words,n)
        topM = sorted([item for item in fd.items()],key=lambda x:x[1], reverse=True)[:m]
        vector = {}
        for i in range(len(topM)):
            vector["word#"+str(i)+" "+str(n)+"gramW"] = topM[i][0]
        return vector
    return feature

def textLength(text):
    text = " ".join(text)
    tokens = Tokenize.byWord(text)
    return {"text Length" : len(Tokenize.byWord(text))}

def percentOfLetters(text):
    text = " ".join(text)
    tokens = Tokenize.byWord(text)
    vector = {}
    total = 0
    for i in range(26):
        vector["pL"+chr(i + ord('a'))] = 0
    for c in text.lower():
        if "pL"+c in vector.keys():
            vector["pL"+c] +=1
            total += 1
    for i in range(26):
        vector["pL"+chr(i + ord('a'))] = int(100*(vector["pL"+chr(i + ord('a'))]/total))
    return vector

def percentOfUpperLetters(text):
    text = " ".join(text)
    tokens = Tokenize.byWord(text)
    uppers = 0
    total = 0    
    for c in text:
        if c.isupper():
            uppers +=1
        total += 1    
    percent = int(100*uppers/total)
    return {"percentUpperCase" : percent}

#Need POS tagger for better features
#We Have a POS tagger!

def topMPOSNgrams(m ,n):
    def feature(text):
        text = " ".join(text)
        tokens = Tokenize.byWord(text)
        POStags = [tag for word, tag in TaggingTools.tagPOS(text)]
        fd = Ngrams.getNgramFreqDist(POStags,n)
        topM = sorted([item for item in fd.items()],key=lambda x:x[1], reverse=True)[:m]
        vector = {}
        for i in range(len(topM)):
            vector["pos#"+str(i)+" "+str(n)+"gram"] = topM[i][0]
        return vector
    return feature

def posDist(text):
    text = " ".join(text)
    tokens = Tokenize.byWord(text)
    POStags = [tag for word, tag in TaggingTools.tagPOS(text)]
    possibleTags = PerceptronTagger().model.classes
    vector = {}
    total = 0
    for tag in possibleTags:
        vector[tag] = 0
    for tag in POStags:
        vector[tag] += 1
        total +=1
    for tag in possibleTags:
        vector[tag] = int(100*vector[tag]/total)
    return vector

#Testing
'''
testext = "ababababababab abx abz aby abt bcbcbcbcbcb dgdgdgd rtrt ff dd dd eg dd eg tt ww xxx www www www"
def testtopMCharacterNgrams():
    print(topMCharacterNgrams(testext,5,2))
    print(topMCharacterNgrams(testext,5,3))

testwords = "We should probably do some Sanity Checks on the Data (e.g. Average # of sentences per answers, avg # of words per answers, Avg # number of answers per Author"
def testtopMWordNgrams():
    print(topMWordNgrams(testext,3,2))
    print(topMWordNgrams(testwords,3,2))
    print(topMWordNgrams(testwords,3,3))

def testAvgWordLength():
    print(avgWordLengthBucketed(testwords))
    print(avgWordLength(testwords))

def testwordLengthDist():
    print(wordLengthDist(testext))
    print(wordLengthDist(testwords))

def testtopMPOSNgrams():
    print(topMPOSNgrams(testwords, 3, 2))

def testposDist():
    print(posDist(testwords))

def testPercents():
    print(sorted([item for item in percentOfLetters(testwords).items()]))
    print(percentOfUpperLetters(testwords))

#testtopMCharacterNgrams()
#testtopMWordNgrams()
#testAvgWordLength()
#testwordLengthDist()
#testtopMPOSNgrams()
#testposDist()
#testPercents()'''
