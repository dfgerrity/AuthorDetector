import nltk
from nltk import LancasterStemmer
import Tokenize

def loadHSWords(filename="words/happyAndSadWords3.txt"):
    f = open(filename)
    happySadScoredWords = []
    for line in f.readlines():
        ws = line.split()
        if len(ws) > 1:
            happySadScoredWords.append({"word":ws[0].strip(), "score":(int(ws[len(ws)-1].strip()))})
    return happySadScoredWords

def featureBinaryScore(sample):
    words = Tokenize.byWord(sample)  
    HSWords = loadHSWords()
    sentimentWordCount = 0
    score = 0
    for w in words:
        for s in HSWords:
            if w == s["word"]:
                score += s["score"]
                sentimentWordCount +=1
    #print("Raw score",score)
    score = int(score / (sentimentWordCount if sentimentWordCount > 0 else 1))
    rating = "+" if score > -1 else "-"
    #print("Ours:", rating, "Score", score)
    return {"HS rating" : rating}

def happySadClassifier(happySadScoredWords, taggedSamples):
    '''Expects happySadScoredWords to be list of dict {"word":word, "score":score}
       Expects each taggedSample in the form {"text" : "text", "overAllRating": number}'''
    pattern = r'''(?x) # set flag to allow verbose regexps
    ([A-Z]\.)+ # abbreviations, e.g. U.S.A.
    | \w+(-\w+)* # words with optional internal hyphens
    | \$?\d+(\.\d+)?%? # currency and percentages, e.g. $12.40, 82%
    | \.\.\. # ellipsis
    | [][.,;"'?():-_`]
    '''
    
    lm = LancasterStemmer()

    print("Stemming HappySad words")
    happySadStems = []
    for entry in happySadScoredWords:#lm.stem(entry["word"])
        happySadStems.append({"stem":entry["word"], "score" : entry["score"], "useCount": 0})    
    dups = []

    print("Finding duplicate stems")
    for i in range(len(happySadStems)):
        for j in range(i,len(happySadStems)):
            if happySadStems[i]["stem"] == happySadStems[j]["stem"] and not happySadStems[i]["stem"] in dups:
                dups.append(happySadStems[j]["stem"])
    #de-duplicate
    print("Removing duplicate stems")
    seen = []
    uniqueHappySadStems = []
    for i in range(len(happySadStems)):
        if not happySadStems[i]["stem"] in dups or not happySadStems[i]["stem"] in seen:
            uniqueHappySadStems.append(happySadStems[i])
            seen.append(happySadStems[i]["stem"])
    
    print("Rating samples")         
    sCount = 0          
    for sample in taggedSamples:        
        sCount +=1
        score = 0
        sentimentWordCount = 0
        words = (nltk.regexp_tokenize(sample["text"].lower(), pattern))
        for w in words:
            stem = w#lm.stem(w)
            for s in uniqueHappySadStems:
                if stem == s["stem"]:
                    s["useCount"] += 1
                    score += s["score"]
                    sentimentWordCount +=1
        score = int(score / (sentimentWordCount if sentimentWordCount > 0 else 1))
        rating = 5 if score > 3 else 4 if score > 2 else 3 if score > -1 else 2 if score > -3 else 1
        sample.update({"auto-rating":rating})
        print("Sample #", sCount, "Human:", sample["overAllRating"], "Ours:", rating, "Score", score)
        if abs(sample["overAllRating"] - sample["auto-rating"]) > 2:
            print("Really missed this one:")
            print(sample["text"]) 
    
    print("Computing accuracy")
    correct = 0
    for s in taggedSamples:
        if ((s["overAllRating"] in [3,4,5] and s["auto-rating"] in [3,4,5]) or
        (s["overAllRating"] in [1,2] and s["auto-rating"] in [1,2])):
            correct += 1        

    accuracy = correct / len(taggedSamples)
    print("Accuracy: ", accuracy)
    print("Most used sentiment stems:")
    print(sorted([entry for entry in uniqueHappySadStems], key=lambda x: x["useCount"], reverse=True)[:5])
    return taggedSamples
