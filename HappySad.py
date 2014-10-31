import nltk
from nltk import LancasterStemmer

def happySadClassifier(happySadScoredWords, taggedSamples):
    '''Expects happySadScoredWords to be dict {"word":word, "score":score}
       Expects each taggedSample in the form {"text" : [paragraph1, paragraph2...], "overAllRating": number}'''
    pattern = r'''(?x) # set flag to allow verbose regexps
    ([A-Z]\.)+ # abbreviations, e.g. U.S.A.
    | \w+(-\w+)* # words with optional internal hyphens
    | \$?\d+(\.\d+)?%? # currency and percentages, e.g. $12.40, 82%
    | \.\.\. # ellipsis
    | [][.,;"'?():-_`]
    '''
    
    lm = LancasterStemmer()

    happySadStems = []
    for entry in happySadWords:
        happySadStems.append({"stem":lm.stem(entry["word"]), "score" : entry["score"], "useCount": 0})    
    dups = []

    for i in range(len(happySadStems)):
        for j in range(i,len(happySadStems)):
            if happySadStems[i]["stem"] == happySadStems[j]["stem"]:
                dups.append(happySadStems[j])
    #de-duplicate
    uniqueHappySadStems = set(happySadStems).difference_update(set(dups)) 
                        
    for sample in taggedSamples:
        score = 0
        words =[]
        for p in sample["text"]:
            words.extend(nltk.regexp_tokenize("".join(p).lower(), pattern))

        for w in words:
            for s in uniqueHappySadStems:
                if lm.stem(w) == s["stem"]:
                    s["useCount"] += 1
                    score += s["score"]
        score = int(score / len(words))
        rating = 5 if score > 3 else 4 if score > 2 else 3 if score > 0 else 2 if score > -2 else 1
        sample.update({"auto-rating":rating})
    
    correct = 0
    for s in taggedSamples:
        if ((s["overAllRating"] in [3,4,5] and s["auto-rating"] in [3,4,5]) or
        (s["overAllRating"] in [1,2] and s["auto-rating"] in [1,2])):
            correct += 1

    accuracy = correct / len(taggedSamples)
    print("Accuracy: ", accuracy)
    print("Most used sentiment stems:")
    print(sorted([entry["stem"] for entry in uniqueHappySadStems], key=lambda x: x["useCount"])[-5:])
    return taggedSamples
