import food
import HappySad

print("Loading Corpus...")
testReviews, trainingReviews = food.createReviewArray()

print("Loading Happy/Sad Words...")
happySadScoredWords = HappySad.loadHSWords("./words/happyAndSadWords2.txt")

#print(happySadScoredWords)
#print(trainingReviews)

#Tagging every paragraph individually
taggedParaGraphs = []
for r in trainingReviews:
    if len(r) < 13:
        print("Warning!, Someone didn't have all four paragraphs")
        #for l in r:
            #print(l +"\n")
    else:
        try:
            taggedParaGraphs.append({"overAllRating" : int(r[4].split(":")[1].strip()), "text": r[9]})
            taggedParaGraphs.append({"overAllRating" : int(r[5].split(":")[1].strip()), "text": r[10]})
            taggedParaGraphs.append({"overAllRating" : int(r[6].split(":")[1].strip()), "text": r[11]})
            taggedParaGraphs.append({"overAllRating" : int(r[7].split(":")[1].strip()), "text": r[12]})
        except:
            print("Bad format")

#print(taggedParaGraphs)
ourtagsAdded = HappySad.happySadClassifier(happySadScoredWords, taggedParaGraphs)