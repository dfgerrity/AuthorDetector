import os 
os.chdir("C:/Users/Dan'l Boone/Documents/GitHub/AuthorDetector/food")
import food

def sanitize(reviews):
    clean = []
    for r in reviews:
        newR = []
        for s in r:
           newR.append(s.replace('\u2019', "'"))
        clean.append(newR)
    return clean

def getRatedParagraphs():
    print("Loading Files")
    testReviews, trainingReviews = food.createReviewArray()
    trainingReviews = sanitize(trainingReviews)
    testReviews = sanitize(testReviews)
    #Tagging every training paragraph individually
    print("Tagging Training data by paragraph")
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
    #Tagging every testing paragraph individually
    print("Tagging Testing data by paragraph")
    testParaGraphs = []
    for r in testReviews:
        if len(r) < 13:
            print("Warning!, Someone didn't have all four paragraphs")
            #for l in r:
                #print(l +"\n")
        else:
            try:
                testParaGraphs.append({"overAllRating" : "unknown", "text": r[9]})
                testParaGraphs.append({"overAllRating" : "unknown", "text": r[10]})
                testParaGraphs.append({"overAllRating" : "unknown", "text": r[11]})
                testParaGraphs.append({"overAllRating" : "unknown", "text": r[12]})
            except:
                print("Bad format")
                print(r)
    return testParaGraphs, taggedParaGraphs 