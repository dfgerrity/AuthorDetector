import os 
os.chdir("C:/Users/Dan'l Boone/Documents/GitHub/AuthorDetector/food")
import food
import random

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
    #input(len(testReviews))
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
                testParaGraphs.append({"overAllRating" : int(r[4].split(":")[1].strip()), "text": r[9]})
                testParaGraphs.append({"overAllRating" : int(r[5].split(":")[1].strip()), "text": r[10]})
                testParaGraphs.append({"overAllRating" : int(r[6].split(":")[1].strip()), "text": r[11]})
                testParaGraphs.append({"overAllRating" : int(r[7].split(":")[1].strip()), "text": r[12]})
            except:
                print("Bad format")
                print(r)
    #input(len(testParaGraphs))
    return testParaGraphs, taggedParaGraphs 

def getRatedReviews():
    print("Loading Files")
    testReviews, trainingReviews = food.createReviewArray()
    trainingReviews = sanitize(trainingReviews)
    testReviews = sanitize(testReviews)
    #Tagging every training review individually as four paragaphs and 1 overall rating label
    print("Tagging Training data by paragraph")
    trainingRatedReviews = []
    for r in trainingReviews:
        if len(r) < 13:
            print("Warning!, Someone didn't have all four paragraphs")
            #for l in r:
                #print(l +"\n")
        else:
            try:
                trainingRatedReviews.append({"overAllRating" : int(r[7].split(":")[1].strip()), "p1": r[9], "p2": r[10],"p3": r[11],"p4": r[12]})
            except:
                print("Bad format")
    #Tagging every testing review individually
    print("Tagging Testing data by paragraph")
    testRatedReviews = []
    for r in testReviews:
        if len(r) < 13:
            print("Warning!, Someone didn't have all four paragraphs")
            #for l in r:
                #print(l +"\n")
        else:
            try:
                testRatedReviews.append({"overAllRating" : int(r[7].split(":")[1].strip()), "p1": r[9], "p2": r[10],"p3": r[11],"p4": r[12]})
            except:
                print("Bad format")
                print(r)
    return testRatedReviews, trainingRatedReviews


def getByAuthor():
    '''Splits reviews tagged by Author into 25% testReviews and 75% trainingReviews'''
    authorMap = food.createReviewerToReviewMap()
    testSet = []
    trainingSet = []
    for name in authorMap.keys():
        if random.random() < .5:
            pick = random.randint(0,len(authorMap[name])-1)
            testSet.append({"author": name[9:11]+name[-3:], "text" : 
                            authorMap[name][pick][9] +
                            authorMap[name][pick][10] + 
                            authorMap[name][pick][11] + 
                            authorMap[name][pick][12]})
            trainingSet.extend([{"author": name[9:11]+name[-3:], "text" : 
                                 authorMap[name][i][9] + 
                                 authorMap[name][i][10] + 
                                 authorMap[name][i][11] + 
                                 authorMap[name][i][12]} for i in range(len(authorMap[name])) if i != pick]) 
        else:
            trainingSet.extend([{"author": name[9:11]+name[-3:], "text" : 
                                 authorMap[name][i][9] + 
                                 authorMap[name][i][10] + 
                                 authorMap[name][i][11] + 
                                 authorMap[name][i][12]} for i in range(len(authorMap[name]))])
    return testSet, trainingSet


