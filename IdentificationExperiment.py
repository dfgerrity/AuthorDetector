import LoadCorpus
import AuthorshipFeatures
import ClassifierRunner
import random
import os
import datetime
import pickle
import glob

def findMtopNgramsInCorpus(tagged_samples,m,n):    
    cumlative = {}
    for ts in tagged_samples[:100]:
        PDF = AuthorshipFeatures.topMCharacterNgramsPDF(100,n)(ts[0])
        #input(PDF)
        if len(cumlative) == 0:
            cumlative.update(PDF)
        else:
            for key in PDF.keys():
                if key in cumlative.keys():
                    cumlative[key] = (cumlative[key] + PDF[key])/2
                else:
                    cumlative[key] = PDF[key]
    return cumlative
            
def pickleTopGrams():
    tagged_samples = LoadCorpus.getTaggedSamples(1000)
    input(len(tagged_samples))
    grams = []
    print("Finding top ten bigrams")
    grams.append((findMtopNgramsInCorpus(tagged_samples,10,2),"10_2"))
    print("Finding top ten trigrams")
    grams.append((findMtopNgramsInCorpus(tagged_samples,10,3),"10_3"))
    print("Finding top ten 4grams")
    grams.append((findMtopNgramsInCorpus(tagged_samples,10,4),"10_4"))
    print("Finding top ten 5grams")
    grams.append((findMtopNgramsInCorpus(tagged_samples,10,5),"10_5"))
    print("Finding top ten 6grams")
    grams.append((findMtopNgramsInCorpus(tagged_samples,10,6),"10_6"))
    print("Saving Top Ngrams...")
    if not os.path.isdir("./savedGrams"):
        os.mkdir("./savedGrams")
    for gram in grams:
        #input(gram[0])
        fileName = datetime.datetime.now().time().isoformat().replace(":", "_")        
        pickle.dump(gram[0],open("./savedGrams/"+gram[1]+fileName+".pickle","wb"))
        print("All Features saved to ./savedGrams/"+gram[1]+fileName+".pickle")

def IDExperimentQuantizedNgrams():
    files = glob.glob("./savedGrams/*.pickle")
    for f in files:
        fs = LoadCorpus.loadFeatureSets(f)
        topM = sorted([item for item in fs.items()],key=lambda x:x[1],reverse=True)[:10]
        input(topM)

def IDExperimentQuantized():
    tagged_samples = getTaggedSamples(1000)

    featureExtractors = []
    featureExtractors.append(AuthorshipFeatures.avgWordLength)
    featureExtractors.append(AuthorshipFeatures.avgWordLengthBucketed)
    featureExtractors.append(AuthorshipFeatures.topMCharacterNgrams(1,2))
    featureExtractors.append(AuthorshipFeatures.topMCharacterNgrams(1,3))
    featureExtractors.append(AuthorshipFeatures.topMWordNgrams(1,2))
    featureExtractors.append(AuthorshipFeatures.topMWordNgrams(1,3))
    featureExtractors.append(AuthorshipFeatures.textLength)
    featureExtractors.append(AuthorshipFeatures.typeTokenRatio(4))
    featureExtractors.append(AuthorshipFeatures.typeTokenRatioBucketed(4))
    featureExtractors.append(AuthorshipFeatures.vocabSize(4))
    featureExtractors.append(AuthorshipFeatures.vocabSizeBucketed(4))
    featureExtractors.append(AuthorshipFeatures.posDist)
    featureExtractors.append(AuthorshipFeatures.percentOfLetters)
    featureExtractors.append(AuthorshipFeatures.topMPOSNgrams(1 ,2))
    featureExtractors.append(AuthorshipFeatures.topMPOSNgrams(1 ,3))

    ClassifierRunner.runNfoldCrossValidation(ClassifierRunner.naiveBayes,tagged_samples,featureExtractors,5,True)

def IDExperimentQuantizedPre():
    fs = LoadCorpus.loadFeatureSets(".\\savedsets\\66Users13_19_26.659569.pickle")
    bucketed = []
    realValues = []
    print("Separating into bucketed and real-valued feature-vectors")
    for set in fs:
        if random.random() < 0.10:
            bSet = {}
            rSet = {}
            for key in set[0].keys():
                if (key == "type/tokenb" or 
                    key == "vocabSizeb" or
                    key == "type/tokenb" or
                    key == "AVG word Length b"):
                    bSet[key] = set[0][key]
                elif (key == "type/token" or 
                    key == "vocabSize" or
                    key == "type/token" or
                    key == "AVG word Length"):
                    rSet[key] = set[0][key]
                else:
                    bSet[key] = set[0][key]
                    rSet[key] = set[0][key]
            bucketed.append((bSet,set[1]))
            realValues.append((rSet,set[1]))
    print("Using",len(bucketed), str(len(bucketed[0][0].keys()))+"-feature vectors")

    ClassifierRunner.runPreExtractedNfoldCrossValidation(ClassifierRunner.decisionTree ,bucketed,5)
    ClassifierRunner.runPreExtractedNfoldCrossValidation(ClassifierRunner.naiveBayes ,bucketed,5)
    ClassifierRunner.runPreExtractedNfoldCrossValidation(ClassifierRunner.SVM ,realValues,5)
    ClassifierRunner.runPreExtractedNfoldCrossValidation(ClassifierRunner.maxEnt,realValues,5)

#IDExperimentQuantizedPre()
#pickleTopGrams()

IDExperimentQuantizedNgrams()


