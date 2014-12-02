import random
import os
import datetime
import pickle

def extractAllTagged(taggedSamples, featureExtractors, save=False):
    '''Expects each taggedSample to be [data, tag] or (data,tag)
       Expects featureExtractors to be list of feature functions
       Automatically sends only the Data part of sample to feature extractors.
       Returns list of tagged feature sets. A tagged feature set is a tuple of 1) a dictionary
       for each sample composed of all features extracted from every feature
       extractor running on that sample and 2) the original tag paired with the sample.'''
    featureSets = []
    uid = 2
    i=0
    print("Running",len(featureExtractors),"Extractor(s) on",  len(taggedSamples), "samples")
    for ts in taggedSamples:
        i+=1
        print("Extracting features from Sample #", i)
        newFeatureVector = {}
        f=0
        for f in featureExtractors:    
            #print("Running extractor #",f)        
            features = f(ts[0]) # Expects data to be first element
            if not features.keys().isdisjoint(newFeatureVector.keys()):
                print("Warning! Two features extractors are claiming the same feature name.")
                print("Applying automatic disambiguation. Second extractor:", f) 
                features = {n+str(uid): v for n,v in features.items()}
                uid+=1
            newFeatureVector.update(features)
        #input(newFeatureVector)
        featureSets.append((newFeatureVector,ts[1]))
    if save:
        print("Saving Feature Sets...")
        if not os.path.isdir("./savedsets"):
            os.mkdir("./savedsets")
        fileName = datetime.datetime.now().time().isoformat().replace(":", "_")        
        pickle.dump(featureSets,open("./savedsets/"+fileName+".pickle","wb"))
        print("All Features saved to ./savedsets/"+fileName+".pickle")
    return featureSets 

def extractAll(samples, featureExtractors, save=False):
    '''Expects each sample to be the data directly accepted by feature extractors
       Expects featureExtractors to be list of feature functions
       Automatically sends only the Data part of sample to feature extractors.
       Returns list of feature sets. Returns a dictionary
       for each sample composed of all features extracted from every feature
       extractor running on that sample.'''
    featureSets = []
    uid = 2
    i=0
    #print("Running",len(featureExtractors),"Extractor(s) on",  len(samples), "samples")
    for s in samples:
        i+=1
        #print("Extracting features from Sample #", i)
        newFeatureVector = {}
        f=0
        for f in featureExtractors:    
            #print("Running extractor #",f)        
            features = f(s) # Expects data to be first element
            if not features.keys().isdisjoint(newFeatureVector.keys()):
                print("Warning! Two features extractors are claiming the same feature name.")
                print("Applying automatic disambiguation.") 
                features = {n+str(uid): v for n,v in features.items()}
                uid+=1
            newFeatureVector.update(features)
        featureSets.append(newFeatureVector)
    if save:
        print("Saving Feature Sets...")
        if not os.path.isdir("./savedsets"):
            os.mkdir("./savedsets")
        pickle.dump(featrueSets,open("./savedsets/"+datetime.datetime.now().time().isoformat()+".pickle"))
        print("All Features saved to ./savedsets/"+datetime.datetime.now().time().isoformat()+".pickle")
    return featureSets                
        

def getTestandTraining(taggedSamples, featureExtractors, trainWeight=2, testWeight=1,save=False):
    '''Expects taggedSamples to be [data, tag] or (data,tag) Returns test,training'''
    featureSets = extractAllTagged(taggedSamples, featureExtractors)
    print("Shuffling featuresets")
    random.shuffle(featureSets)
    unit = int(len(featureSets) / (trainWeight + testWeight))
    cutoff = testWeight*unit
    print("Divided in to ", trainWeight, "training samples for every", testWeight, "test sample(s)")
    test, training = featureSets[:cutoff], featureSets[cutoff:]
    return test, training

def getNfolds(taggedSamples, featureExtractors, n=5, save=False):
    '''Expects taggedSamples to be [data, tag] or (data,tag) Returns folds[] '''
    if n < 2:
        #print("Cannot do cross-fold validation on 1 fold. Increasing to 2")
        n = 2
    featureSets = extractAllTagged(taggedSamples, featureExtractors, save)
    #print("Shuffling featuresets")
    random.shuffle(featureSets)
    unit = int(len(featureSets) / n)
    #print("Divided into", n, "folds")
    folds = []
    for i in range(n-1):
        folds.append(featureSets[unit*i:unit*(i+1)])
    folds.append(featureSets[unit*(n-1):])
    return folds
 
def unitTests():   
    folds = getNfolds([(1,1), (2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10)], [lambda x:{"f":x}], 1)
    for fold in folds:
        print(fold)
     