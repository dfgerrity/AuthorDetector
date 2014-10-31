import random


def extractAll(taggedSamples, featureExtractors):
    '''Expects taggedSamples to be [data, tag] or (data,tag)
       Expects featureExtractors to be list of feature functions
       Automatically sends only the Data part of sample to feature extractors.
       Returns list of feature dictionaries, one dictionary for each sample composed of all 
       features extracted from every feature extractor running on that sample.'''
    featureSets = []
    uid = 2
    print("Running",len(featureExtractors),"Extractors on",  len(taggedSamples), "samples")
    for ts in taggedSamples:
        newFeatureVector = {}
        for f in featureExtractors:            
            features = f(ts[0])
            if not features.keys().isdisjoint(newFeatureVector.keys()):
                print("Warning! Two features extractors are claiming the same feature name.")
                print("Applying automatic disambiguation.") 
                features = {n+str(uid): v for n,v in features.items()}
                uid+=1
            newFeatureVector.update(features)
        featureSets.append(newFeatureVector)                 
        

def getTestandTraining(taggedSamples, featureExtractors, trainWeight=2, testWeight=1):
    '''Expects taggedSamples to be [data, tag] or (data,tag) Returns test,training'''
    featureSets = extractAll(taggedSamples, featureExtractors)
    print("Shuffling featuresets")
    random.shuffle(featureSets)
    unit = int(len(featureSets) / (trainWeight + testWeight))
    cutoff = testWeight*unit
    print("Divided in to ". trainWeight, "training samples for every", testWeight, "test sample(s)")
    test, training = featureSets[:cutoff], featureSets[cutoff:]
    return test, training
     