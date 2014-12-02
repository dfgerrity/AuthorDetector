import LoadCorpus
import AuthorshipFeatures
import ClassifierRunner

def IDExperimentQuantized():
    print("Loading Normalized Corpus...")
    corpus = LoadCorpus.normalize(LoadCorpus.loadCorpus())

    print("Collapsing corpus into one big list")
    everything = []
    for author in corpus.keys():
        everything.extend(corpus[author])

    print("Converting to (data, label) format")
    tagged_samples = [(e[0]["text"],e[1]["author"]) for e in everything]

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
    fs = LoadCorpus.loadFeatureSets(".\\savedsets\\16Users03_32_59.369006.pickle")
    bucketed = []
    realValues = []
    print("Separating into bucketed and real-valued feature-vectors")
    for set in fs:
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
    print("Using",len(bucketed[0][0].keys()),"feature vectors")

    ClassifierRunner.runPreExtractedNfoldCrossValidation(ClassifierRunner.decisionTree ,bucketed,5)
    ClassifierRunner.runPreExtractedNfoldCrossValidation(ClassifierRunner.naiveBayes ,bucketed,5)
    ClassifierRunner.runPreExtractedNfoldCrossValidation(ClassifierRunner.SVM ,realValues,5)
    ClassifierRunner.runPreExtractedNfoldCrossValidation(ClassifierRunner.maxEnt,realValues,5)

IDExperimentQuantizedPre()

