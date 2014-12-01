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

IDExperimentQuantized()

