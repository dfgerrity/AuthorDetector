import sys 
sys.path.append("..")
#import os
#os.chdir("C:/Users/Dan'l Boone/Documents/GitHub/AuthorDetector/food")
import food
import PreProcess
import HappySad
import Extractor
import Evaluator
import ClassifierRunner
import AuthorshipFeatures

print("Loading Corpus...")
testReviews, trainingReviews = PreProcess.getByAuthor()


def partI(classifier):
    print("PART I Classify by author")
    
    print("Classify by using HappySad score as features for:")
    print("Naive Bayes")
    authorTagTraining = [(e["text"], e["author"]) for e in trainingReviews] 
    authorTagTesting = [(e["text"], e["author"]) for e in testReviews]
    featureExtractors = []
    if classifier == ClassifierRunner.naiveBayes:
        featureExtractors.append(HappySad.featureNumericScore)
        featureExtractors.append(HappySad.featureHitCountBucketed)
        featureExtractors.append(AuthorshipFeatures.typeTokenRatioBucketed)
        featureExtractors.append(AuthorshipFeatures.vocabSizeBucketed)
    else:
        featureExtractors.append(HappySad.featureNumericScore)
        featureExtractors.append(HappySad.featureHitCount)
        featureExtractors.append(AuthorshipFeatures.typeTokenRatio)
        featureExtractors.append(AuthorshipFeatures.vocabSize)

    #BASELINE RUN
    print("Running Baseline")
    trainedBaseline = ClassifierRunner.runNfoldCrossValidation(ClassifierRunner.mostCommonTag, authorTagTraining, featureExtractors, 4)
    predictionsBaseline = [c[2] for c in trainedBaseline]
    truthsBaseline = [c[3] for c in trainedBaseline]
    predictionsTesting,bAcc = ClassifierRunner.predictTagged(trainedBaseline[0][0], featureExtractors, authorTagTesting)
    truthsTesting = [c[1] for c in authorTagTesting]
    bRMS = Evaluator.rmsBinaryDifference(predictionsTesting, truthsTesting)
    print("BaseLine RMS Error:", bRMS)

    #OUR CLASSIFIER RUN
    trainedClassifiers = ClassifierRunner.runNfoldCrossValidation(classifier, authorTagTraining, featureExtractors, 4)
    predictions = [c[2] for c in trainedClassifiers]
    truths = [c[3] for c in trainedClassifiers]
    print("Running most accurate trained classifier on test set")
    predictionsTesting, cAcc = ClassifierRunner.predictTagged(trainedClassifiers[0][0], featureExtractors, authorTagTesting)
    truthsTesting = [c[1] for c in authorTagTesting]
    cRMS = Evaluator.rmsBinaryDifference(predictionsTesting, truthsTesting)
    Evaluator.createConfusionMatrix([t for d,t in authorTagTraining], predictionsTesting, truthsTesting)
    print("Our RMS Error:", cRMS)
    print("Accuracy improvement over baseline:", cAcc - bAcc)
    print("RMS Error reduction from baseline:", bRMS - cRMS)

def runPartI():
    for i in range(5):
        print("Naive Bayes run #", i)
        partI(ClassifierRunner.naiveBayes)
    for i in range(5):
        print("Decision Tree run #", i)
        partI(ClassifierRunner.decisionTree)
    for i in range(5):
        print("Max Entropy run #", i)
        partI(ClassifierRunner.maxEnt)

runPartI()