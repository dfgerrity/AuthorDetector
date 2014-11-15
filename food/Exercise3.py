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


def partI():
    print("PART I Classify by author")
    
    print("Classify by using HappySad score as features for:")
    print("Naive Bayes")
    authorTagTraining = [(e["text"], e["author"]) for e in trainingReviews] 
    authorTagTesting = [(e["text"], e["author"]) for e in testReviews]
    featureExtractors = []
    featureExtractors.append(HappySad.featureNumericScore)
    featureExtractors.append(HappySad.featureHitCount)
    featureExtractors.append(AuthorshipFeatures.typeTokenRatioBucketed)
    featureExtractors.append(AuthorshipFeatures.vocabSizeBucketed)

    #BASELINE RUN
    print("Running Baseline")
    trainedBaseline = ClassifierRunner.runNfoldCrossValidation(ClassifierRunner.mostCommonTag, authorTagTraining, featureExtractors, 4)
    ClassifierRunner.predictTagged(trainedBaseline[0][0], featureExtractors, authorTagTesting)
    predictionsBaseline = [c[2] for c in trainedBaseline]
    truthsBaseline = [c[3] for c in trainedBaseline]
    bRMS = Evaluator.reportAvgBinaryRMS(predictionsBaseline, truthsBaseline)

    #OUR CLASSIFIER RUN
    trainedClassifiers = ClassifierRunner.runNfoldCrossValidation(ClassifierRunner.naiveBayes, authorTagTraining, featureExtractors, 4)
    print("Running most accurate trained classifier on test set")
    ClassifierRunner.predictTagged(trainedClassifiers[0][0], featureExtractors, authorTagTesting)
    predictions = [c[2] for c in trainedClassifiers]
    truths = [c[3] for c in trainedClassifiers]
    cRMS = Evaluator.reportAvgBinaryRMS(predictions, truths)
    print("Accuracy improvement over baseline:", trainedClassifiers[0][1] - trainedBaseline[0][1])
    print("RMS Error reduction from baseline:", bRMS - cRMS)

partI()