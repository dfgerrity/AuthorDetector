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
import Exercise1

print("Loading Corpus...")
testReviews, trainingReviews = PreProcess.getRatedReviews()

def partI():
    print("PART I Just classify by author")
    
    print("Classify by using HappySad score as features for:")
    print("Naive Bayes")
    binaryTagTraining = [(e["text"], "+") if e["overAllRating"] in [4,5] else (e["text"], "-") for e in trainingParagraphs] 
    binaryTagTesting = [e["text"] for e in testParagraphs]
    featureExtractors = []
    featureExtractors.append(HappySad.featureNumericScore)
    featureExtractors.append(HappySad.featureHitCount)

    #BASELINE RUN
    print("Running Baseline")
    trainedBaseline = ClassifierRunner.runNfoldCrossValidation(ClassifierRunner.mostCommonTag, binaryTagTraining, binaryTagTesting, featureExtractors, 4)
    predictionsBaseline = [c[2] for c in trainedBaseline]
    truthsBaseline = [c[3] for c in trainedBaseline]
    bRMS = Evaluator.reportAvgBinaryRMS(predictionsBaseline, truthsBaseline)

    #OUR CLASSIFIER RUN
    print("Running Our Classifier")
    trainedClassifiers = ClassifierRunner.runNfoldCrossValidation(ClassifierRunner.naiveBayes, binaryTagTraining, binaryTagTesting, featureExtractors, 4)
    predictions = [c[2] for c in trainedClassifiers]
    truths = [c[3] for c in trainedClassifiers]
    cRMS = Evaluator.reportAvgBinaryRMS(predictions, truths)
    print("Accuracy improvement over baseline:", trainedClassifiers[0][1] - trainedBaseline[0][1])
    print("RMS Error reduction from baseline:", bRMS - cRMS)