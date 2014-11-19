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

testReviews = [] 
trainingReviews = []

Iclassifiers =[]
IEx1features = []
IIclassifiers =[]
IIEx1features = []

def setup():
    global testReviews
    global trainingReviews
    global Iclassifiers
    global IEx1features
    global IIclassifiers
    global IIEx1features
    print("Loading Corpus...")
    testReviews, trainingReviews = PreProcess.getRatedReviews()
    print(len([r for r in trainingReviews if r["overAllRating"] == 1]))
    print(len([r for r in trainingReviews if r["overAllRating"] == 2]))
    print(len([r for r in trainingReviews if r["overAllRating"] == 3]))
    print(len([r for r in trainingReviews if r["overAllRating"] == 4]))
    print(len([r for r in trainingReviews if r["overAllRating"] == 5]))

    print("IGNORE////////////////")
    #Get the classifier from Exercise 1 to compute rating for each paragraph
    Iclassifiers, IEx1features = Exercise1.partI(ClassifierRunner.naiveBayes)
    #Get the classifier from Exercise 1 to compute rating for each paragraph
    IIclassifiers, IIEx1features = Exercise1.partII(ClassifierRunner.maxEnt)
    print("END IGNORE////////////////")

def partI(classifier):
    setup()
    paragraphClassifier = Iclassifiers[0][0]

    overallTagTraining = [([e["p1"], e["p2"], e["p3"], e["p4"]], "+" if e["overAllRating"] in [4,5] else "-") for e in trainingReviews] 
    overallTagTesting = [([e["p1"], e["p2"], e["p3"], e["p4"]], "+" if e["overAllRating"] in [4,5] else "-") for e in testReviews]    

    #Use ratings of each paragraph from Ex
    def featureParagraphNumericRatings(sample):
        #input(sample)
        Ex1FeatureSet = Extractor.extractAll(sample, IEx1features)
        #input(Ex1FeatureSet)
        rating = [number for data, number in paragraphClassifier(Ex1FeatureSet)]
        #input(rating)
        return {"Food": rating[0], "Service": rating[1], "Venue" : rating[2], "OverallP" : rating[3]}
    featureExtractors = []
    featureExtractors.append(featureParagraphNumericRatings)

    #BASELINE RUN
    print("Running Baseline")
    trainedBaseline = ClassifierRunner.runNfoldCrossValidation(ClassifierRunner.mostCommonTag, overallTagTraining, featureExtractors, 4)
    predictionsBaseline = [c[2] for c in trainedBaseline]
    truthsBaseline = [c[3] for c in trainedBaseline]
    predictionsTesting, bAcc = ClassifierRunner.predictTagged(trainedBaseline[0][0], featureExtractors, overallTagTesting)
    truthsTesting = [c[1] for c in overallTagTesting]
    bRMS = Evaluator.rmsBinaryDifference(predictionsTesting, truthsTesting)
    print("BaseLine RMS Error:", bRMS)
    #bRMS = Evaluator.reportAvgBinaryRMS(predictionsBaseline, truthsBaseline)

    #OUR CLASSIFIER RUN
    trainedClassifiers = ClassifierRunner.runNfoldCrossValidation(classifier, overallTagTraining, featureExtractors, 4)
    predictions = [c[2] for c in trainedClassifiers]
    truths = [c[3] for c in trainedClassifiers]
    print("Running most accurate trained classifier on test set")
    predictionsTesting, cAcc = ClassifierRunner.predictTagged(trainedClassifiers[0][0], featureExtractors, overallTagTesting)
    truthsTesting = [c[1] for c in overallTagTesting]
    #input(predictionsTesting)
    #input(truthsTesting)
    cRMS = Evaluator.rmsBinaryDifference(predictionsTesting, truthsTesting)
    print("Our RMS Error:", cRMS)
    print("Accuracy improvement over baseline:", cAcc - bAcc)
    print("RMS Error reduction from baseline:", bRMS - cRMS)

def partII(classifier):
    setup()
    paragraphClassifier = IIclassifiers[0][0]

    overallTagTraining = [([e["p1"], e["p2"], e["p3"], e["p4"]], e["overAllRating"]) for e in trainingReviews] 
    overallTagTesting = [([e["p1"], e["p2"], e["p3"], e["p4"]], e["overAllRating"]) for e in testReviews]     

    #Use ratings of each paragraph from Ex
    def featureParagraphNumericRatings(sample):
        #input(sample)
        Ex1FeatureSet = Extractor.extractAll(sample, IIEx1features)
        #input(Ex1FeatureSet)
        rating = [number for data, number in paragraphClassifier(Ex1FeatureSet)]
        #input(rating)
        return {"Food": rating[0], "Service": rating[1], "Venue" : rating[2], "OverallP" : rating[3]}
    featureExtractors = []
    featureExtractors.append(featureParagraphNumericRatings)

    #BASELINE RUN
    print("Running Baseline")
    trainedBaseline = ClassifierRunner.runNfoldCrossValidation(ClassifierRunner.mostCommonTag, overallTagTraining, featureExtractors, 4)
    predictionsBaseline = [c[2] for c in trainedBaseline]
    truthsBaseline = [c[3] for c in trainedBaseline]
    predictionsTesting, bAcc = ClassifierRunner.predictTagged(trainedBaseline[0][0], featureExtractors, overallTagTesting)
    truthsTesting = [c[1] for c in overallTagTesting]
    bRMS = Evaluator.rmsDifference(predictionsTesting, truthsTesting)
    print("BaseLine RMS Error:", bRMS)
    #bRMS = Evaluator.reportAvgBinaryRMS(predictionsBaseline, truthsBaseline)

    #OUR CLASSIFIER RUN
    trainedClassifiers = ClassifierRunner.runNfoldCrossValidation(classifier, overallTagTraining, featureExtractors, 4)
    ClassifierRunner.predictTagged(trainedClassifiers[0][0], featureExtractors, overallTagTesting)
    predictions = [c[2] for c in trainedClassifiers]
    truths = [c[3] for c in trainedClassifiers]
    print("Running most accurate trained classifier on test set")
    predictionsTesting, cAcc = ClassifierRunner.predictTagged(trainedClassifiers[0][0], featureExtractors, overallTagTesting)
    truthsTesting = [c[1] for c in overallTagTesting]
    cRMS = Evaluator.rmsDifference(predictionsTesting, truthsTesting)
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

def runPartII():
    for i in range(5):
        print("Naive Bayes run #", i)
        partII(ClassifierRunner.naiveBayes)
    for i in range(5):
        print("Decision Tree run #", i)
        partII(ClassifierRunner.decisionTree)
    for i in range(5):
        print("Max Entropy run #", i)
        partII(ClassifierRunner.maxEnt)

runPartI()
