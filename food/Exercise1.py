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

testParagraphs = [] 
trainingParagraphs = []
happySadScoredWords = []

def setup():
    global testParagraphs
    global trainingParagraphs
    global happySadScoredWords
    testParagraphs = [] 
    trainingParagraphs = []
    happySadScoredWords = []
    print("Loading Corpus...")
    testParagraphs, trainingParagraphs = PreProcess.getRatedParagraphs()
    print(len([r for r in trainingParagraphs if r["overAllRating"] == 1]))
    print(len([r for r in trainingParagraphs if r["overAllRating"] == 2]))
    print(len([r for r in trainingParagraphs if r["overAllRating"] == 3]))
    print(len([r for r in trainingParagraphs if r["overAllRating"] == 4]))
    print(len([r for r in trainingParagraphs if r["overAllRating"] == 5]))
    print("Loading Happy/Sad Words...")
    happySadScoredWords = HappySad.loadHSWords("./words/happyAndSadWords3.txt")

def partI(classifier):
    global testParagraphs
    global trainingParagraphs
    global happySadScoredWords
    setup()
    print("PART I Just classify by positive or negative where \"+\" = {4,5} and \"-\" = {1,2,3}")
   # print("Classify by straight HappySad score mapping:")
    #ourtagsAdded = HappySad.happySadClassifier(happySadScoredWords, trainingParagraphs)
   # print("Classify by using HappySad score as features for:")
    
    binaryTagTraining = [(e["text"], "+") if e["overAllRating"] in [4,5] else (e["text"], "-") for e in trainingParagraphs] 
    binaryTagTesting = [(e["text"], "+") if e["overAllRating"] in [4,5] else (e["text"], "-") for e in testParagraphs]
    featureExtractors = []
    featureExtractors.append(HappySad.featureBinaryScore)
    featureExtractors.append(HappySad.featureHitCountBucketed)

    #BASELINE RUN
    print("Running Baseline")
    trainedBaseline = ClassifierRunner.runNfoldCrossValidation(ClassifierRunner.mostCommonTag, binaryTagTraining, featureExtractors, 4)
    predictionsBaseline = [c[2] for c in trainedBaseline]
    truthsBaseline = [c[3] for c in trainedBaseline]
    #bRMS = Evaluator.reportAvgBinaryRMS(predictionsBaseline, truthsBaseline)
    predictionsTesting, bAcc = ClassifierRunner.predictTagged(trainedBaseline[0][0], featureExtractors, binaryTagTesting)
    truthsTesting = [c[1] for c in binaryTagTesting]
    #input(predictionsTesting)
    #input(truthsTesting)
    bRMS = Evaluator.rmsBinaryDifference(predictionsTesting, truthsTesting)
    print("BaseLine RMS Error:", bRMS)

    #OUR CLASSIFIER RUN
    print("Running Our Classifier")
    trainedClassifiers = ClassifierRunner.runNfoldCrossValidation(classifier, binaryTagTraining, featureExtractors, 4)
    predictions = [c[2] for c in trainedClassifiers]
    truths = [c[3] for c in trainedClassifiers]
    predictionsTesting,cAcc = ClassifierRunner.predictTagged(trainedClassifiers[0][0], featureExtractors, binaryTagTesting)
    truthsTesting = [c[1] for c in binaryTagTesting]
    cRMS = Evaluator.rmsBinaryDifference(predictionsTesting, truthsTesting)
    print("Our RMS Error:", cRMS)
    print("Accuracy improvement over baseline:", cAcc - bAcc)
    print("RMS Error reduction from baseline:", bRMS - cRMS)
    print("Running most accurate trained classifier on test set")

    return (trainedClassifiers, featureExtractors) # for use in Exercise 2

def partII(classifier):
    setup()
    print("PART II Predict classify by numeric rating 1-5 ")    
    numericTagTraining = [(e["text"], e["overAllRating"]) for e in trainingParagraphs] 
    numericTagTesting = [(e["text"], e["overAllRating"]) for e in testParagraphs]
    featureExtractors = []
    featureExtractors.append(HappySad.featureNumericScore)
    featureExtractors.append(HappySad.featureHitCountBucketed)

    #BASELINE RUN
    print("Running Baseline")
    trainedBaseline = ClassifierRunner.runNfoldCrossValidation(ClassifierRunner.mostCommonTag, numericTagTraining, featureExtractors, 4)
    predictionsBaseline = [c[2] for c in trainedBaseline]
    truthsBaseline = [c[3] for c in trainedBaseline]
    #bRMS = Evaluator.reportAvgBinaryRMS(predictionsBaseline, truthsBaseline)
    predictionsTesting,bAcc = ClassifierRunner.predictTagged(trainedBaseline[0][0], featureExtractors, numericTagTesting)
    truthsTesting = [c[1] for c in numericTagTesting]
    bRMS = Evaluator.rmsBinaryDifference(predictionsTesting, truthsTesting)
    print("BaseLine RMS Error:", bRMS)

    #OUR CLASSIFIER RUN
    trainedClassifiers = ClassifierRunner.runNfoldCrossValidation(classifier, numericTagTraining, featureExtractors, 4)
        
    predictions = [c[2] for c in trainedClassifiers]
    truths = [c[3] for c in trainedClassifiers]
    print("Running most accurate trained classifier on test set")
    predictionsTesting, cAcc = ClassifierRunner.predictTagged(trainedClassifiers[0][0], featureExtractors, numericTagTesting)
    truthsTesting = [c[1] for c in numericTagTesting]
    cRMS = Evaluator.rmsBinaryDifference(predictionsTesting, truthsTesting)
    print("Our RMS Error:", cRMS)
    print("Accuracy improvement over baseline:", cAcc - bAcc)
    print("RMS Error reduction from baseline:", bRMS - cRMS)
    

    return (trainedClassifiers, featureExtractors) # for use in Exercise 2

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

#partII(ClassifierRunner.naiveBayes)
runPartII()
