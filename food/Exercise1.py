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

print("Loading Corpus...")
testParagraphs, trainingParagraphs = PreProcess.getRatedParagraphs()
print(len([r for r in trainingParagraphs if r["overAllRating"] == 1]))
print(len([r for r in trainingParagraphs if r["overAllRating"] == 2]))
print(len([r for r in trainingParagraphs if r["overAllRating"] == 3]))
print(len([r for r in trainingParagraphs if r["overAllRating"] == 4]))
print(len([r for r in trainingParagraphs if r["overAllRating"] == 5]))


print("Loading Happy/Sad Words...")
happySadScoredWords = HappySad.loadHSWords("./words/happyAndSadWords3.txt")

def partI():
    print("PART I Just classify by positive or negative where \"+\" = {4,5} and \"-\" = {1,2,3}")
    print("Classify by straight HappySad score mapping:")
    #ourtagsAdded = HappySad.happySadClassifier(happySadScoredWords, trainingParagraphs)
    print("Classify by using HappySad score as features for:")
    print("Naive Bayes")
    binaryTagTraining = [(e["text"], "+") if e["overAllRating"] in [4,5] else (e["text"], "-") for e in trainingParagraphs] 
    binaryTagTesting = [(e["text"], e["overAllRating"]) for e in testParagraphs]
    featureExtractors = []
    featureExtractors.append(HappySad.featureNumericScore)
    featureExtractors.append(HappySad.featureHitCount)

    #BASELINE RUN
    print("Running Baseline")
    trainedBaseline = ClassifierRunner.runNfoldCrossValidation(ClassifierRunner.mostCommonTag, binaryTagTraining, featureExtractors, 4)
    ClassifierRunner.predictTagged(trainedBaseline[0][0], featureExtractors, binaryTagTesting)
    predictionsBaseline = [c[2] for c in trainedBaseline]
    truthsBaseline = [c[3] for c in trainedBaseline]
    bRMS = Evaluator.reportAvgBinaryRMS(predictionsBaseline, truthsBaseline)

    #OUR CLASSIFIER RUN
    print("Running Our Classifier")
    trainedClassifiers = ClassifierRunner.runNfoldCrossValidation(ClassifierRunner.naiveBayes, binaryTagTraining, featureExtractors, 4)
    ClassifierRunner.predictTagged(trainedClassifiers[0][0], featureExtractors, binaryTagTesting)
    print("Running most accurate trained classifier on test set")

    predictions = [c[2] for c in trainedClassifiers]
    truths = [c[3] for c in trainedClassifiers]
    cRMS = Evaluator.reportAvgBinaryRMS(predictions, truths)
    print("Accuracy improvement over baseline:", trainedClassifiers[0][1] - trainedBaseline[0][1])
    print("RMS Error reduction from baseline:", bRMS - cRMS)

    return (trainedClassifiers, featureExtractors) # for use in Exercise 2

def partII():
    print("PART II Predict classify by numeric rating 1-5 ")
    print("Classify by using HappySad raw score as features for:")
    print("Naive Bayes")
    numericTagTraining = [(e["text"], e["overAllRating"]) for e in trainingParagraphs] 
    numericTagTesting = [e["text"] for e in testParagraphs]
    featureExtractors = []
    featureExtractors.append(HappySad.featureNumericScore)
    featureExtractors.append(HappySad.featureHitCount)

    #BASELINE RUN
    print("Running Baseline")
    trainedBaseline = ClassifierRunner.runNfoldCrossValidation(ClassifierRunner.mostCommonTag, numericTagTraining, featureExtractors, 4)
    ClassifierRunner.predictTagged(trainedBaseline[0][0], featureExtractors, numericTagTesting)
    predictionsBaseline = [c[2] for c in trainedBaseline]
    truthsBaseline = [c[3] for c in trainedBaseline]
    bRMS = Evaluator.reportAvgBinaryRMS(predictionsBaseline, truthsBaseline)

    #OUR CLASSIFIER RUN
    trainedClassifiers = ClassifierRunner.runNfoldCrossValidation(ClassifierRunner.naiveBayes, numericTagTraining, featureExtractors, 4)
    ClassifierRunner.predictTagged(trainedClassifiers[0][0], featureExtractors, numericTagTesting)
    predictions = [c[2] for c in trainedClassifiers]
    truths = [c[3] for c in trainedClassifiers]
    cRMS = Evaluator.reportAvgBinaryRMS(predictions, truths)
    print("Accuracy improvement over baseline:", trainedClassifiers[0][1] - trainedBaseline[0][1])
    print("RMS Error reduction from baseline:", bRMS - cRMS)

    return (trainedClassifiers, featureExtractors) # for use in Exercise 2

#partI()
partII()