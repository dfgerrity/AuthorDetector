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
print(len([r for r in trainingReviews if r["overAllRating"] == 1]))
print(len([r for r in trainingReviews if r["overAllRating"] == 2]))
print(len([r for r in trainingReviews if r["overAllRating"] == 3]))
print(len([r for r in trainingReviews if r["overAllRating"] == 4]))
print(len([r for r in trainingReviews if r["overAllRating"] == 5]))

def partI():
    #Get the classifier from Exercise 1 to compute rating for each paragraph
    classifiers, Ex1features = Exercise1.partI()
    paragraphClassifier = classifiers[0][0]

    overallTagTraining = [([e["p1"], e["p2"], e["p3"], e["p4"]], "+" if e["overAllRating"] in [4,5] else "-") for e in trainingReviews] 
    overallTagTesting = [([e["p1"], e["p2"], e["p3"], e["p4"]], "+" if e["overAllRating"] in [4,5] else "-") for e in testReviews]    

    #Use ratings of each paragraph from Ex
    def featureParagraphNumericRatings(sample):
        #input(sample)
        Ex1FeatureSet = Extractor.extractAll(sample, Ex1features)
        #input(Ex1FeatureSet)
        rating = [number for data, number in paragraphClassifier(Ex1FeatureSet)]
        #input(rating)
        return {"Food": rating[0], "Service": rating[1], "Venue" : rating[2], "OverallP" : rating[3]}
    featureExtractors = []
    featureExtractors.append(featureParagraphNumericRatings)

    #BASELINE RUN
    print("Running Baseline")
    trainedBaseline = ClassifierRunner.runNfoldCrossValidation(ClassifierRunner.mostCommonTag, overallTagTraining, featureExtractors, 4)
    print("Running most accurate trained classifier on test set")
    ClassifierRunner.predictTagged(trainedBaseline[0][0], featureExtractors, overallTagTesting)
    predictionsBaseline = [c[2] for c in trainedBaseline]
    truthsBaseline = [c[3] for c in trainedBaseline]
    bRMS = Evaluator.reportAvgBinaryRMS(predictionsBaseline, truthsBaseline)

    #OUR CLASSIFIER RUN
    trainedClassifiers = ClassifierRunner.runNfoldCrossValidation(ClassifierRunner.naiveBayes, overallTagTraining, featureExtractors, 4)
    print("Running most accurate trained classifier on test set")
    ClassifierRunner.predictTagged(trainedClassifiers[0][0], featureExtractors, overallTagTesting)
    predictions = [c[2] for c in trainedClassifiers]
    truths = [c[3] for c in trainedClassifiers]
    cRMS = Evaluator.reportAvgBinaryRMS(predictions, truths)
    print("Accuracy improvement over baseline:", trainedClassifiers[0][1] - trainedBaseline[0][1])
    print("RMS Error reduction from baseline:", bRMS - cRMS)

def partII():
    #Get the classifier from Exercise 1 to compute rating for each paragraph
    classifiers, Ex1features = Exercise1.partII()
    paragraphClassifier = classifiers[0][0]

    overallTagTraining = [([e["p1"], e["p2"], e["p3"], e["p4"]], e["overAllRating"]) for e in trainingReviews] 
    overallTagTesting = [([e["p1"], e["p2"], e["p3"], e["p4"]], e["overAllRating"]) for e in testReviews]     

    #Use ratings of each paragraph from Ex
    def featureParagraphNumericRatings(sample):
        #input(sample)
        Ex1FeatureSet = Extractor.extractAll(sample, Ex1features)
        #input(Ex1FeatureSet)
        rating = [number for data, number in paragraphClassifier(Ex1FeatureSet)]
        #input(rating)
        return {"Food": rating[0], "Service": rating[1], "Venue" : rating[2], "OverallP" : rating[3]}
    featureExtractors = []
    featureExtractors.append(featureParagraphNumericRatings)

    #BASELINE RUN
    print("Running Baseline")
    trainedBaseline = ClassifierRunner.runNfoldCrossValidation(ClassifierRunner.mostCommonTag, overallTagTraining, featureExtractors, 4)
    ClassifierRunner.predictTagged(trainedBaseline[0][0], featureExtractors, overallTagTesting)
    predictionsBaseline = [c[2] for c in trainedBaseline]
    truthsBaseline = [c[3] for c in trainedBaseline]
    bRMS = Evaluator.reportAvgBinaryRMS(predictionsBaseline, truthsBaseline)

    #OUR CLASSIFIER RUN
    trainedClassifiers = ClassifierRunner.runNfoldCrossValidation(ClassifierRunner.naiveBayes, overallTagTraining, featureExtractors, 4)
    ClassifierRunner.predictTagged(trainedClassifiers[0][0], featureExtractors, overallTagTesting)
    predictions = [c[2] for c in trainedClassifiers]
    truths = [c[3] for c in trainedClassifiers]
    cRMS = Evaluator.reportAvgBinaryRMS(predictions, truths)
    print("Accuracy improvement over baseline:", trainedClassifiers[0][1] - trainedBaseline[0][1])
    print("RMS Error reduction from baseline:", bRMS - cRMS)


partI()
#partII()
