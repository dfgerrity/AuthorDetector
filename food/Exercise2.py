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
    overallTagTesting = [([e["p1"], e["p2"], e["p3"], e["p4"]]) for e in testReviews] #dummy tagging "x"    

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
    trainedClassifiers = ClassifierRunner.runNfoldCrossValidation(ClassifierRunner.naiveBayes, overallTagTraining, overallTagTesting, featureExtractors, 4)
    predictions = [c[2] for c in trainedClassifiers]
    truths = [c[3] for c in trainedClassifiers]
    Evaluator.reportAvgBinaryRMS(predictions, truths)

def partII():
    #Get the classifier from Exercise 1 to compute rating for each paragraph
    classifiers, Ex1features = Exercise1.partII()
    paragraphClassifier = classifiers[0][0]

    overallTagTraining = [([e["p1"], e["p2"], e["p3"], e["p4"]], e["overAllRating"]) for e in trainingReviews] 
    overallTagTesting = [([e["p1"], e["p2"], e["p3"], e["p4"]]) for e in testReviews] #dummy tagging "x"    

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
    trainedClassifiers = ClassifierRunner.runNfoldCrossValidation(ClassifierRunner.naiveBayes, overallTagTraining, overallTagTesting, featureExtractors, 4)
    predictions = [c[2] for c in trainedClassifiers]
    truths = [c[3] for c in trainedClassifiers]
    Evaluator.reportAvgRMS(predictions, truths)

partI()
#partII()
