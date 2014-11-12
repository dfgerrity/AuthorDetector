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
print(len(trainingParagraphs))

print("Loading Happy/Sad Words...")
happySadScoredWords = HappySad.loadHSWords("./words/happyAndSadWords3.txt")

print("PART I Just classify by positive or negative where \"+\" = {4,5} and \"-\" = {1,2,3}")
print("Classify by straight HappySad score mapping:")
ourtagsAdded = HappySad.happySadClassifier(happySadScoredWords, trainingParagraphs)
print("Classify by using HappySad score as features for:")
print("Naive Bayes")
binaryTagTraining = [(e["text"], "+") if e["overAllRating"] in [4,5] else (e["text"], "-") for e in trainingParagraphs] 
binaryTagTesting = [(e["text"], "x") for e in testParagraphs]
featureExtractors = []
featureExtractors.append(HappySad.featureBinaryScore)
trainedClassifiers = ClassifierRunner.runNfoldCrossValidation(ClassifierRunner.naiveBayes, binaryTagTraining, binaryTagTesting, featureExtractors, 4)
predictions = [c[2] for c in trainedClassifiers]
truths = [c[3] for c in trainedClassifiers]
Evaluator.reportAvgBinaryRMS(predictions, truths)

print("PART II Predict lassify by numeric rating 1-5 ")