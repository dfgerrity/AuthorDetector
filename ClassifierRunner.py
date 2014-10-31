import nltk
import Extractor

def runNaiveBayes(taggedSamples, featureExtractors, trainWeight=2, testWeight=1):
    print("Compiling training and test sets")
    test_set, training_set = Extractor.getTestandTraining(taggedSamples, featureExtractors, 2,1)
    print("Running Classifier")
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)
