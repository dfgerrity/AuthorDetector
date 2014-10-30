import nltk
import Extractor

def runNaiveBayes(taggedSamples, featureExtractors, trainWeight=2, testWeight=1):
    test_set, training = Extractor.getTestandTraining(taggedSamples, featureExtractors, 2,1)
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)
