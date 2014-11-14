import nltk
import Extractor
import Evaluator

def naiveBayes(training_set, test_set, MIF=5):
    print("Training a new Naive Bayes classifier")
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("Running new Naive Bayes classifier")
    accuracy = nltk.classify.accuracy(classifier, test_set)
    trueLabels = [l for d, l in test_set]
    predictedLabels = classifier.classify_many([d for d,t in test_set])
    print("Accuracy:",accuracy)
    classifier.show_most_informative_features(MIF)
    def runTrained(tagglessTest_set):        
        print("Running pre-trained Naive Bayes classifier")
        predictions = classifier.classify_many(tagglessTest_set)
        print(predictions)
        return [e for e in zip(tagglessTest_set, predictions)]
    return (runTrained, accuracy, predictedLabels, trueLabels)

#GIS is the algorithm, we could try IIS, MEGAM and TADM
def maxEnt(training_set, test_set, MIF=5):
    classifier = nltk.classify.MaxentClassifier.train(training_set,"GIS", trace=0, max_iter=1000)
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)
    print("Running new Max Ent classifier")
    accuracy = nltk.classify.accuracy(classifier, test_set)
    trueLabels = [l for d, l in test_set]
    predictedLabels = classifier.classify_many([d for d,t in test_set])
    print("Accuracy:",accuracy)
    classifier.show_most_informative_features(MIF)
    def runTrained(tagglessTest_set):        
        print("Running pre-trained Max Ent classifier")
        predictions = classifier.classify_many(tagglessTest_set)
        print(predictions)
        return [e for e in zip(tagglessTest_set, predictions)]
    return (runTrained, accuracy, predictedLabels, trueLabels)

def decisionTree(training_set, test_set, MIF=5):
    classifier = nltk.classify.DecisionTreeClassifier.train(training_set, entropy_cutoff=0, support_cutoff=0)
    print(nltk.classify.accuracy(classifier, test_set))
#     classifier.show_most_informative_features(5)
    print("Running new Decision Tree classifier")
    accuracy = nltk.classify.accuracy(classifier, test_set)
    trueLabels = [l for d, l in test_set]
    predictedLabels = classifier.classify_many([d for d,t in test_set])
    print("Accuracy:",accuracy)
#     classifier.show_most_informative_features(MIF)
    def runTrained(tagglessTest_set):        
        print("Running pre-trained Decision Tree classifier")
        predictions = classifier.classify_many(tagglessTest_set)
        print(predictions)
        return [e for e in zip(tagglessTest_set, predictions)]
    return (runTrained, accuracy, predictedLabels, trueLabels)   


def runNfoldCrossValidation(classifier, trainingSamples, testingSamples, featureExtractors, n):
    classifiers = []
    folds = Extractor.getNfolds(trainingSamples, featureExtractors, n)
    for i in range(len(folds)):
         training = []
         for j in range(len(folds)):
             if not j == i:
                 training.extend(folds[j])
         classifiers.append(classifier(training, folds[i]))
    test_set = Extractor.extractAll(testingSamples, featureExtractors)
    classifiers.sort(key=lambda x: x[1], reverse = True)
    for i in range(n):
        print("Accuracy for classifier", i+1, ":", classifiers[i][1]) 
    print("Running most accurate trained classifier on test set") 
    predictions = [{"data": e[0], "features" : e[1][0], "predicted_label" : e[1][1]} 
           for e in zip(testingSamples, classifiers[0][0](test_set))] 
    print("PREDICTIONS:") 
    for i in range(len(predictions)):
        print("Prediction #", i+1, ":", predictions[i])
        print("")
    return classifiers
    

def runSingleFold(classifier, taggedSamples, featureExtractors, trainWeight=2, testWeight=1):
    print("Compiling training and test sets")
    test_set, training_set = Extractor.getTestandTraining(taggedSamples, featureExtractors, 2,1)
    print("Running NaiveBayes Classifier")
    return classifier(training_set, test_set)
    