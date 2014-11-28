import nltk
from nltk import FreqDist
import Extractor
import Evaluator
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier

def mostCommonTag(training_set, test_set):
    #print("Training a new Most Common Tag baseline classifier")
    tags = [tag for data, tag in training_set]
    fd = sorted([entry for entry in FreqDist(tags).items()], key=lambda x: x[1], reverse = True)
    mct = fd[0][0]
    def classify(samples):
        return [mct for s in samples]
    #print("Running new Most Common Tag baseline Bayes classifier")
    accuracy = len([1 for data,tag in test_set if tag == mct])/len(test_set)
    trueLabels = [l for d, l in test_set]
    predictedLabels = classify([d for d,t in test_set])
    #print("Accuracy:",accuracy)
    def runTrained(test_set, hasTags=False):
        #print("Running pre-trained Most Common Tag baseline classifier")
        if hasTags:
            tagglessTest_set = [data for data, tag in test_set] 
            acc = len([1 for data,tag in test_set if tag == mct])/len(test_set)
            predictions = classify(tagglessTest_set)           
            print("Accuracy:",acc)
            return ([e for e in zip(test_set, predictions)], acc) 
        else:
            tagglessTest_set = test_set        
        predictions = classify(tagglessTest_set)
        #print("Predicted Labels:",predictions)
        return [e for e in zip(test_set, predictions)]
    return (runTrained, accuracy, predictedLabels, trueLabels)


def naiveBayes(training_set, test_set, MIF=5):
    #print("Training a new Naive Bayes classifier")
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    #print("Running new Naive Bayes classifier")
    accuracy = nltk.classify.accuracy(classifier, test_set)
    trueLabels = [l for d, l in test_set]
    predictedLabels = classifier.classify_many([d for d,t in test_set])
    #print("Accuracy:",accuracy)
    classifier.show_most_informative_features(MIF)
    def runTrained(test_set, hasTags=False):
        #print("Running pre-trained Naive Bayes classifier")
        if hasTags:
            tagglessTest_set = [data for data, tag in test_set]
            acc = nltk.classify.accuracy(classifier, test_set)
            print("Accuracy:", acc)
            predictions = classifier.classify_many(tagglessTest_set)
            return ([e for e in zip(tagglessTest_set, predictions)], acc)
        else:
            tagglessTest_set = test_set         
        predictions = classifier.classify_many(tagglessTest_set)
        #print("Predicted Labels:",predictions)
        return [e for e in zip(tagglessTest_set, predictions)]
    return (runTrained, accuracy, predictedLabels, trueLabels)

#GIS is the algorithm, we could try IIS, MEGAM and TADM
def maxEnt(training_set, test_set, MIF=5):
    classifier = nltk.classify.MaxentClassifier.train(training_set,"GIS", trace=0, max_iter=1000)
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)
    #print("Running new Max Ent classifier")
    accuracy = nltk.classify.accuracy(classifier, test_set)
    trueLabels = [l for d, l in test_set]
    predictedLabels = classifier.classify_many([d for d,t in test_set])
    #print("Accuracy:",accuracy)
    classifier.show_most_informative_features(MIF)
    def runTrained(test_set, hasTags=False):
        #print("Running pre-trained Max Ent classifier")
        if hasTags:
            tagglessTest_set = [data for data, tag in test_set]
            acc = nltk.classify.accuracy(classifier, test_set)
            print("Accuracy:", acc)
            predictions = classifier.classify_many(tagglessTest_set)
            return ([e for e in zip(tagglessTest_set, predictions)], acc)
        else:
            tagglessTest_set = test_set       
        predictions = classifier.classify_many(tagglessTest_set)
        #print("Predicted Labels:",predictions)
        return [e for e in zip(tagglessTest_set, predictions)]
    return (runTrained, accuracy, predictedLabels, trueLabels)

def decisionTree(training_set, test_set, MIF=5):
    classifier = nltk.classify.DecisionTreeClassifier.train(training_set, entropy_cutoff=0, support_cutoff=0)
    print(nltk.classify.accuracy(classifier, test_set))
#     classifier.show_most_informative_features(5)
    #print("Running new Decision Tree classifier")
    accuracy = nltk.classify.accuracy(classifier, test_set)
    trueLabels = [l for d, l in test_set]
    predictedLabels = classifier.classify_many([d for d,t in test_set])
    #print("Accuracy:",accuracy)
#     classifier.show_most_informative_features(MIF)
    def runTrained(test_set, hasTags=False):
        #print("Running pre-trained Decision Tree classifier")
        if hasTags:
            tagglessTest_set = [data for data, tag in test_set]
            acc = nltk.classify.accuracy(classifier, test_set)
            print("Accuracy:", acc)
            predictions = classifier.classify_many(tagglessTest_set)
            return ([e for e in zip(tagglessTest_set, predictions)], acc)
        else:
            tagglessTest_set = test_set         
        predictions = classifier.classify_many(tagglessTest_set)
        #print("Predicted Labels:",predictions)
        return [e for e in zip(tagglessTest_set, predictions)]
    return (runTrained, accuracy, predictedLabels, trueLabels)   

def SVM(training_set, test_set):
    classifier = SklearnClassifier(LinearSVC())
    print("Training a new SVM classifier")
    classifier.train(training_set)
    print("Accuracy of SVM in training:",nltk.classify.accuracy(classifier, test_set))
#     classifier.show_most_informative_features(5)
    #print("Running new Decision Tree classifier")
    accuracy = nltk.classify.accuracy(classifier, test_set)
    trueLabels = [l for d, l in test_set]
    predictedLabels = classifier.classify_many([d for d,t in test_set])
    #print("Accuracy:",accuracy)
#     classifier.show_most_informative_features(MIF)
    def runTrained(test_set, hasTags=False):
        #print("Running pre-trained Decision Tree classifier")
        if hasTags:
            tagglessTest_set = [data for data, tag in test_set]
            acc = nltk.classify.accuracy(classifier, test_set)
            print("Accuracy:", acc)
            predictions = classifier.classify_many(tagglessTest_set)
            return ([e for e in zip(tagglessTest_set, predictions)], acc)
        else:
            tagglessTest_set = test_set         
        predictions = classifier.classify_many(tagglessTest_set)
        #print("Predicted Labels:",predictions)
        return [e for e in zip(tagglessTest_set, predictions)]
    return (runTrained, accuracy, predictedLabels, trueLabels) 


def runNfoldCrossValidation(classifier, trainingSamples, featureExtractors, n):
    classifiers = []
    folds = Extractor.getNfolds(trainingSamples, featureExtractors, n)
    for i in range(len(folds)):
         training = []
         for j in range(len(folds)):
             if not j == i:
                 training.extend(folds[j])
         classifiers.append(classifier(training, folds[i]))    
    classifiers.sort(key=lambda x: x[1], reverse = True)
    for i in range(n):
        #print("Accuracy for classifier", i+1, ":", classifiers[i][1])
        pass
    return classifiers
   
def predictTagless(classifier, featureExtractors, taglessTestSet):     
    test_set = Extractor.extractAll(taglessTestSet, featureExtractors)
    predictions = [{"data": e[0], "features" : e[1][0], "predicted_label" : e[1][1]} 
           for e in zip(taglessTestSet, classifier(test_set))] 
    #print("PREDICTIONS:") 
    #for i in range(len(predictions)):
    #    print("Prediction #", i+1, ":", predictions[i])
    #    print("") 
    return [e["predicted_label"] for e in predictions]


def predictTagged(classifier, featureExtractors, taggedTestSet):     
    test_set = Extractor.extractAllTagged(taggedTestSet, featureExtractors)
    pLabels, acc = classifier(test_set, True)
    predictions = [{"data": e[0], "features" : e[1][0], "predicted_label" : e[1][1]} 
           for e in zip(taggedTestSet, pLabels)] 
    print("PREDICTIONS:") 
    for i in range(len(predictions)):
        print("Prediction #", i+1, ":", "Predicted Label:", predictions[i]["predicted_label"], "Actual Label:", predictions[i]["data"][1])
        #print("")
    return [e["predicted_label"] for e in predictions], acc

def runSingleFold(classifier, taggedSamples, featureExtractors, trainWeight=2, testWeight=1):
    print("Compiling training and test sets")
    test_set, training_set = Extractor.getTestandTraining(taggedSamples, featureExtractors, 2,1)
    print("Running Classifier")
    return classifier(training_set, test_set)
    