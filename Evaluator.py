import math

def rmsDifference(prediction, truth):
    '''prediction and truth vectors must be of same length'''
    sumOfSquares = 0
    for i in range(len(truth)):
        sumOfSquares += math.pow(prediction[i] - truth[i], 2)
    avg = sumOfSquares / len(truth)
    return math.sqrt(avg)

def rmsBinaryDifference(prediction, truth):
    '''prediction and truth vectors must be of same length'''
    sumOfSquares = 0
    for i in range(len(truth)):
        if prediction[i] != truth[i]:
            sumOfSquares += math.pow(1, 2)
    avg = sumOfSquares / len(truth)
    return math.sqrt(avg)

def reportAvgBinaryRMS(predictions, truths):
    '''predictions and truths vectors must be of same length'''
    totalrms = 0
    for i in range(len(predictions)):
        rmsError = rmsBinaryDifference(predictions[i], truths[i])
        totalrms += rmsError
        print("RMS Error:", rmsError)
    print("Average RMS Error", totalrms / len(predictions))
    return totalrms / len(predictions)

def reportAvgRMS(predictions, truths):
    '''predictions and truths vectors must be of same length'''
    totalrms = 0
    for i in range(len(predictions)):
        rmsError = rmsDifference(predictions[i], truths[i])
        totalrms += rmsError
        print("RMS Error:", rmsError)
    print("Average RMS Error", totalrms / len(predictions))
    return totalrms / len(predictions)

def createConfusionMatrix(labels, predictions, truths):
    matrix = []
    fieldSize=6
    format = "%"+str(fieldSize)+"s"
    labels = set(labels)
    labels.update(set(predictions))
    labels.update(set(truths))
    labels = list(labels)

    for label in labels:
        matrix.append([0]*len(labels))
    for i in range(len(truths)):
        matrix[labels.index(truths[i])][labels.index(predictions[i])]+=1

    print(format % "", end="")
    for i in range(len(labels)):
        print(format % labels[i],end="")
    print("")

    for i in range(len(labels)):
        print(format % labels[i],end="")
        for j in range(len(labels)):
            print(format % matrix[i][j],end="")
        print("")
    return matrix

labels = ["a", "b", "c", "d", "f", "g"]
pred = ["c", "b", "c" , "f", "d", "c", "a"]
truths = ["c", "b", "c" , "f", "d", "c", "a"]

#createConfusionMatrix(labels, pred, truths)