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