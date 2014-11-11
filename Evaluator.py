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
        if prediction[i] == truth[i]:
            sumOfSquares += math.pow(1, 2)
    avg = sumOfSquares / len(truth)
    return math.sqrt(avg)