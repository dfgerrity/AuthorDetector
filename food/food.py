'''
Created on Oct 21, 2014

@author: computer
'''

import glob
import fileinput
import re
import nltk   
import math
from random import shuffle
from urllib.request import urlopen
from sys import argv

testReviews = []
trainingReviews = []
reviewerToTestReviewsMap = {}
reviewerToTrainingReviewsMap = {}
test = []
training = []

def createOutPutFiles():
    files = glob.glob("./test/*.html")
#     print(files)
    for file in files:
#         print(file)
        txt = open(file, encoding='utf8')
        s= remove_tags(txt.read())
        t = open("./testOut/" + file.replace("./test", ""), "w+")
#         print(s)
        t.write(s)

def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]

def createReviewerToReviewMap():
    createReviewArray()
    global reviewerToTrainingReviewsMap
    reviewerToTrainingReviewsMap = {}
    global reviewerToTestReviewsMap
    reviewerToTestReviewsMap = {}
   
    for tuple in test:
        s = re.sub('[^a-zA-Z:]', '', tuple[0])
        if s in reviewerToTrainingReviewsMap:
            reviewerToTrainingReviewsMap[s].append(tuple)
        else:
            reviewerToTrainingReviewsMap[s] = [tuple]
   
    for tuple in training:
        s = re.sub('[^a-zA-Z:]', '', tuple[0])
        if s in reviewerToTrainingReviewsMap:
            reviewerToTrainingReviewsMap[s].append(tuple)
        else:
            reviewerToTrainingReviewsMap[s] = [tuple]        
   
    return reviewerToTrainingReviewsMap

# Written review is given its own index in the array. The next 4 indices
# are the paragraphs of the review. If we need to join them later we can
def createReviewArray():
    testReviews = []
    trainingReviews = []
    reviewerToTestReviewsMap = {}
    reviewerToTrainingReviewsMap = {}
    test = []
    training = []

    files = glob.glob("./trainingOut/*.html")

    for file in files:

        txt = open(file, encoding='utf-8')
        lines = txt.readlines()
        
        if '\n' in lines or '\xa0' in lines or '\xa0\n' in lines or '\u2019' in lines:
            lines = remove_values_from_list(lines, '\n')
            lines = remove_values_from_list(lines, '\xa0')
            lines = remove_values_from_list(lines, '\xa0\n')            
            lines = remove_values_from_list(lines, '\u2019')
        if len(lines) != 13:
            print(len(lines), " ", lines)
        
        lines = [s.replace('\xa0', " ") for s in lines]
        
        trainingReviews.append(lines)
    
#     print("TRAINING")
#     for r in trainingReviews:
#         print(r)
#         for attribute in r:
#             print(attribute)
#         print("\n===================")
      
      
#     files = glob.glob("./testOut/*.html")
#     
#     for testFile in files:
#         print(testFile)
#         txt = open(testFile, encoding='utf-8')
#         lines = txt.readlines()
#         
#         if '\n' in lines or '\xa0' in lines or '\xa0\n' in lines or '\u2019' in lines:
#             lines = remove_values_from_list(lines, '\n')
#             lines = remove_values_from_list(lines, '\xa0')
#             lines = remove_values_from_list(lines, '\xa0\n')
#             lines = remove_values_from_list(lines, '\u2019')
#         testReviews.append(lines)
#     reviewMap = createReviewerToReviewMap()
#     
    global test
    test = []
    global training
    training = []
#     
#     for person in reviewerToReviewsMap.keys():
#         count = 0
#         l = shuffle(reviewerToReviewsMap[person])
#         for review in l:
#             if count % 2 == 0:
#                 training.append(review)
#             else: 
#                 test.append(review)
#             count += 1
#     
#     
#     for review in test: 
#         print(review)
    
    shuffle(trainingReviews)
    
    for i in range(0,math.ceil(len(trainingReviews)*.75)):
        training.append(trainingReviews[i])
    for i in range(math.ceil(len(trainingReviews)*.75), len(trainingReviews)):
        test.append(trainingReviews[i])
    
    return test, training
    
#     print("TEST")
#     for r in testReviews:
#         print(r)
#         for attribute in r:
#             print(attribute)
#         print("\n===================")
        
def showReviewListContents():
    
    print("TEST")
    for r in testReviews:
        print(r)
        for attribute in r:
            print(attribute)
        print("\n===================")
            
            
    print("TRAINING")
    for r in trainingReviews:
        print(r)
        for attribute in r:
            print(attribute)
        print("\n===================")
            
def remove_tags(input_text):
    # convert in_text to a mutable object (e.g. list)
    s_list = list(input_text)
    i,j = 0,0
    while i < len(s_list):
        # iterate until a left-angle bracket is found
        if s_list[i] == '<':
            while s_list[i] != '>':
                # pop everything from the the left-angle bracket until the right-angle bracket
                s_list.pop(i)
            # pops the right-angle bracket, too
            s_list.pop(i)
        else:
            i=i+1
    # convert the list back into text
    join_char=''
    return join_char.join(s_list)
 
#Now just pass an HTML formatted text through this function .It remove the tags and return the string
if __name__ == '__main__':
    createReviewArray()
    createReviewerToReviewMap()


