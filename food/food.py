'''
Created on Oct 21, 2014

@author: computer
'''

import glob
import fileinput
import nltk   
from urllib.request import urlopen
from sys import argv

testReviews = []
trainingReviews = []
reviewerToReviewsMap = {}

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
    for tuple in trainingReviews:

        if tuple[0] in reviewerToReviewsMap:
            reviewerToReviewsMap[tuple[0]].append(tuple)
        else:
            reviewerToReviewsMap[tuple[0]] = [tuple]
    return reviewerToReviewsMap

# Written review is given its own index in the array. The next 4 indices
# are the paragraphs of the review. If we need to join them later we can
def createReviewArray():
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
      
      
    files = glob.glob("./testOut/*.html")
    
    for testFile in files:
        print(testFile)
        txt = open(testFile, encoding='utf-8')
        lines = txt.readlines()
        
        if '\n' in lines or '\xa0' in lines or '\xa0\n' in lines or '\u2019' in lines:
            lines = remove_values_from_list(lines, '\n')
            lines = remove_values_from_list(lines, '\xa0')
            lines = remove_values_from_list(lines, '\xa0\n')
            lines = remove_values_from_list(lines, '\u2019')
        testReviews.append(lines)
    return testReviews, trainingReviews
    
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
    createReviewerToReviewmap()


