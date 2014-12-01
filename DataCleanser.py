import csv
import glob
import io
import re
import pickle

from bs4 import BeautifulSoup

averageNumberOfSentecesPerAnswer = 0
averageNumberOfWordsPerAnswer = 0

def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]

def removeTags(page):
    soup = BeautifulSoup(page)

    to_extract = soup.findAll('code')
    for item in to_extract:
        s = item.extract()

    return ''.join(BeautifulSoup(soup.get_text()).findAll(text=True))

def createDataMap():
    map = {}
    testList = []
    files = glob.glob("./stackOverFlowData/*.csv")
    print(files)
    for file in files:
        i = 0
        with open(file) as csvFile:
            csvs = csv.reader(csvFile)
            for row in csvs:
                if i == 0:
                    i+=1
                    continue
                
                question = removeTags(row[0])
                if question in testList:
                    continue
                userID = unicode(removeTags(row[1]))
                userName = removeTags(row[2])
                answer = removeTags(row[3])
                list = []
                
                list.append(question)
                list.append(userID)
                list.append(userName)
                list.append(answer)
                if userID not in map.keys():
                    testList.append(question)
                    llist = []
                    llist.append(list)
                    map[userID] = llist
                    i+=1
                else:
                    i+=1
                    testList.append(question)
                    map[userID].append(list)
    return map

def createArrayOfSentencesAndAuthor():
    global averageNumberOfSentecesPerAnswer
    global averageNumberOfWordsPerAnswer
    
    map = createDataMap()
    list = []
    for key, values in map.items():
        list = []
        for value in values:
            sentences = remove_values_from_list(re.split(r' *[\.\?!\n][\'"\)\]]* *', value[3]),'')
            words = value[3].split(' ')
            averageNumberOfSentecesPerAnswer += len(sentences)
            averageNumberOfWordsPerAnswer += len(words)
            list.append(({'text': sentences}, {"author":value[1]}))
             
        # write python dict to a file
        output = open('./pickle/' + key +".pickle", 'wb')
        pickle.dump(list, output)
        output.close()
       
    print("averageNumberOfWordsPerAnswer: ", averageNumberOfWordsPerAnswer)
    print("averageNumberOfSentecesPerAnswer: ", averageNumberOfSentecesPerAnswer)
    
if __name__ == '__main__':
    createArrayOfSentencesAndAuthor()

