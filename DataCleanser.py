import csv
import glob
import io

from bs4 import BeautifulSoup


def removeTags(page):
    soup = BeautifulSoup(page)

    to_extract = soup.findAll('code')
    for item in to_extract:
        s = item.extract()

    return ''.join(BeautifulSoup(soup.get_text()).findAll(text=True))

def createDataMap():
    map = {}
    
    files = glob.glob("./stackOverFlowData/*.csv")
    print(files)
    for file in files:
        with open(file) as csvFile:
            csvs = csv.reader(csvFile)
            for row in csvs:
                question = removeTags(row[0])
                if question not in map.keys():
                    userID = removeTags(row[1])
                    userName = removeTags(row[2])
                    answer = removeTags(row[3])
                    
                    list = []
                    
                    list.append(question)
                    list.append(userID)
                    list.append(userName)
                    list.append(answer)
                    
                    map[question] = list

    return map

if __name__ == '__main__':
    createDataMap()