from bs4 import BeautifulSoup

#This may work. I have not tested, since I cant get BS working
def removeTags(html):
    return ''.join(BeautifulSoup(page).findAll(text=True))

def createDataMap():
    map = {}
    
    files = glob.glob("./stackOverFlowData/*.csv")
    for file in files:
        with open(file, encoding="latin-1") as csvFile:
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

