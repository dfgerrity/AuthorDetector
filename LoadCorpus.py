import glob
import pickle
import copy
import random
import Tokenize

def loadCorpus():
    corpus = {}
    files = glob.glob("./pickle/*.pickle")
    #print(files)
    total=0
    for file in files:
        #print("Loading file:", file)
        entries = pickle.load(open(file, "rb"))
        #print("Loaded", len(entries),"entries")
        for entry in entries:
            #input(entry)
            total +=1
            if entry[1]["author"] in corpus.keys():
                corpus[entry[1]["author"]].append(entry)
            else:
                corpus[entry[1]["author"]] = [entry]

    print("Constructed corpus with",total,"entries, with an average of", int(total/len(corpus.keys())), "per author.") 
    return corpus

MIN_SAMPLES = 1000
MIN_SAMPLE_LEN = 5

def normalize(data):
    normal = {}
    print("Normalizing Corpus...")
    for author in data.keys():
        goodSamples = []
        for sample in data[author]:
            if len(sample[0]["text"]) > MIN_SAMPLE_LEN:
                goodSamples.append(sample)
        if len(goodSamples) > MIN_SAMPLES:
            random.shuffle(goodSamples)
            normal[author] = goodSamples[:MIN_SAMPLES]
    print("Normalized corpus to contain only the", len(normal.keys()),"authors who have",MIN_SAMPLES,"samples of",MIN_SAMPLE_LEN,"sentences or more.")
    return normal 

def getPerAuthorTraining():
    training_sets = {}
    corpus = loadCorpus()
    corpus = normalize(corpus)
    for author in corpus.keys():
        training_sets[author] = copy.deepcopy(corpus[author])  
        positiveCount = len(training_sets[author])  
        print("Number of positive samples:", positiveCount)    
        for i in range(positiveCount):
            imposter = random.choice(list(corpus.keys()))
            picked = []
            pick = random.randint(0,len(corpus[imposter])-1)
            while author + str(pick) in picked:
                pick = random.randint(len(corpus[imposter]))
            picked.append(author + str(pick))
            training_sets[author].append(corpus[imposter][pick])
        print("Total number of samples:", len(training_sets[author])) 
    print("Built",len(training_sets),"training sets for each author with a even number of positive and negative samples.")
    return training_sets


#getPerAuthorTraining()