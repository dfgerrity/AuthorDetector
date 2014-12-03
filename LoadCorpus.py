import glob
import pickle
import copy
import random
import Tokenize
import os
import DataCleanser

def loadCorpus():
    corpus = {}
    files = glob.glob("./pickle/*.pickle")
    #print(files)
    total=0
    for file in files:
        #print("Loading file:", file)
        f = open(file, "rb")
        entries = pickle.load(f)
        f.close()
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

def loadFeatureSets(path):
    '''Expects path to be pickle file of a list of featureSets: ({name:val...}, label)'''
    print("Loading feature sets from file:", path)
    f = open(path, "rb")
    featureSets = pickle.load(f)
    f.close()
    return featureSets

def loadNgramsSets(path):
    '''Expects path to be pickle file of a dict of ngram probabilites: {name:val...}'''
    print("Loading ngram sets from file:", path)
    f = open(path, "rb")
    ngrams = pickle.load(f)
    f.close()
    return ngrams

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

def makeTXTPerNsamples(n):
    print("Loading Normalized Corpus...")
    corpus = normalize(loadCorpus())
    count = 0 
    if not os.path.isdir("./TXT"):
        os.mkdir("./TXT")
    for author in corpus.keys():
        print("Generating txt files for author", count)
        count+=1
        samples = corpus[author]
        oneFile = []
        for i in range(len(samples)):
            oneFile.append(samples[i])
            if len(oneFile) ==  n:
                if not os.path.isdir("./TXT/"+author):
                    os.mkdir("./TXT/"+author)
                output = open("./TXT/"+author+"/record"+str(i)+".txt", "w")
                for sample in oneFile:
                    output.writelines(filter(DataCleanser.onlyascii," ".join(sample[0]["text"])))
                output.close()
                oneFile = []

def getTaggedSamples(samplesPerTag=1000):
    print("Loading Normalized Corpus...")
    corpus = normalize(loadCorpus())

    print("Collapsing corpus into one big list")
    everything = []
    for author in corpus.keys():
        start = random.randint(0,len(corpus[author]) - samplesPerTag)
        everything.extend(corpus[author][start:start+samplesPerTag])

    print("Converting to (data, label) format")
    tagged_samples = [(e[0]["text"],e[1]["author"]) for e in everything]
    return tagged_samples

def makeTXTtrainAndtestPerNsamples(n):
    print("Loading Normalized Corpus...")
    corpus = normalize(loadCorpus())
    count = 0 
    if not os.path.isdir("./TXTtrain"):
        os.mkdir("./TXTtrain")
    if not os.path.isdir("./TXTtest"):
        os.mkdir("./TXTtest")
    for author in corpus.keys():
        print("Generating txt files for author", count)
        count+=1
        samples = corpus[author]
        oneFile = []
        for i in range(len(samples)):
            oneFile.append(samples[i])
            if len(oneFile) ==  n:
                if random.random() > 0.25:
                    if not os.path.isdir("./TXTtrain/"+author):
                        os.mkdir("./TXTtrain/"+author)
                    output = open("./TXTtrain/"+author+"/record"+str(i)+".txt", "w")
                    for sample in oneFile:
                        output.writelines(filter(DataCleanser.onlyascii," ".join(sample[0]["text"])))                    
                else:
                    if not os.path.isdir("./TXTtest/"+author):
                        os.mkdir("./TXTtest/"+author)
                    output = open("./TXTtest/"+author+"/record"+str(i)+".txt", "w")                    
                    for sample in oneFile:
                        output.writelines(filter(DataCleanser.onlyascii," ".join(sample[0]["text"])))
                output.close()
                oneFile = []

#makeTXTtrainAndtestPerNsamples(1)
