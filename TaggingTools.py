import textblob
import nltk
from textblob import TextBlob
from textblob_aptagger import PerceptronTagger
def selectTag(multi_taggedData, tag, removePrefix=False):
    '''Expects a dictionary of multi-tagged data {"text":data, "tags":[tags]}
       Searches through a multi-tagged (tagged with a list of tags) data set and selects entries
       with a tag that is prefixed by the given tag.
       Returns a list of tuples (data,tag) where tag is only the tags whose prefix matches the passed tag
       If remove prefix is set to True, the prefix is removed from the returned tags'''
    selected = []
    for entry in multi_taggedData:        
        for t in entry["tags"]:                
            if t[:len(tag)] == tag:
                if removePrefix:
                    selected.append((entry["text"], t[len(tag):]))
                else:
                    selected.append((entry["text"], t))
                break
    return selected

def selectTags(multi_taggedData, tags, removePrefix=False):
    '''Expects a dictionary of multi-tagged data {"text":data, "tags":[tags]}
       Searches through a multi-tagged (tagged with a list of tags) data set and selects entries with tags that are
       prefixed by the given tags.
       Returns a list of tuples (data,tags) where tags are the tags whose prefix matches the passed tags
       If remove prefix is set to True, the prefix is removed from the returned tags'''
    selected = []
    for entry in multi_taggedData:  
        toAttach = []     
        for t in entry["tags"]: 
            for gt in tags:               
                if t[:len(gt)] == gt:
                    if removePrefix:
                        toAttach.append(t[len(gt):])
                    else:
                        toAttach.append(t)
        if len(toAttach) == len(tags):
            selected.append((entry["text"], toAttach))
    return selected

def taggerFunction():
    s = "This is a sentence that I am going to try and tag. Lets see if it works!"
    blob = TextBlob(s, pos_tagger=PerceptronTagger())
    
    print(blob.tags)
    
    
if __name__ == '__main__':
    taggerFunction()