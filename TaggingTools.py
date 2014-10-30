
def selectTag(multi_taggedData, tag):
    '''Expects a dictionary of multi-tagged data {"text":data, "tags":[tags]}
       Searches through a multi-tagged (tagged with a list of tags) data set and selects entries with a tag is prefixed by the given tag.
       Returns a list of tuples (data,tag) where tag is only the tags whose prefix matches the passed tag'''
    selected = []
    for entry in multi_taggedData:        
        for t in entry["tags"]:                
            if t[:len(tag)] == tag:
                selected.append((entry["text"], t[len(tag):]))
    return selected