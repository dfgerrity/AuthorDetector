
def selectTag(multi_taggedData, tag, removePrefix=false):
    '''Expects a dictionary of multi-tagged data {"text":data, "tags":[tags]}
       Searches through a multi-tagged (tagged with a list of tags) data set and selects entries with a tag is prefixed by the given tag.
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
    return selected