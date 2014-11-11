import nltk

def byWord(text):
    pattern = r'''(?x) # set flag to allow verbose regexps
    ([A-Z]\.)+ # abbreviations, e.g. U.S.A.
    | \w+(-\w+)* # words with optional internal hyphens
    | \$?\d+(\.\d+)?%? # currency and percentages, e.g. $12.40, 82%
    | \.\.\. # ellipsis
    | [][.,;"'?():-_`]
    '''
    return (nltk.regexp_tokenize(text.lower(), pattern))
    