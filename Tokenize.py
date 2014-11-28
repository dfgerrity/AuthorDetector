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

def byWordAlphaOnly(text):
    tokens = byWord(text)
    return [token for token in tokens if token.isalpha()]

def byWordStem(text):
    words = byWordAlphaOnly(text)
    lc = LancasterStemmer()
    return [lc.stem(word) for word in words if token.isalpha()]
    