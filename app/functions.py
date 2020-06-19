import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'maxent_treebank_pos_tagger', 'stopwords'])

from nltk import word_tokenize, pos_tag
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords 
import re
import joblib

stopwords_set = set(stopwords.words('english'))

def get_wordnet_pos(treebank_tag):
    '''This function converts Treebank part-of-speech tags into Wordnet POS tags, if theres no match,
    return NOUN

    Parameters:
    treebank_tag (string): Treebank POS tag

    Returns:
    string: Wordnet POS tag
    '''

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
        
def preprocess(text):
    '''Main text cleaning function, to be ran rowwise on source data

    Parameters:
    text (string): Non-altered input text

    Returns:
    string: lemmatized text without stopwords and words containing special characters/numbers
    '''

    # tokenize text
    tokens = word_tokenize(text.lower())
    # only alphabetical characters
    tokens = [w for w in tokens if re.fullmatch('[a-z]+', w)]
    # remove stopwords
    tokens = [w for w in tokens if not w in stopwords_set] 
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok, tag in pos_tag(tokens):
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok, get_wordnet_pos(tag))
        clean_tokens.append(clean_tok)

    # we concatenate the text again, so we can use the default tokenizer of TfidfVectorizer
    return (' ').join(clean_tokens)