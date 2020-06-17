import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'maxent_treebank_pos_tagger', 'stopwords'])

from nltk import word_tokenize, pos_tag
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords 
import re
import joblib

def get_wordnet_pos(treebank_tag):

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

    # tokenize text
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if re.fullmatch('[a-z]+', w)] 
    tokens = [w for w in tokens if not w in stopwords.words('english')] 
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok, tag in pos_tag(tokens):
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok, get_wordnet_pos(tag))
        clean_tokens.append(clean_tok)

    return (' ').join(clean_tokens)