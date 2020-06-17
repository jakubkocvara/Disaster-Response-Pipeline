import sys
import os
import pandas as pd
from sqlalchemy import create_engine

import joblib
import multiprocessing as mp

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV

sys.path.append(os.path.abspath("lib"))
from functions import preprocess

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('select * from cleaned', con = engine)
    category_names = df.columns[4:].tolist()
    return df['message'], df[category_names], category_names


def preprocess_text(X, parallel = True):
    print(X.shape[0])
    if parallel:
        with mp.Pool(mp.cpu_count()) as pool:
            preprocessed = pool.map(preprocess, X)
    else:
        X = X.apply(preprocess)
    return X

def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs = -1))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    # predict on test data
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns = Y_test.columns.tolist())
    df_metrics = pd.DataFrame(columns = ['column', 'precision', 'recall', 'f1_score'])
    for col in Y_test.columns.tolist():
        precision = precision_score(Y_test[col], Y_pred_df[col], average = 'weighted')
        recall = recall_score(Y_test[col], Y_pred_df[col], average = 'weighted')
        f1 = f1_score(Y_test[col], Y_pred_df[col], average = 'weighted')
        row = pd.DataFrame([[col, precision, recall, f1]], columns = ['column', 'precision', 'recall', 'f1_score'])
        df_metrics = df_metrics.append(row)
    print(df_metrics)


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        print('Preprocessing text...')

        X = preprocess_text(X)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()