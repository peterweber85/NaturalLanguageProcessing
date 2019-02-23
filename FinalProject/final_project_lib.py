
import numpy as np
import pandas as pd
import math
import random
import string
from string import punctuation

from nltk.corpus import movie_reviews
from nltk.corpus import stopwords

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import gensim

class SentimentAnalysis:
    """
    Class for Movie Reviews Sentiment Analysis
    """        
    def read_documents(self):
        """
        Creates: 
            - self.words: (list of list) of reviews
            - self.sentiments: (list) of sentiments
        """
        words_cat = [(list(movie_reviews.words(fileid)), category)
                        for category in movie_reviews.categories()
                        for fileid in movie_reviews.fileids(category)]
        
        np.random.shuffle(words_cat)
        self.words = [document[0] for document in words_cat]
        self.sentiments = [document[1] for document in words_cat]
        print("Documents loaded!")

    def get_documents(self):
        """
        Getter for self.word and self.sentiments
        """
        return(self.words, self.sentiments)
    
    def __clean_review__(self,tokens):
        """        
        Taken from: https://machinelearningmastery.com/
        deep-learning-bag-of-words-model-sentiment-analysis/

        Inputs:
            - tokens: tokenized review
        Outputs:
            - tokens: cleaned review
        """
        stop_words = set(stopwords.words('english'))
        ## remove punctuation from each token
        tokens = [word for word in tokens if not word in string.punctuation]
        ## remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        ## filter out stop words
        tokens = [w for w in tokens if not w in stop_words]
        ## filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        return tokens

    def __convert_sentiment__(self,sentiment):
        """
        Converts sentiment to boolean => 0/1
        """
        if sentiment == 'neg':
            return 0
        elif sentiment == 'pos':
            return 1

    def clean_documents(self):
        """        
        Creates:
            - self.docs: (list of list) of clean reviews
            - self.sens: (list) of boolean (0/1) sentiments
        """
        self.words = [self.__clean_review__(words) for words in self.words]
        self.sentiments = [self.__convert_sentiment__(sentiment) for sentiment in self.sentiments]
        print("Documents cleaned!")


    def create_ngram_matrix(self, n = 3, max_feat = None, binary = False):
        """
        Inputs: 
            - n: (int) n of ngram
            - max_feat: (int) maximum features to consider, ordered by occurence
            - binary: (bool) integer or binary features
        Outputs:
            - X: (ndarray) ngram count matrix
            - self.sentiments: (list) of sentiments
        """
        words = [" ".join(word) for word in self.words]
        ngram_vectorizer = CountVectorizer(ngram_range = (1, n),
                                           max_features = max_feat, 
                                           binary = binary)
        X = ngram_vectorizer.fit_transform(words).toarray()
        return(X, self.sentiments)

    def create_tfidf_matrix(self, n = 3, max_feat = None, binary = False, inplace = False):
        """
        Inputs: 
            - n: (int) n of ngram
            - max_feat: (int) maximum features to consider, ordered by occurence
            - binary: (bool) integer or binary features
        Outputs:
            - X: (ndarray) ngram tfidf count matrix
            - self.sentiments: (list) of sentiments
        """
        words = [" ".join(word) for word in self.words]
        tfidf_vectorizer = TfidfVectorizer(ngram_range = (1, n),
                                               max_features = max_feat, 
                                               binary = binary)
        X = tfidf_vectorizer.fit_transform(words).toarray()
        return(X, self.sentiments)

    def cross_validate_random_forest(self, X, y):
        """
        Inputs:
            - X (ndarray): feature matrix, whole dataset
            - y (list or ndarray): target vector, whole dataset
        Outputs:
            - score: (list) 4-fold cross validation score computed using rf classifier
            - prediction: (list) of predicted sentiments with dimension of y
        """
        clf = RandomForestClassifier(n_estimators=1000, max_depth=2, random_state=0)
        score = cross_val_score(clf, X, y, cv=4) 
        prediction = cross_val_predict(clf, X, y, cv=4)
        return(score, prediction)

    def train_doc2vec(self): 
        """
        Operates on self.words

        Outputs:
            - model: trained doc2vec model
        """    
        reviews = [gensim.models.doc2vec.LabeledSentence(
            words = self.words[idx], tags = ['REVIEW_%s' % idx]) for idx in range(len(self.words))] 
        
        model = gensim.models.Doc2Vec(vector_size = 2000, workers = 10, window = 5, alpha=.025, min_alpha=.025, min_count=1)
        model.build_vocab(reviews)

        for epoch in range(10):
            model.train(reviews, total_examples = model.corpus_count, epochs = model.epochs)
            model.alpha -= 0.002   # decrease the learning rate`
            model.min_alpha = model.alpha  # fix the learning rate, no decay
        return(model)

    def create_embedding_matrix(self, model):
        """
        Creates word embeddings for the corpus based on the model.
        One review is represented by three embedding vectors: min, max, and mean of
        every vector element.

        Inputs: 
            - (model) output of train_doc2vec
        Outputs:
            - X: (ndarray) of review embeddings, see above
            - y: (list) of sentiments

        """
        review_embedding_min = np.array([model.wv[token].min(axis = 0) for token in self.words])
        review_embedding_max = np.array([model.wv[token].max(axis = 0) for token in self.words])
        review_embedding_mean = np.array([model.wv[token].mean(axis = 0) for token in self.words])
        X = np.c_[review_embedding_min, review_embedding_mean, review_embedding_max]
        return(X, self.sentiments)

    def most_common(self,array):
        """
        Helper for vote majority.
        Inputs:
            - array: (ndarray) with dimension 3 x n
        Outputs:
            - most common element for every row of input array
        """
        lst = array.tolist()
        return max(set(lst), key=lst.count)

    def vote_majority(self,y1,y2,y3):
        """
        Inputs:
            - y1, y2, y3: 3 (list's/ndarray's) of predicted sentiments
        Outputs:
            - y: most common element for every row for concatenated inputs
        """
        y = np.c_[y1, y2, y3]
        y = np.apply_along_axis(self.most_common, 1, y)
        return(y)