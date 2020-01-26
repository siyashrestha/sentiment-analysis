#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import re
import os


# In[2]:


files_pos = os.listdir('F:/anaconda/datasets/aclImdb/train/pos')
files_pos = [open('F:/anaconda/datasets/aclImdb/train/pos/'+f, 'r', encoding = 'utf8').read() for f in files_pos]
files_neg = os.listdir('F:/anaconda/datasets/aclImdb/train/neg')
files_neg = [open('F:/anaconda/datasets/aclImdb/train/neg/'+f, 'r', encoding = 'utf8').read() for f in files_neg]


# In[10]:


len(files_pos),len(files_neg)


# In[11]:


files_pos = files_pos[:2000]
files_neg = files_neg[:2000]
len(files_pos),len(files_neg)


# In[12]:


all_words = []
documents = []


# In[13]:


from nltk.corpus import stopwords
import re


# In[14]:


stop_words = list(set(stopwords.words('english')))

#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]


# In[15]:


for p in  files_pos:
    
    # create a list of tuples where the first element of each tuple is a review
    # the second element is the label
    documents.append( (p, "pos") )
    
    # remove punctuations
    cleaned = re.sub(r'[^(a-zA-Z)\s]','', p)
    
    # tokenize 
    tokenized = word_tokenize(cleaned)
    
    # remove stopwords 
    stopped = [w for w in tokenized if not w in stop_words]
    
    # parts of speech tagging for each word 
    pos = nltk.pos_tag(stopped)
    
    # make a list of  all adjectives identified by the allowed word types list above
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


# In[16]:


for p in files_neg:
    # create a list of tuples where the first element of each tuple is a review
    # the second element is the label
    documents.append( (p, "neg") )
    
    # remove punctuations
    cleaned = re.sub(r'[^(a-zA-Z)\s]','', p)
    
    # tokenize 
    tokenized = word_tokenize(cleaned)
    
    # remove stopwords 
    stopped = [w for w in tokenized if not w in stop_words]
    
    # parts of speech tagging for each word 
    neg = nltk.pos_tag(stopped)
    
    # make a list of  all adjectives identified by the allowed word types list above
    for w in neg:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


# In[17]:


len(all_words)
all_words=nltk.FreqDist(all_words)
import matplotlib.pyplot as plt
all_words.plot(20,cumulative=False)
plt.show()


# In[21]:


from wordcloud import WordCloud
text = ' '.join(all_words)
wordcloud = WordCloud().generate(text)
plt.figure(figsize=(15,9))
plt.imshow(wordcloud,interpolation ="bilinear")
plt.axis("off")
plt.show()


# In[22]:


word_features=list(all_words.keys())[:2000]
word_features
# function to create a dictionary of features for each review in the list document.
# The keys are the words in word_features 
# The values of each key are either true or false for wether that feature appears in the review or not
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# In[23]:


# Creating features for each review
featuresets = [(find_features(rev), category) for (rev, category) in documents]


# In[24]:


# Shuffling the documents 
random.shuffle(featuresets)
print(len(featuresets))


# In[25]:


training_set = featuresets[:3500]
testing_set = featuresets[3500:]
print( 'training_set :', len(training_set), '\ntesting_set :', len(testing_set))


# In[26]:


classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)


# In[27]:


MNB_clf = SklearnClassifier(MultinomialNB())
MNB_clf.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_clf, testing_set))*100)


# In[28]:


BNB_clf = SklearnClassifier(BernoulliNB())
BNB_clf.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BNB_clf, testing_set))*100)


# In[29]:


LogReg_clf = SklearnClassifier(LogisticRegression())
LogReg_clf.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogReg_clf, testing_set))*100)


# In[1]:


dataset_path='F:/anaconda/datasets/aclImdb/'


# In[2]:


import numpy
import pickle as pkl
from collections import OrderedDict
from nltk.corpus import stopwords


# In[3]:


import glob
import os
import re
import string


# In[5]:


def extract_words(sentences):
    result = []
    stop = stopwords.words('english')
    trash_characters = '?.,!:;"$%^&*()#@+/0123456789<>=\\[]_~{}|`'
    trans = str.maketrans(trash_characters, ' '*len(trash_characters))

    for text in sentences:
        text = re.sub(r'[^\x00-\x7F]+',' ', text)
        text = text.replace('<br />', ' ')
        text = text.replace('--', ' ').replace('\'s', '')
        text = text.translate(trans)
        text = ' '.join([w for w in text.split() if w not in stop])

        words = []
        for word in text.split():
            word = word.lstrip('-\'\"').rstrip('-\'\"')
            if len(word)>2:
                words.append(word.lower())
        text = ' '.join(words)
        result.append(text.strip())
    return result


# In[6]:


def grab_data(path):
    sentences = []
    currdir = os.getcwd()
    os.chdir(path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)
    sentences = extract_words(sentences)

    return sentences


# In[7]:


def main():
    
    path = dataset_path

    train_x_pos = grab_data('F:/anaconda/datasets/aclImdb/train/pos')
    train_x_neg = grab_data('F:/anaconda/datasets/aclImdb/train/neg')
    train_x = train_x_pos + train_x_neg
    train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)

    test_x_pos = grab_data('F:/anaconda/datasets/aclImdb/train/pos')
    test_x_neg = grab_data('F:/anaconda/datasets/aclImdb/train/neg')
    test_x = test_x_pos + test_x_neg
    test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)

    f = open('F:/anaconda/datasets/aclImdb/train.pkl', 'wb')
    pkl.dump((train_x, train_y), f, -1)
    f.close()
    f = open('F:/anaconda/datasets/aclImdb/test.pkl', 'wb')
    pkl.dump((test_x, test_y), f, -1)
    f.close()


# In[8]:


if __name__ == 'main':
    main()


# In[9]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import six.moves.cPickle as pickle


# In[10]:


# Load All Reviews in train and test datasets
f = open('F:/anaconda/datasets/aclImdb/train.pkl', 'rb')
reviews = pickle.load(f)
f.close()

f = open('F:/anaconda/datasets/aclImdb/test.pkl', 'rb')
test = pickle.load(f)
f.close()


# In[11]:


# Generate counts from text using a vectorizer.  
# There are other vectorizers available, and lots of options you can set.
# This performs our step of computing word counts.
vectorizer = CountVectorizer(ngram_range=(1, 3))
train_features = vectorizer.fit_transform([r for r in reviews[0]])
test_features = vectorizer.transform([r for r in test[0]])


# In[12]:


# Fit a naive bayes model to the training data.
# This will train the model using the word counts we computer, 
#       and the existing classifications in the training set.
nb = MultinomialNB()
nb.fit(train_features, [int(r) for r in reviews[1]])


# In[13]:


# Now we can use the model to predict classifications for our test features.
predictions = nb.predict(test_features)


# In[14]:


# Compute the error.  
print(metrics.classification_report(test[1], predictions))
print("accuracy: {0}".format(metrics.accuracy_score(test[1], predictions)))


# In[ ]:


while True:
    sentences = []
    sentence = input("\n\033[93mPlease enter a sentence to get sentiment evaluated. Enter \"exit\" to quit.\033[0m\n")
    if sentence == "exit":
        print("\033[93mexit program ...\033[0m\n")
        break
    else:
        sentences.append(sentence)
        input_features = vectorizer.transform(extract_words(sentences))
        prediction = nb.predict(input_features)
        if prediction[0] == 1 :
            print("---- \033[92mpositive\033[0m\n")          #not severe
        else:
            print("---- \033[91mnegative\033[0m\n")         #severe


# In[ ]:




