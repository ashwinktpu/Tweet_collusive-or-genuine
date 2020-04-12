#Data Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Data Preprocessing and Feature Engineering
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Model Selection and Validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score


fullData = pd.read_csv("C:/Users/ashwina/Desktop/tweets.csv")
fullData.head()


from sklearn import preprocessing
number = preprocessing.LabelEncoder() 

#Target variable is also a categorical so convert it
fullData["label"] = number.fit_transform(fullData["label"].astype('str'))
fullData.head()



import nltk
nltk.download('punkt')
def form_sentence(tweet):
    tweet_blob = TextBlob(tweet)
    return ' '.join(tweet_blob.words)

#print(form_sentence(fullData['tweet_text'].iloc[10]))
#print(fullData['tweet_text'].iloc[10])


import nltk
nltk.download('stopwords')
def no_user_alpha(tweet):
    tweet_list = [ele for ele in tweet.split() if ele != 'user']
    clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
    clean_s = ' '.join(clean_tokens)
    clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
    return clean_mess
#print(no_user_alpha(form_sentence(fullData['tweet_text'].iloc[10])))
#print(fullData['tweet_text'].iloc[10])


import nltk
nltk.download('wordnet')
def normalization(tweet_list):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized_tweet.append(normalized_text)
        return normalized_tweet

#tweet_list = 'I was playing with my friends with whom I used to play, when you called me yesterday'.split()
#print(normalization(tweet_list))

from sklearn.feature_extraction.text import CountVectorizer
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer='word')),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


msg_train, msg_test, label_train, label_test = train_test_split(fullData['tweet_text'], fullData['label'], test_size=0.20)
pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
print(classification_report(predictions,label_test))
print(confusion_matrix(predictions,label_test))
print(accuracy_score(predictions,label_test))
