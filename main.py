import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


##NLP model
ds = pd.read_csv(r"C:\Users\Rachit\Desktop\spam$ham\spam.csv")
ds= ds.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
ds.rename(columns={'v1':'labels','v2':'message'}, inplace=True)
ds.drop_duplicates(inplace=True)
ds['labels'] = ds['labels'].map({'ham':0,'spam':1})
#print(ds.head())

def clean_data(message):
    msg_without_punc = [character for character in message if character not in string.punctuation]
    msg_without_punc = ''.join(msg_without_punc)

    seperator = ' '
    return seperator.join([word for word in msg_without_punc.split() if word.lower() not in stopwords.words('english')])

ds['message'] = ds['message'].apply(clean_data)

x = ds['message']
y = ds['labels']
cv = CountVectorizer()

x = cv.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0 )

model = MultinomialNB().fit(x_train, y_train)

predict = model.predict(x_test) 

#print(accuracy_score(y_test, predict))
#print(confusion_matrix(y_test, predict))
#print(classification_report(y_test, predict))

def pred(text):
    labels = ['Not Spam','Spam']
    x = cv.transform(text).toarray()
    p = model.predict(x)
    s = [str(i) for i in p]
    v = int(''.join(s))
    return str('This message is looking to be: '+labels[v])

st.image('images.jpg')
st.title('HAM & SPAM Classifier')
user_input = st.text_input('Write your text.')
submit = st.button('Predict')
if submit:
    answer = pred([user_input])
    st.text(answer)
