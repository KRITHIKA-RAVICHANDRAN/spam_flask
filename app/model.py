#import statements

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



#----------------Input Data and Preprocessing-----------------

#1.Read the CSV file

df = pd.read_csv('spam.csv', encoding="latin-1")

#2.Drop the columns Unnamed:2, Unnamed:3, Unnamed:4
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

#3. Rename the columns v1 as label and v2 as message
df.rename(columns={'v1': 'label','v2': 'message'}, inplace=True)

#4. Map all ham labels to 0 and spam values to 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

#5. Assign Message column to X
X = df['message']

#6. Assign label column to Y
Y= df['label']


#---------------------------Feature Extraction----------------

#7.Initialise the countvectorizer
cv=CountVectorizer()

#8.Fit tranform the data X in the vectorizer and store the result in X
X = cv.fit_transform(X)


#9.save your vectorizer in 'vector.pkl' file
pickle.dump(cv, open("vector.pkl", "wb"))


#------------------------Classification---------------------

'''10. Split the dataset into training data and testing data with train_test_split function
Note: parameters test_size=0.33, random_state=42'''

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

#11. Initialise multimimial_naive_bayes classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()

#12.Fit the training data with labels in Naive Bayes classifier 'clf'
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
y_pred = clf.predict(X_test)
#print(classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred))
#13. Store your classifier in 'NB_spam_model.pkl' file
joblib.dump(clf, 'NB_spam_model.pkl')