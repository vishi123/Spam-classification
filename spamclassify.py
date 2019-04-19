
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('spam.csv',encoding = "latin1")
dataset=dataset.drop(axis=1, columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'])

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 5572):
    review = re.sub('[^a-zA-Z]', ' ', dataset['v2'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
df=pd.DataFrame(y_test)

#Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB() 
classifier.fit(X_train, y_train)

#logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#support vector machine
from sklearn.svm import SVC
classifier = SVC(gamma=0.1, C=1, kernel='rbf')
classifier.fit(X_train,y_train)

#decision tree classifier
from sklearn import tree
classifier = tree.DecisionTreeClassifier() 
classifier.fit(X_train,y_train) 

#Multi layer perceptron classifier
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(500,500))
classifier.fit(X_train,y_train)

#Predicting the Test set results
y_pred = classifier.predict(X_test) 
 
#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 

acc=((cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]))*100