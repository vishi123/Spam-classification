# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:03:28 2019

@author: Manvi Gupta
"""

"""from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
  return "Hi, this is client server model!"

if __name__ == "__main__":
  app.run()"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [str(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)
    import re
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    corpu = []
    revie = re.sub('[^a-zA-Z]', ' ', str(int_features))
    revie = revie.lower()
    revie = revie.split()
    ps = PorterStemmer()
    revie = [ps.stem(word) for word in revie if not word in set(stopwords.words('english'))]
    revie = ' '.join(revie)
    corpu.append(revie)
    cv = CountVectorizer(max_features = 2000)
    Xi = cv.fit_transform(corpu).toarray()
    #Xi=np.reshape(Xi, (1, 2000))
    print(Xi)
    prediction = model.predict(Xi)

     
    #output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)