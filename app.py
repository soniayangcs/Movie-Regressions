from flask import Flask, jsonify, render_template, request, redirect
from sklearn.externals import joblib
import pandas as pd
import numpy as np
#Remove stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]"')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    
    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE," ",text)
    text = re.sub(BAD_SYMBOLS_RE,"",text)
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    
    return text

app = Flask(__name__)

#chris add
@app.route("/")

def main():
    return render_template('index2.html')
if __name__ == "__main__":
    app.debug = True
    app.run()


@app.route('/predict', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        testword = request.form["wordName"]
        cleanword = text_prepare(testword)
        transformed = vec.transform([cleanword])
        transformedp = vecp.transform([cleanword])
        text_features = pd.DataFrame(transformed.todense())
        text_featuresp = pd.DataFrame(transformedp.todense())
        text_features.columns = vec.get_feature_names()
        text_featuresp.columns = vecp.get_feature_names()
        prediction = clf.predict(text_features.values).tolist()
        predictionp = clfp.predict(text_featuresp.values).tolist()
        #return jsonify({'prediction': list(prediction); 'overview': list(testword)})
        return render_template("results.html", prediction="$"+str(int(prediction[0])),popularity=predictionp[0],overview=testword) 

    return render_template('form.html')
     
if __name__ == '__main__':
     # Load your vectorizer
     vec = joblib.load("vectorizer.pkl")
     vecp = joblib.load("vectorizer_popularity.pkl")
     clf = joblib.load("model.pkl")
     clfp = joblib.load("model_popularity.pkl")
     
     app.run()