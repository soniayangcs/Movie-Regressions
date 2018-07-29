from flask import Flask, jsonify, render_template, request, redirect
from sklearn.externals import joblib
import pandas as pd
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

@app.route('/predict', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        testword = request.form["wordName"]
        cleanword = text_prepare(testword)
        transformed = vec.transform([cleanword])
        text_features = pd.DataFrame(transformed.todense())
        text_features.columns = vec.get_feature_names()
        prediction = clf.predict(text_features.values).tolist()
        #return jsonify({'prediction': list(prediction)})
        return render_template("results.html", prediction=prediction) 

    return render_template('form.html')
     
if __name__ == '__main__':
     # Load your vectorizer
     vec = joblib.load("vectorizer.pkl")
     clf = joblib.load('model.pkl')
     app.run()