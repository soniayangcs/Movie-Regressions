from flask import Flask, jsonify, render_template, request, redirect
from sklearn.externals import joblib
import pandas as pd
#Remove stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer

REPLACE_BY_SPACE_REREPLACE_  = re.compile('[/(){}\[\]\|@,;]"')
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
        vec = TfidfVectorizer()
        vec.fit(list(cleanword))
        transformed = vec.transform(list(cleanword))
        text_features = pd.DataFrame(transformed.todense())
        text_features.columns = vec.get_feature_names()
        prediction = clf.predict(testword)
        return jsonify({'prediction': list(testword)})
    
    return render_template('form.html')
     
if __name__ == '__main__':
     clf = joblib.load('model.pkl')
     app.run()