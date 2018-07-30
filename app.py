from flask import Flask, jsonify, render_template, request, redirect
from sklearn.externals import joblib
import pandas as pd
import numpy as np
#Remove stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

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

def get_models():
    """Generate a library of base learners."""
    nb = GaussianNB()
    svc = SVC(C=100, probability=True)
    knn = KNeighborsClassifier(n_neighbors=3)
    lr = LogisticRegression(C=100, random_state=0)
    nn = MLPClassifier((80, 10), early_stopping=False, random_state=0)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=0)
    rf = RandomForestClassifier(n_estimators=10, max_features=3, random_state=0)

    models = {'svm': svc,
              'knn': knn,
              'naive bayes': nb,
              'mlp-nn': nn,
              'random forest': rf,
              'gbm': gb,
              'logistic': lr,
              }

    return models


def train_predict(model_list):
    """Fit models in list on training set and return preds"""


    print("Fitting models.")
    model_results = {}
    for i, (name, m) in enumerate(get_models().items()):
        print("%s..." % name, end=" ", flush=False)
        m.fit(x_train_f, y_train)
        model_results[name] = m.predict(X_test)
        print("done")
    
    #import pdb; pdb.set_trace()
    P = pd.DataFrame(model_results)
    print("Done.\n")
    
    lr = LogisticRegression(C=100, random_state=0)
    lr.fit(P,y_test)

    ensemble=lr.predict(P)
    return ensemble
    
#set up dataframe
df = pd.DataFrame()
df["popularity"] = [0]
df['Crime_genre'] = [0]
df['vote_count'] = [0]
df['Western_genre'] = [0]
df['Adventure_genre'] = [0]
df['Science Fiction_genre'] = [0]
df['Dec_releaseMon'] = [0]
df['May_releaseMon'] = [0]
df['Oct_releaseMon']=[0]
df['Jul_releaseMon']=[0]
df['Mar_releaseMon']=[0]
df['Nov_releaseMon']=[0]
df['Apr_releaseMon']=[0]
df['Jun_releaseMon']=[0]
df['Feb_releaseMon']=[0]
df['Aug_releaseMon']=[0]
df['Sep_releaseMon']=[0]
df['Jan_releaseMon']=[0]

app = Flask(__name__)

#chris add
#@app.route("/")
#def index():
#    return render_template('index2.html')


@app.route('/predicto', methods=['GET','POST'])
def poverview():
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
        #return jsonify([{'prediction': list(prediction), 'overview': list(testword)}])
        return render_template("overviewresult.html", prediction="$"+str(int(prediction[0])),popularity=predictionp[0],overview=testword)
    
    return render_template('overviewform.html')

@app.route('/predicte', methods=['GET','POST'])
def pensemble():
    if request.method == 'POST':
        popularity = request.form["pop"]
        vote_count = request.form["vote"]
        genre = request.form["genre"]
        month = request.form["month"]
        df[month]=[1]
        df["popularity"] = [int(popularity)]
        df["vote_count"] = [int(vote_count)]
        if genre == "Other_genre":
            df['Western_genre'] = [0]
        else:
            df[genre]=[1]
        predictione = clfe.predict(df.values).tolist()
        #return jsonify(json.dumps(json.loads(df.to_json(orient='records'))))
        return render_template("ensembleresult.html", popularity=popularity,vote_count=vote_count, genre=genre,month=month) 

    return render_template('ensembleform.html')
     
if __name__ == '__main__':
     # Load your vectorizer
     vec = joblib.load("NLP_vectorizer_revenue.pkl")
     vecp = joblib.load("NLP_vectorizer_popularity.pkl")
     clf = joblib.load("NLP_model_revenue.pkl")
     clfp = joblib.load("NLP_model_popularity.pkl")
     clfe = joblib.load("ensemble_model.pkl")
     app.run(debug=True)