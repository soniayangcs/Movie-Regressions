# Movie Regressions
Using regression models to predict the effects of different attributes on box office performances and verification that descriptor can be used as a predictor of movie performance.

## Members
* Chris 
* Sue Helen
* Lindsay
* Nana
* Sonia

## Data Source:
* [Kaggle DataSet](https://www.kaggle.com/tmdb/tmdb-movie-metadata#tmdb_5000_movies.csv/) - Movie

## Running Flask and Heroku Deployment
* To create an interactive experience for the client, we deployed both [Profitability Prediction Class model](https://meowmovie5000meow.herokuapp.com/predicte) and [NLP Model](https://meowmovie5000meow.herokuapp.com/predicto).

### Running Flask
* To run Flask successfully, we need to:
    1. Download all the models using `joblib` through ipython notebook;
    2. Load all the models into `app.py` when preparing the python code for running the Flask;
    3. Create html templates that can successfully capture all the data we need in the correct form and display results in a readable format;
    4. Perform **feature engineering** on data received from html. For NLP Models, we also need to load in the vectorizers that had been trained in ipython notebook to vectorize the overview text. 

* Roadblock encountered: 
    1. The Profitability Prediction Class model first used 7 different models to predict the result and then used ensemble method to make the final prediction. Solely using the ensemble model package was not sufficient to predict the result (will have code error). As such, we need to download all 7 prediction models in addition to the ensemble model and fit the overview text through all these models. 

### Heroku Deployment
* To deploy model successfully through Heroku, we need to:
    1. Include Procfile to initial python running on `app.py`;
    2. Include `requirement.txt` file to specify all packages/libraries needs to be downloaded in order to make the `app.py` run           successfully through Heroku;
    3. Include all pickled models, `app.py`, html templates (with associated CSS files);

* Roadblocks encountered:
    1. All python packages/libraries used in either ipython notebook or in `app.py` needs to be included in the `requirement.txt`.
    2. For package `nltk`, solely include the name in the requirements is not sufficient. we also need to create a new text file             `nltk.txt` to specify the **corpora** (for our purpose, we need **stopword**) needed in the code. 
    3. The size of the files (including all models, packages, libraries) deployed to Heroku needs to be smaller than 360.0MB. As such,     we need to create a new repo solely for deployment of the models.
    
## Story


```
Input story text here.
```

## Possible Predictive models:
```
How does Revenue correlate
How does Rating correlate
```

## INPUTS: (Red items are omitted)

```
budget
genres
homepage
id
keywords
original_language
original_title
overview
popularity
production_companies
production_countries
Release_date (Broken down to the release month) 
revenue
runtime
spoken_languages
status
tagline
title
vote_average
Vote_count
```

* note: popularity score is calculated based on several factors, such as number of votes, views, users who marked the movie as favorite and/or added is to their watchlist, etc. -> https://developers.themoviedb.org/3/getting-started/popularity

## VISUALIZATIONS

Tableau: 
* this visualization was created based off the cleaned CSV data only to see if there are any correlations or trends before using machine learning
* a 'pre-story' of sorts to see if our predictions match up and have similar trends or differ
* graphs examining budget, popularity, and time of the year vs. gross revenue: https://public.tableau.com/profile/sonia.yang#!/vizhome/MovieData_37/FactorsAffectingRevenue
* graphs examining popularity vs. vote average, and also movie genres ranked from most to least common: https://public.tableau.com/profile/sonia.yang#!/vizhome/MovieData_37/VoteAverageSummary
* graphs showing the top grossing films for the four most common movie genres (drama, comedy, thriller, action): https://public.tableau.com/profile/sonia.yang#!/vizhome/MovieData_37/TopByGenre
