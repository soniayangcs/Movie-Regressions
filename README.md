# Project-3.-Movie-Regressions
Project 3: Movie regression models to predict the effects of different attributes on box performances and verification that descriptor can be used as a predictor of movie performance.

## Members
* Chris 
* Sue
* Helen
* Nana
* Sonia

## Data Source:
* [Kaggle DataSet](https://www.kaggle.com/tmdb/tmdb-movie-metadata#tmdb_5000_movies.csv/) - Movie

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
* graphs examining budget, popularity, and time of the year vs. gross revenue: https://public.tableau.com/profile/sonia.yang#!/vizhome/MovieData_37/FactorsAffectingRevenue?publish=yes
* graphs examining popularity vs. vote average, and also movie genres ranked from most to least common: https://public.tableau.com/profile/sonia.yang#!/vizhome/MovieData_37/VoteAverageSummary
* graphs showing the top grossing films for the four most common movie genres (drama, comedy, thriller, action): https://public.tableau.com/profile/sonia.yang#!/vizhome/MovieData_37/TopByGenre