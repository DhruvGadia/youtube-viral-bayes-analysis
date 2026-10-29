# YouTube Viral Video Analysis Using Bayes’ Theorem

A mini-project exploring first-trending YouTube videos in India using Bayesian probabilities and an R naive Bayes classifier.

## Data

Source: [Kaggle Trending YouTube Video Statistics](https://www.kaggle.com/datasets/datasnaek/youtube-new). Download `INvideos.csv` and `IN_category_id.json` into a local folder. The analysis uses first observed trending records from 2017–2018. Trending selection means findings cannot represent all YouTube uploads.

## Run

Install R and the `jsonlite` package (`install.packages("jsonlite")`). Edit `raw_dir` in `youtube_viral_bayes_project.R` to point to the downloaded files, then run the script. It writes cleaned data and model results to an `outputs` folder.

The viral label denotes the top quartile of first-trending views, based on the training set. Engagement rates are measured at the first trending snapshot, so they cannot predict virality before publication. Results show associations, not causal effects.

The original presentation and analysis outputs were calculated separately from the R script. Random sampling differs across implementations, so the R output will not reproduce the presentation's exact figures. The script has not been executed in R in the original environment.
