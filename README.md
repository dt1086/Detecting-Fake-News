# Detecting Fake News
**Author**: David Tian

## Overview
This project is about detecting fake news, specifically full news articles you (the reader) may come across online. I'll be addressing the issue through analyzing news articles, primarily in the political space, to build a classification model.

### Origin of Fake News
In 2016, Buzzfeed's media editor, Craig Silverman, noticed that a stream of completely made-up stories that seemed to originate  from a small Eastern European town. After investigation, Craig and his colleague discovered that the young people in the town were manufacturing fictional stories to spread on media, in order to cash via Facebook advertising. While the topic of whether these people had an interest in American politics is up for debate, this is seen as the origin of the term "fake news", as we understand it today.<sup>[1](https://www.bbc.com/news/blogs-trending-42724320)</sup>

### Why is fake news a problem?
According to York College of Pennsylvania:
>Fake news is designed to hit you exactly in the emotions where you are weakest. They'll make you feel something, and from a statistical and psychological perspective that often means that the specific emotion being targeted is anger. That anger tends to spread like a wildfire, makes being neutral very difficult, and paradoxically prevents communication between the various sides of an argument as echo chambers develop to reinforce a side's particular stance on a topic.<sup>[2](https://library.ycp.edu/c.php?g=935163&p=6756543)</sup>

York College of PA continues to define an echo chamber as:
>A closed group of people (including media organizations) who support one another in a particular belief and keep out opposing beliefs.<sup>[2](https://library.ycp.edu/c.php?g=935163&p=6756543)</sup>

The rise of fake news is a problem because it is becoming increasingly difficult to have civil discourse, where different parties can have respectful, back-and-forth conversations with one another due to fake news reinforcing echo chambers and polarizing society as a whole.

### Why did I personally choose this as a project?
Not too long ago, while I was visiting family for the holidays, there was an animated dinner-table discussion where parties could not even agree on the most basic facts. Each party would speak their own truth, presenting news articles/sources that they were most familiar with. As a bystander, I knew that fake news was to blame, since I consider all the parties involved to be logical and well-reasoned individuals.

## Data
1. ISOT: dataset provided by the University of Victoria's Information Security and Object Technology (ISOT) research lab. The articles range from the time period: 2016 - 2017.<sup>[3](https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php)</sup>
2. Kaggle: dataset provided by University of Tennessee's Machine Learning Club.<sup>[4](https://www.kaggle.com/c/fake-news/overview)</sup> An effort was made to gain additional insights on the origins of the data, but no further description was provided. <sup>[5](https://ibb.co/ZgZ3XkH)</sup>
3. DataFlair: Insights on the origins of the data are unclear, left a couple of comments inquiring about the origins, but no response was given. Preview of the data suggest that the topic of the articles are focused on the 2016 US Election.<sup>[6](https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/)</sup>

After removing nulls, the data sources aggregate to 71,436 articles. A label of 0 indicates that an article is real, while a label of 1 indicates that an article is fake.

I created different columns where I performed various text-cleaning methods (tokenizing, removing punctuation/numbers, lemmatizing, stemming). Here is a summary of the columns I would go on to model and what they represent:
* `text_cleaned` - text from article with Reuters heading removed, punctuation, whitespaces and numbers removed
* `title` - raw title of article
* `text_tokenized_string` - `text_cleaned` with stop words removed
* `text_pos_lemmatized` - `text_tokenized_string` lemmatized based on part-of-speech tagging
* `text_snowball_stemmed` - `text_tokenized_string` stemmed based on snowball stemmer
* `text_porter_stemmed` - `text_tokenized_string` stemmed based on porter stemmer
* `text_lancaster_stemmed` - `text_tokenized_string` stemmed based on lancaster stemmer

## Modeling Process

### Managing Class Imbalance
After cleaning and exploring the data, substantial thought went into managing class imbalance. While my raw data consisted of a roughly 50/50 split between real/fake articles, research had to be done to determine what the "real-world" distribution was between real and fake news articles, in order to best train a model. According to VOX, in 2020, 17% of engagement with the top 100-performing news sources on social media was dubious.<sup>[7](https://www.vox.com/policy-and-politics/2020/12/22/22195488/fake-news-social-media-2020)</sup> Based on this, I created an imbalanced dataset, with an 80/20 split between real/fake articles, that was to be used for modeling. I've taken care to respect this imbalance for all iterations of my cross-validation process.

After train/test splitting the imbalanced dataset, my training set consisted of 26,231 real articles and 6,836 fake articles.

### Model Evaluation
Before running different models, I had to decide how I was going to evaluate the performance of the different models. In the context of fake news, my method of evaluation was to put the strongest emphasis on keeping the False Negative Rate as low as possible (when a news article is fake, how often does my model predict it to be real), since it would be detrimental to the credibility of my model to classify a fake article as real. With this in mind, other metrics I considered were to have a False Positive Rate, and a high F1 Score.

The baseline for my model results, using a dummy classifier, produced a cross-validation accurcacy score of **79%**.

### Narrowing Down Algorithms
The first question I wanted to answer in my modeling process was to narrow down the algorithms for grid-searching later on. After using TFIDFVectorizer on the `text` field, I explored a series of algorithms (Multinomial Naive Bayes, Random Forest Classifier, Passive Aggressive Classifier, XGBoost, AdaBoost, Extra Tree Classifier), and saw that the Passive Aggressive Classifer (PAC) and XGBoost (XGB) were the highest performing with the highest scores for XGB being cross-validation accuracy score of **95.4%**, False Negative Rates of **13.5%**, and F1 Scores of **88.6%**.

### Narrowing Down Data
Because TFIDFVectorizer looks to perform data-cleaning, my next step was to see whether the manually cleaned fields (removing stop words, lemmatizing, stemming) would outperform the vectorizer's cleaning of the `text_cleaned` field. 
- As expected, the title of an article (`title`) was not a stronger predictor than the body of the article (`text`)
- Removal of stop words (`text_tokenized_string`) did not improve results
- Lemmatizing (`text_pos_lemmatized`) did not improve results
- Out of the stemmers, the porter stemmer (`text_porter_stemmed`) provided the strongest results. However, stemming overall did not improve results.

### GridSearch
Now that we've narrowed down the algorithms to PAC and XGB, and narrowed our data to the raw text of the article, I then looked to performed a GridSearch in order to find the parameters that would optimize my vectorizer:
* TFIDFVectorizer
    * no stop word removal (default parameter)
    * extract unigrams and bigrams
    * build vocabulary that only consider the top features ordered by term frequency (default parameter)

Next, I looked to fine-tuning the PAC and XGB algorithms and found the following optimizations:
* PAC
    * 0.1 regularization parameter
    * hinge as the loss function
    * 100 iterations
* XGB
    * gbtree learner
    * max depth of 6

One last GridSearch I explored was to oversample/undersample the minority/majority class using SMOTE/RandomUnderSampler:
* SMOTE
    * over-sample minority class with 1:3 ratio of Fake:Real news
* RandomUnderSampler
    * under-sample majority with 1:3 ratio of Fake:Real news

### Final Model
After performing various grid searches, the best model was the Passive Aggressive Classifier, which boasts a cross-validation accuracy score of **96.3%**, a False Negative Rate of **10.8%**, and an F1 Score of **91.0%**.

After re-fitting my optimized model on my training set to make predictions on my test set, my optimized model had a test set accuracy score of **96.5%**! In addition, my model had a False Negative Rate of **10.2%** on the test set.

### Model Interpretation
The features with the highest positive coefficients in predicting an article to be fake are: 'hillary', 'on october', 'america', 'obama', and 'fbi'. <br>
The features which with the highest negative coefficients in predicting an article to be real are: 'on twitter', 'president donald'.


## Contact Information
For any additional questions, please contact me at **dt1086@stern.nyu.edu**

## Repository Structure
```
├── README.md                                         <- The top-level README for reviewers of this project
├── Modeling.ipynb                                    <- Modeling and Optimization in Jupyter Notebook
├── EDA.ipynb                                         <- Exploratory Data Exploration in Jupyter Notebook 
└──  Data                                             <- Sourced externally
```

