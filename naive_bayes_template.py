### Naive Bayes Gender Classifer Template

# This assumes you have the nltk installed in your enviroment & have downloaded the 'names' corpus.
# Change the metric to one of the defined features functions or write your own
# Source the script to retrive the output or import the definitions in the console with 'from naive_bayes_template import *'

# Credit: http://www.nltk.org/book/ch06.html

## Library
import nltk

## Classifiers

# gender_features
def gender_features(name):
  return {'last_letter': name[-1]}
  
# gender_features_overfit
def gender_features_overfit(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features

# gender_features_suffix
def gender_features_suffix(word):
  return {'suffix1': word[-1:], 'suffix2': word[-2:]}

# examples & class labels
from nltk.corpus import names

labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])

import random

random.shuffle(labeled_names)

# choose features
def choose_features(metric):
  if metric == "gender_features":
    return([(gender_features(n), gender) for (n, gender) in labeled_names])
  if metric == "gender_features_overfit":
    return([(gender_features_overfit(n), gender) for (n, gender) in labeled_names])
  if metric == "gender_features_suffix":
    return([(gender_features_suffix(n), gender) for (n, gender) in labeled_names])
  else:
    print("No Metric Found")

# training & test set
featuresets = choose_features(metric = "gender_features_suffix")
train_set, dev_test_set, test_set = featuresets[6900:], featuresets[6900:7400], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# print accuracy
print(nltk.classify.accuracy(classifier, dev_test_set))

# print 5 most important features
print(classifier.show_most_informative_features(5))
