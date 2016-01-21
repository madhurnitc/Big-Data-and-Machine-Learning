import csv
import StringIO
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes

# Initialize necessary global variables
conf = None
sc = None
PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()


# Initialize Apache Spark Context

def initializeSpark():
	global conf, sc
	conf = SparkConf().setAppName("Buffalo 311 Data Classification")
	sc = SparkContext(conf = conf)

# Function to read the data from the CSV file.

def readData():
	inputFile = "311data_cleaned_Mycleaned.csv"
	input = sc.textFile(inputFile).map(lambda line: line.split(",")).filter(lambda line: len(line)>1).map(lambda line: (line[0], 		int(line[1])))
	return input


# Function to break text into "tokens", lowercase them, remove punctuation and stopwords, and stem them

def tokenize(text):
    tokens = word_tokenize(text)
    lowercased = [t.lower() for t in tokens]
    no_punctuation = []
    for word in lowercased:
        punct_removed = ''.join([letter for letter in word if not letter in PUNCTUATION])
        no_punctuation.append(punct_removed)
    no_stopwords = [w for w in no_punctuation if not w in STOPWORDS]
    stemmed = [STEMMER.stem(w) for w in no_stopwords]
    return [w for w in stemmed if w]

# Programs start from here 

initializeSpark()

# Read the data 
data = readData()

# Prepare text for analysis using our tokenize function to clean it up

data_cleaned = data.map(lambda (text, label): (tokenize(text),float(label)))

# Hashing term frequency vectorizer with  features

htf = HashingTF(25000)

# Create an RDD of LabeledPoints using category labels as labels and tokenized, hashed text as feature vectors

data_hashed = data_cleaned.map(lambda (text, label): LabeledPoint(float(label), htf.transform(text)))

# Ask Spark to persist the RDD so it won't have to be re-created later

data_hashed.persist()

# Split data 70/30 into training and test data sets

train_hashed, test_hashed = data_hashed.randomSplit([0.7, 0.3])

# Train a Naive Bayes model on the training data

model = NaiveBayes.train(train_hashed)

# Compare predicted labels to actual labels

prediction_and_labels = test_hashed.map(lambda point: (model.predict(point.features), point.label))

# Filter to only correct predictions

correct = prediction_and_labels.filter(lambda (predicted, actual): predicted == actual)

# Calculate and print accuracy rate

accuracy = correct.count() / float(test_hashed.count())
print "Classifier correctly predicted category of the complaints " + str(accuracy * 100) + " percent of the time"



