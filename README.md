import numpy as np 
import pandas as pd

import os
for dirname, _, filenames in os.walk(r"C:\Users\Aizhan\Downloads\amazon_reviews.csv\amazon_reviews.csv"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk

Cleaning data


df = pd.read_csv(r"C:\Users\Aizhan\Downloads\amazon_reviews.csv\amazon_reviews.csv")
df = df.tail(1000)

list(df)

df.rename(columns = {'Unnamed: 0':'Id'}, inplace = True)

df.tail()

df.info()

#checking shape of data
print(df.shape)

#create bar plot from data
ax = df['overall'].value_counts().sort_index() \
    .plot(kind='bar', 
          title='Count of Reviews by Stars',
          figsize=(10, 5))

ax.set_xlabel('Review Stars')
plt.show()

#Making NLP from 1 data
example = df['reviewText'][4900]
print(example)

#tokenize the review
nltk.download('punkt')
tokens = nltk.word_tokenize(example)
tokens[:10]

#search for pos
nltk.download('averaged_perceptron_tagger')
tagged = nltk.pos_tag(tokens)
tagged[:10]

#put tagged review into entities
nltk.download('words')
nltk.download('maxent_ne_chunker')
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()

Now we will use VADER MODEL

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

sia

#implement sia in example
sia.polarity_scores(example)

#run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['reviewText']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

#showing results
res

res_df = pd.DataFrame.from_dict(res, orient='index')

print(res_df)

#saving vader results
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')

#we have sentiment score and metadata
vaders.head()

#plot VADER Results
ax = sns.barplot(data=vaders, x='overall', y='compound')
ax.set_title('Compund Score by Amazon Star Review')
plt.show()

#visualizing results
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='overall', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='overall', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='overall', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()

# Roberta Model

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "distilroberta-base"  # Replace with a valid model name
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


scores_dict = {
    'roberta_class_1' : scores[0],
    'roberta_class_2' : scores[1]
}


from transformers import RobertaTokenizer, RobertaForSequenceClassification
from scipy.special import softmax

# Load the tokenizer and model from Hugging Face
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')

# Assuming 'example' is your text string that you want to analyze
# example = "Your text here."

# Encoding the text using the tokenizer
encoded_text = tokenizer(example, return_tensors='pt')

# Getting the model output
output = model(**encoded_text)

# Process the output to get scores
scores = output[0][0].detach().numpy()
scores = softmax(scores)

# Adjusting the dictionary creation based on the number of output classes
# Assuming it's a binary classification model
scores_dict = {
    'roberta_class_1' : scores[0],
    'roberta_class_2' : scores[1]
}

print(scores_dict)




#run roberta

encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[0]
}
print(scores_dict)

#run on entire dataset
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[0]
    }
    return scores_dict

pip install torch torchvision torchaudio


pip install transformers


from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import pipeline


#from custom_sentiment_analysis import polarity_scores_roberta


res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['reviewText']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')

#visualize result dic
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

results_df.head()

# Compare Scores Between Two Models

results_df.columns

sns.pairplot(data=results_df, 
             vars=['neg', 'neu', 'pos', 'compound'],
             hue='overall',
             palette='tab10')
plt.show()


# Review

# Finding the most positive review in 1-star reviews based on 'pos' column
most_positive_review = results_df.query('overall == 1.0') \
                                 .sort_values('pos', ascending=False)['reviewText'].values[0]

print(most_positive_review)


#the most positive in star 1 review
results_df.query('overall == 1.0') \
    .sort_values('vader_pos', ascending=False)['reviewText'].values[0]

# Finding the most negative sentiment review in 5-star reviews based on 'neg' column
most_negative_review_5_star = results_df.query('overall == 5.0') \
                                        .sort_values('neg', ascending=False)['reviewText'].values[0]

print(most_negative_review_5_star)


# huggingface transformers pipeline

from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis")

sent_pipeline(example)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Assuming df is your DataFrame

# Define sentiment labels based on 'overall' rating
def define_sentiment(rating):
    if rating >= 4: return 'positive'
    elif rating <= 2: return 'negative'
    else: return 'neutral'

df['sentiment'] = df['overall'].apply(define_sentiment)

# Preprocess data
def preprocess(text):
    tokens = word_tokenize(str(text).lower())
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

df['processed_reviews'] = df['reviewText'].apply(preprocess)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['processed_reviews'])
y = df['sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predicting new comments
def predict_sentiment(comment):
    processed = preprocess(comment)
    vectorized = vectorizer.transform([processed])
    return model.predict(vectorized)

# Example usage
print(predict_sentiment("This product is a bad"))


from transformers import pipeline

# Load a pre-trained sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')

# Function to predict sentiment
def predict_sentiment(comment):
    return sentiment_pipeline(comment)

# Example usage
print(predict_sentiment('Good one'))


