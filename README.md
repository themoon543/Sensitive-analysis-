# Sensitive-analysis-
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

#Now we create a Roberta Model
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

!pip install torch torchvision torchaudio

MODEL = f"cardiffanlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

#VADER results is here
print(example)
sia.polarity_scores(example)

#run roberta
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
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
        'roberta_pos' : scores[2]
    }
    return scores_dict

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

results_df.columns

sns.pairplot(data=results_df, vars=['vader_neg', 'vader_neu', 'vader_pos','roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='overall',
            palette='tab10')
plt.show()

#the most positive in star 1 review
results_df.query('overall == 1.0') \
    .sort_values('roberta_pos', ascending=False)['reviewText'].values[0]

#the most positive in star 1 review
results_df.query('overall == 1.0') \
    .sort_values('vader_pos', ascending=False)['reviewText'].values[0]

#negative sentiment 5-star view
results_df.query('overall == 5.0') \
    .sort_values('roberta_neg', ascending=False)['reviewText'].values[0]

#the most positive in star 1 review
results_df.query('overall == 5.0') \
    .sort_values('vader_neg', ascending=False)['reviewText'].values[0]

from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis")

sent_pipeline(example)


