import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load dataset
dataset = pd.read_csv('dataset.csv')
dataset['text_preprocessed'] = dataset['text_preprocessed'].astype(str)

# Preprocess text
for i, text in enumerate(dataset['text_preprocessed']):
    dataset.at[i, 'text_preprocessed'] = text.replace("'", "")\
        .replace(',', '').replace(']', '').replace('[', '').split()

# Select subset of data to visualize
visualize = dataset.head(10)

# Generate word list
words = [word for row in visualize['text_preprocessed'] for word in row]

# Create word cloud
wordcloud = WordCloud(width=500, height=400, background_color='white',
                      min_font_size=10).generate(" ".join(words))

# Plot and save word cloud
fig, ax = plt.subplots(figsize=(30, 10))
ax.set_title('Most Common Words', fontsize=15)
ax.grid(False)
ax.imshow(wordcloud)
ax.axis('off')
fig.tight_layout(pad=0)
plt.savefig('wordcloud.png')

matplotlib.use('Agg')

# Create a bar chart from the value counts
ax = dataset['polarity'].value_counts().plot(kind='bar', figsize=(
    12, 8), title='Frequency Sentiment', xlabel='Sentiment', ylabel='Frequency', legend=False)

# Display the figure to verify that it looks correct
plt.show()

# Save the figure to a file
fig = ax.get_figure()
fig.savefig('sentiment_frequency.png')

# TF IDF
corpus = dataset['toSantance']

# Remove NaN values from the corpus
corpus = corpus.dropna()

# Create the TfidfVectorizer object
vectorizer = TfidfVectorizer(analyzer='word', max_features=2000)

# Fit the vectorizer to the corpus and transform the corpus
tfidf_matrix = vectorizer.fit_transform(corpus)

# add padding to the matrix
tfidf_matrix = tfidf_matrix.toarray()

# Create a DataFrame from the matrix
tfidf_df = pd.DataFrame(
    tfidf_matrix, columns=vectorizer.get_feature_names_out())

# Create a bar chart from the value counts
barchart = tfidf_df.sum().sort_values(ascending=False).head(10).plot(kind='bar', figsize=(
    12, 8), title='Most Common Words', xlabel='Words', ylabel='Frequency', legend=False)

# save figure
fig = barchart.get_figure()
fig.savefig('most_common_words.png')

# Create Test and Train data
polarity_encode = {'negative': 0, 'positive': 1}
x = dataset['toSantance'].values
y = dataset['polarity'].map(polarity_encode).values

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
