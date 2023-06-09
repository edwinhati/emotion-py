from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Load dataset
dataset = pd.read_csv('dataset.csv')

# Preprocess text
dataset['text_preprocessed'] = dataset['text_preprocessed'].astype(str)
for i, text in enumerate(dataset['text_preprocessed']):
    dataset.at[i, 'text_preprocessed'] = text.replace("'", "")\
        .replace(',', '').replace(']', '').replace('[', '').split()

# Encode polarity as numeric labels
polarity_encode = {'negative': 0, 'positive': 1}
dataset['polarity'] = dataset['polarity'].map(polarity_encode)

# Drop NaN values
dataset = dataset.dropna()

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    dataset['toSantance'], dataset['polarity'], test_size=0.2, random_state=0)

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(analyzer='word', max_features=2000)
tfidf_matrix_train = vectorizer.fit_transform(X_train)
tfidf_matrix_test = vectorizer.transform(X_test)

# Train decision tree model
clf = DecisionTreeClassifier(random_state=0)
clf.fit(tfidf_matrix_train, y_train)

# Evaluate model on test set
accuracy = clf.score(tfidf_matrix_test, y_test)
print('Decision tree accuracy:', accuracy)
