import pandas as pd
import csv

lexicon_positive = dict()
lexicon_negative = dict()

try:
    with open('positive.tsv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)
        for row in reader:
            lexicon_positive[row[0]] = int(row[1])
except FileNotFoundError:
    print("Positive file not found")

try:
    with open('negative.tsv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)
        for row in reader:
            lexicon_negative[row[0]] = int(row[1])
except FileNotFoundError:
    print("Negative file not found")


def emotion_analyst(text):
    emotion_score = 0
    for word in text:
        if (word in lexicon_positive):
            emotion_score = emotion_score + lexicon_positive[word]
    for word in text:
        if (word in lexicon_negative):
            emotion_score = emotion_score + lexicon_negative[word]
    emotion = ""
    if (emotion_score > 0):
        emotion = "happy"
    elif (emotion_score < 0):
        if (emotion_score < -20):
            emotion = "fear"
        elif (emotion_score < -10):
            emotion = "angry"
        else:
            emotion = "sad"
    else:
        emotion = "undefined"
    return emotion_score, emotion



dataset = pd.read_csv('dataset.csv')
dataset['text_preprocessed'] = dataset['text_preprocessed'].astype(str)

for i, text in enumerate(dataset['text_preprocessed']):
    dataset.at[i, 'text_preprocessed'] = text.replace("'", "")\
        .replace(',', '').replace(']', '').replace('[', '').split()


results = dataset['text_preprocessed'].apply(emotion_analyst)
results = list(zip(*results))
dataset['emotion_score'] = results[0]
dataset['emotion'] = results[1]

dataset.to_csv(r'emotion.csv', index=False, header=True, index_label=None)
