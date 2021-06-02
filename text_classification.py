import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import re
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
def preprocess(data):
    data_list = []
    stemmer = WordNetLemmatizer()
    for sen in range(0, len(data)):
        # Remove all the special characters
        data_doc = re.sub(r'\W', ' ', str(data[sen]))

        # remove HTML tags
        data_doc = re.sub(r'<.*?>', '', data_doc)

        # remove all single characters
        data_doc = re.sub(r'\s+[a-zA-Z]\s+', ' ', data_doc)

        # Remove single characters from the start
        data_doc = re.sub(r'\^[a-zA-Z]\s+', ' ', data_doc)

        # Substituting multiple spaces with single space
        data_doc = re.sub(r'\s+', ' ', data_doc, flags=re.I)

        # Removing prefixed 'b'
        data_doc = re.sub(r'^b\s+', '', data_doc)
        # Converting to Lowercase
        data_doc = data_doc.lower()

        # replace punctuation characters with spaces
        punc = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        punc_dict = dict((types, " ") for types in punc)
        punc_map = str.maketrans(punc_dict)
        data_doc = data_doc.translate(punc_map)

        # Lemmatization
        data_doc = data_doc.split()

        data_doc = [stemmer.lemmatize(word) for word in data_doc]
        data_doc = ' '.join(data_doc)

        data_list.append(data_doc)

    return data_list


def imdb_data(data_dir):
    features = {}
    for datasplit in ["train", "test"]:
        features[datasplit] = []
        for sentiment in ["neg", "pos"]:
            score = 1 if sentiment == "pos" else 0

            path = os.path.join(data_dir, datasplit, sentiment)
            file_names = os.listdir(path)
            for filename in file_names:
                with open(os.path.join(path, filename), "r", encoding="utf-8") as file:
                    review = file.read()
                    features[datasplit].append([review, score])

    np.random.shuffle(features["train"])
    features["train"] = pd.DataFrame(features["train"], columns=['text', 'sentiment'])

    np.random.shuffle(features["test"])
    features["test"] = pd.DataFrame(features["test"], columns=['text', 'sentiment'])

    return features["train"], features["test"]

def fun():
    train_data, test_data = imdb_data(data_dir="aclImdb/")

    print(train_data.head())
    print(test_data.head())
    train_data_list = preprocess(train_data['text'])
    test_data_list = preprocess(test_data['text'])

    # Vectorization
    tfidfvectorizer = TfidfVectorizer()
    training_data_text = tfidfvectorizer.fit_transform(train_data_list)
    testing_data_text = tfidfvectorizer.transform(test_data_list)

    '''Model Training and Testing'''
    svcmodel = SVC(
    kernel = 'rbf',
    C=1,
    gamma = 0.1,
    decision_function_shape='ovr',
    random_state = 0,
    )
    svcmodel.fit(training_data_text, train_data['sentiment'])
    y_pred = svcmodel.predict(testing_data_text)

    print(confusion_matrix(test_data['sentiment'], y_pred))
    conf_matrix = confusion_matrix(test_data['sentiment'], y_pred)
    print("report", classification_report(test_data['sentiment'], y_pred))
    print("Accuracy:", accuracy_score(test_data['sentiment'], y_pred))
    acc_score = accuracy_score(test_data['sentiment'], y_pred)
    plt.figure(figsize=(9, 9))
    sns.heatmap(conf_matrix, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(acc_score)
    plt.title(all_sample_title, size=15)
    plt.show()

    # Random forest Classifier
    randomforestclf = RandomForestClassifier(n_estimators=100)

    # Train the model
    randomforestclf.fit(training_data_text, train_data['sentiment'])

    y_pred = randomforestclf.predict(testing_data_text)
    # forest metrics
    print(confusion_matrix(test_data['sentiment'], y_pred))
    conf_matrix = confusion_matrix(test_data['sentiment'], y_pred)
    print("report", classification_report(test_data['sentiment'], y_pred))
    print("Accuracy:", accuracy_score(test_data['sentiment'], y_pred))
    acc_score = accuracy_score(test_data['sentiment'], y_pred)
    plt.figure(figsize=(9, 9))
    sns.heatmap(conf_matrix, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(acc_score)
    plt.title(all_sample_title, size=15)
    plt.show()



if __name__ == "__main__":
    fun()
