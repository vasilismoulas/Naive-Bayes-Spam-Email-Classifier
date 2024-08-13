import pandas as pd
import os
import re
from NLT.text_preprocess import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from math import log
from tqdm import tqdm

def load_emails_from_directory(directories: list = ["3/dataset/enron1/ham", "3/dataset/enron1/spam"]) -> pd.DataFrame:
    emails = []
    labels = []
    for directory in directories:
        label = directory.split('/')[3]
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                with open(os.path.join(directory, filename), 'r', encoding='utf-8', errors='ignore') as file:
                    emails.append(file.read())
                    labels.append(label)
    data = pd.DataFrame({'email_content': emails, 'label_cl': labels})
    return data

def preprocess_emails() -> pd.DataFrame:
    emails = load_emails_from_directory()
    train_set, test_set = train_test_dataset_split(emails)
    
    train_set['email_content'] = train_set['email_content'].str.lower()
    test_set['email_content'] = test_set['email_content'].str.lower()
    
    train_set['email_content'] = train_set['email_content'].apply(tokenizer)
    test_set['email_content'] = test_set['email_content'].apply(tokenizer)

    train_set['email_content'] = train_set['email_content'].apply(remove_punctuation)
    test_set['email_content'] = test_set['email_content'].apply(remove_punctuation)

    train_set['email_content'] = train_set['email_content'].apply(lambda x: ' '.join(x))
    test_set['email_content'] = test_set['email_content'].apply(lambda x: ' '.join(x))
    
    return train_set, test_set

def train_test_dataset_split(emails: pd.DataFrame) -> pd.DataFrame:
    train_set, test_set = train_test_split(emails, test_size=0.2, train_size=0.8)
    return train_set, test_set

def classify(message: str, parameters_spam: dict, parameters_ham: dict, p_spam: float, p_ham: float) -> str:
    message = re.sub('\W', ' ', message)
    message = message.lower().split()

    p_spam_given_message = p_spam
    p_ham_given_message = p_ham

    for word in message:
        if word in parameters_spam:
            p_spam_given_message *= parameters_spam[word]
        if word in parameters_ham:
            p_ham_given_message *= parameters_ham[word]

    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    elif p_ham_given_message < p_spam_given_message:
        return 'spam'
    else:
        return 'equal'

def classify_prevent_underflow(message: str, parameters_spam: dict, parameters_ham: dict, p_spam: float, p_ham: float) -> str:
    message = re.sub('\W', ' ', message)
    message = message.lower().split()

    p_spam_given_message = log(p_spam)
    p_ham_given_message = log(p_ham)

    for word in message:
        if word in parameters_spam:
            p_spam_given_message += log(parameters_spam[word])
        if word in parameters_ham:
            p_ham_given_message += log(parameters_ham[word])

    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    elif p_ham_given_message < p_spam_given_message:
        return 'spam'
    else:
        return 'equal'

def naive_bayes() -> float:
    train_set, test_set = preprocess_emails()

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_set['email_content'])

    word_counts = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    training_set_clean = pd.concat([train_set.reset_index(drop=True), word_counts.reset_index(drop=True)], axis=1)

    spam_messages = training_set_clean.loc[training_set_clean['label_cl'] == 'spam']
    ham_messages = training_set_clean.loc[training_set_clean['label_cl'] == 'ham']

    p_spam = len(spam_messages) / len(training_set_clean)
    p_ham = len(ham_messages) / len(training_set_clean)

    n_words_per_spam_message = spam_messages['email_content'].apply(len)
    n_spam = n_words_per_spam_message.sum()

    n_words_per_ham_message = ham_messages['email_content'].apply(len)
    n_ham = n_words_per_ham_message.sum()

    vocabulary = vectorizer.vocabulary_
    n_vocabulary = len(vocabulary)

    alpha = 1

    parameters_spam = {unique_word: 0 for unique_word in vocabulary}
    parameters_ham = {unique_word: 0 for unique_word in vocabulary}

    for word in tqdm(vocabulary.keys(), desc="Calculating parameters"):
        n_word_given_spam = spam_messages[word].sum()
        p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + (alpha + 1) * n_vocabulary)
        parameters_spam[word] = p_word_given_spam

        n_word_given_ham = ham_messages[word].sum()
        p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + (alpha + 1) * n_vocabulary)
        parameters_ham[word] = p_word_given_ham

    acc = 0
    for index in tqdm(test_set.index, desc="Testing Naive Bayes Algorithm"):
        results = classify_prevent_underflow(test_set['email_content'][index], parameters_spam, parameters_ham, p_spam, p_ham)
        if test_set['label_cl'][index] == results:
            acc += 1

    accuracy = acc / len(test_set)
    return accuracy