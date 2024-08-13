import unittest
import sys
import os
import pandas as pd

#importing parent direcotry
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Naive_Bayes import*

class Testing(unittest.TestCase):
    '''
    The basic class that inherits unittest.TestCase for testing Naive Bayes Algorithm.
    '''
    def test_load_emails_from_directory(directories: list = ["ham","spam"]):
        emails = load_emails_from_directory()
        print(emails)

        print("\ntest_load_emails_from_directory\n")
    
    def test_train_test_dataset_split(emails: pd.DataFrame):
        emails = load_emails_from_directory()
        train, test = train_test_dataset_split(emails)
        print(train.shape,'\t',test.shape)

        print('\n',)

        print("\ntest_train_test_dataset_split\n")

    def test_naive_bayes(self):
        print("\nSTART---test_naive_bayes\n")
        accuracy = naive_bayes()
        print("Accuracy: ", accuracy)

        print("\nEND---test_naive_bayes\n")

if __name__ == '__main__':
    unittest.main()