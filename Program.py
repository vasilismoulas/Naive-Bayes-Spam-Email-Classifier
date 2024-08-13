from Naive_Bayes import*
from NLT import*
from tqdm import tqdm


def main():
   accuracy = naive_bayes()
   print("Accuracy: ", accuracy)

if __name__=='__main__':
    main()