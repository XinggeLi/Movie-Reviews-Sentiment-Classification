import itertools
import string
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import nltk
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet

from helper import *


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)




def extract_word(input_string: str) -> list[str]:
    """Preprocess review text into list of tokens.

    Convert input string to lowercase, replace punctuation with spaces, and split along
    whitespace. Return the resulting array.

    Example:
        > extract_word("I love EECS 445. It's my favorite course!")
        > ["i", "love", "eecs", "445", "it", "s", "my", "favorite", "course"]

    Args:
        input_string: text for a single review

    Returns:
        a list of words, extracted and preprocessed according to the directions
        above.
    """
    # TODO: Implement this function
    newstr = input_string.lower()
    for i in range(len(string.punctuation)):
        newstr = newstr.replace(string.punctuation[i], " ")
    return newstr.split()


def extract_dictionary(df: pd.DataFrame) -> dict[str, int]:
    """
    Map words to index.

    Reads a pandas dataframe, and returns a dictionary of distinct words mapping from each
    distinct word to its index (ordered by when it was found).

    Example:
        Input df:

        | reviewText                    | label | ... |
        | It was the best of times.     |  1    | ... |
        | It was the blurst of times.   | -1    | ... |

        The output should be a dictionary of indices ordered by first occurence in
        the entire dataset. The index should be autoincrementing, starting at 0:

        {
            it: 0,
            was: 1,
            the: 2,
            best: 3,
            of: 4,
            times: 5,
            blurst: 6,
        }

    Args:
        df: dataframe/output of load_data()

    Returns:
        a dictionary mapping words to an index
    """
    word_dict = {}

    i = 0
    for row in df['reviewText']:
        wordlist = extract_word(row)
        for word in wordlist:
            if word not in word_dict:
                word_dict[word] = i
                i += 1
    return word_dict


def generate_feature_matrix(
    df: pd.DataFrame, word_dict: dict[str, int]
) -> npt.NDArray[np.float64]:
    """
    Create matrix of feature vectors for dataset.

    Reads a dataframe and the dictionary of unique words to generate a matrix
    of {1, 0} feature vectors for each review. For each review, extract a token
    list and use word_dict to find the index for each token in the token list.
    If the token is in the dictionary, set the corresponding index in the review's
    feature vector to 1. The resulting feature matrix should be of dimension
    (# of reviews, # of words in dictionary).

    Args:
        df: dataframe that has the text and labels
        word_dict: dictionary of words mapping to indices

    Returns:
        a numpy matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
   

    for i, review in enumerate(df["reviewText"]):
        wordlist = extract_word(review)
        for word in wordlist:
            index = word_dict.get(word)
            if index != None:
                feature_matrix[i, index] = 1

    return feature_matrix


def performance(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.int64],
    metric: str = "accuracy",
) -> np.float64:
    """
    Calculate performance metrics.

    Performance metrics are evaluated on the true labels y_true versus the
    predicted labels y_pred.

    Args:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')

    Returns:
        the performance as an np.float64
    """
   
    if metric == 'accuracy':
        return metrics.accuracy_score(y_true, y_pred)
    elif metric == 'f1-score':
        return metrics.f1_score(y_true, y_pred)
    elif metric == 'auroc':
        return metrics.roc_auc_score(y_true, y_pred)
    elif metric == 'precision':
        return metrics.precision_score(y_true, y_pred)
    elif metric == 'sensitivity':
        return metrics.recall_score(y_true, y_pred)
    elif metric == 'specificity':
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)
    else:
        raise ValueError("Invalid metric.")


def cv_performance(
    clf: LinearSVC | SVC,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    k: int = 5,
    metric: str = "accuracy",
) -> float:
    """
    Split data into k folds and run cross-validation.

    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates and returns the k-fold cross-validation performance metric for
    classifier clf by averaging the performance across folds.

    Args:
        clf: an instance of LinearSVC() or SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1, -1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')

    Returns:
        average 'test' performance across the k folds as np.float64
    """
    scores = []
    skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=False)

    for train_index, test_index in skf.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        # train data
        clf.fit(X_train, y_train)
        if metric == "auroc":
            y_pred = clf.decision_function(X_test)
        else:
            y_pred = clf.predict(X_test)
        scores.append(performance(y_test, y_pred, metric))

    return np.array(scores).mean()


def select_param_linear(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    k: int = 5,
    metric: str = "accuracy",
    C_range: list[float] = [],
    loss: str = "hinge",
    penalty: str = "l2",
    dual: bool = True,
) -> float:
    """
    Search for hyperparameters from the given candidates of linear SVM with
    best k-fold CV performance.

    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        loss: string specifying the loss function used (default="hinge",
             other option of "squared_hinge")
        penalty: string specifying the penalty type used (default="l2",
             other option of "l1")
        dual: boolean specifying whether to use the dual formulation of the
             linear SVM (set True for penalty "l2" and False for penalty "l1")

    Returns:
        the parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """

    max_performance = -np.inf
    best_C = None

    for i in C_range:
        clf = LinearSVC(C=i, loss=loss, penalty=penalty, dual=dual, random_state=445)
        current = cv_performance(clf, X, y, k, metric)
        if current > max_performance:
            max_performance = current
            best_C = i
    
    return best_C


def plot_weight(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    penalty: str,
    C_range: list[float],
    loss: str,
    dual: bool,
) -> None:
    """
    Create a plot of the L0 norm learned by a classifier for each C in C_range.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        penalty: string for penalty type to be forwarded to the LinearSVC constructor
        C_range: list of C values to train a classifier on
        loss: string for loss function to be forwarded to the LinearSVC constructor
        dual: whether to solve the dual or primal optimization problem, to be
            forwarded to the LinearSVC constructor

    Returns: None
        Saves a plot of the L0 norms to the filesystem.
    """
    norm0 = []

    for i in C_range:
        clf = LinearSVC(penalty=penalty, C=i, loss=loss, dual=dual)
        clf.fit(X, y)
        norm0.append(np.count_nonzero(clf.coef_))

    plt.plot(C_range, norm0)
    plt.xscale("log")
    plt.legend(["L0-norm"])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title("Norm-" + penalty + "_penalty.png")
    plt.savefig("Norm-" + penalty + "_penalty.png")
    plt.close()


def select_param_quadratic(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    k: int = 5,
    metric: str = "accuracy",
    param_range: npt.NDArray[np.float64] = [],
) -> tuple[float, float]:
    """
    Search for hyperparameters from the given candidates of quadratic SVM
    with best k-fold CV performance.

    Sweeps different settings for the hyperparameters of a quadratic-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
        param_range: a (num_param, 2)-sized array containing the
            parameter values to search over. The first column should
            represent the values for C, and the second column should
            represent the values for r. Each row of this array thus
            represents a pair of parameters to be tried together.

    Returns:
        The parameter values for a quadratic-kernel SVM that maximize
        the average 5-fold CV performance as a pair (C,r)
    """

    best_C_val, best_r_val = 0.0, 0.0
    max_performance = -np.inf

    for (c, r) in param_range:
        clf = SVC(kernel='poly', degree=2, C=c, coef0=r, gamma='auto', random_state=445)
        current = cv_performance(clf, X, y, k, metric)
        if current > max_performance:
            max_performance = current
            best_C_val = c
            best_r_val = r

    return best_C_val, best_r_val



def train_word2vec(filename: str) -> Word2Vec:
    """
    Train a Word2Vec model using the Gensim library.

    First, iterate through all reviews in the dataframe, run your extract_word() function
    on each review, and append the result to the sentences list. Next, instantiate an
    instance of the Word2Vec class, using your sentences list as a parameter and using workers=1.

    Args:
        filename: name of the dataset csv

    Returns:
        created Word2Vec model
    """
    df = load_data(filename)
    sentences = []

    for review in df['reviewText']:
        words = extract_word(review)
        sentences.append(words)
    
    return Word2Vec(sentences, workers=1)

def compute_association(filename: str, w: str, A: list[str], B: list[str]) -> float:
    """
    Args:
        filename: name of the dataset csv
        w: a word represented as a string
        A: set of English words
        B: set of English words

    Returns:
        association between w, A, and B as defined in the spec
    """
    model = train_word2vec(filename)

    def words_to_array(s: list[str]) -> npt.NDArray[np.float64]:
        """Convert a list of string words into a 2D numpy array of word embeddings,
        where the ith row is the embedding vector for the ith word in the input set (0-indexed).

            Args:
                s (list[str]): List of words to convert to word embeddings

            Returns:
                npt.NDArray[np.float64]: Numpy array of word embeddings
        """
        emb = []
        for word in s:
            if word in model.wv:
                emb.append(model.wv[word])
        return np.array(emb)

    def cosine_similarity(
        array: npt.NDArray[np.float64], w: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate the cosine similarities between w and the input set.

        Args:
            array: array representation of the input set
            w: word embedding for w

        Returns:
            1D Numpy Array where the ith element is the cosine similarity between the word
            embedding for w and the ith embedding in input set
        """
        #for emb in array:
            # Compute dot product
        dot_product = np.dot(array, w)
            # Compute magnitudes
        norms = np.linalg.norm(w) * np.linalg.norm(array, axis=1)
            # Calculate cosine similarity
        similarity = dot_product / norms
            # similarities.append(similarity)
        return similarity

    test_arr = np.array([[4, 5, 6], [9, 8, 7]])
    test_w = np.array([1, 2, 3])
    test_sol = np.array([0.97463185, 0.88265899])
    assert np.allclose(
        cosine_similarity(test_arr, test_w), test_sol, atol=0.00000001
    ), "Cosine similarity test 1 failed"

    # Test case 2: Orthogonal vectors
    test_arr = np.array([[1, 0], [0, 1]])
    test_w = np.array([0, 1])
    expected_result = np.array([0, 1])
    assert np.allclose(cosine_similarity(test_arr, test_w), expected_result)

    emb_A = words_to_array(A)
    emb_B = words_to_array(B)
    emb_w = model.wv[w]
    sim_A = cosine_similarity(emb_A, emb_w)
    sim_B = cosine_similarity(emb_B, emb_w)
    return np.mean(sim_A) - np.mean(sim_B)

def select_param_rbf(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    k: int = 5,
    metric: str = "accuracy",
    param_range: npt.NDArray[np.float64] = [],
) -> tuple[float, float]:
    best_C_val, best_r_val = 0.0, 0.0
    max_performance = -np.inf

    for c in param_range:
        svm_clf = SVC(kernel='rbf', C=c, gamma=1)
        clf = OneVsRestClassifier(svm_clf)
        current = cv_performance(clf, X, y, k, metric)
        print(c)
        print(current)
        if current > max_performance:
            max_performance = current
            best_C_val = c

    return best_C_val

def remove_stopwords(words: list[str]) -> list[str]:
    """
    Args:
        words: list of strings

    Returns:
        list of strings with stopwords removed
    """
    stop_words = set(stopwords.words('english'))
    removed_words = [word for word in words if word not in stop_words]
    return removed_words

def mod_extract_dictionary(df: pd.DataFrame) -> dict[str, int]:
    """
    Args:
        df: dataframe/output of load_data()

    Returns:
        a dictionary mapping words to an index
    """
    word_dict = {}

    # Stop words removal
    stop_words = set(stopwords.words('english'))

    i = 0
    for row in df['reviewText']:
        wordlist = extract_word(row)
        wordlist = remove_stopwords(wordlist)
        for word in wordlist:
            if word in word_dict:
                word_dict[word] = i
                i += 1
    return word_dict

def lemmatizer(
    df: pd.DataFrame
) -> list[str]:
    wnl = WordNetLemmatizer()
    lematized_doc = []
    for review in df["reviewText"]:
        words = extract_word(review)
        words = [wnl.lemmatize(word, pos=wordnet.VERB) for word in words]
        sentence = ' '.join(words)
        lematized_doc.append(sentence)
    return lematized_doc

def convert_time(t: str) -> list[np.float64]:
    t = t.replace(",", "")
    date = t.split(" ")
    time = []
    time.append(int(date[0]) / 100)
    time.append(int(date[1]) / 100)
    time.append((int(date[-1])) / 100)
    time = time
    return time

def mod_generate_feature_matrix(
    df: pd.DataFrame, vctrizer: TfidfVectorizer,
) -> npt.NDArray[np.float64]:
    
    # number_of_reviews = df.shape[0]
    # number_of_words = len(word_dict)
    # feature_matrix = np.zeros((number_of_reviews, number_of_words))
    
    # stop_words = set(stopwords.words('english'))

    # for i, review in enumerate(df['reviewText']):
    #     wordlist = extract_word(review)

    #     # Stop words removal
    #     wordlist = [word for word in wordlist if word not in stop_words]

    #     for word in wordlist:
    #         index = word_dict.get(word)
    #         if index != None:
    #             feature_matrix[i, index] = 1

    lemm_words = lemmatizer(df)
    sparse_matrix = (vctrizer.transform(lemm_words))
    words_matrix = sparse_matrix.toarray()
    reviewtime = np.zeros((words_matrix.shape[0], 3))
    idx = 0
    for time in df["reviewTime"]:
        reviewtime[idx] = convert_time(time)
        idx = idx + 1
    feature_matrix = np.hstack((words_matrix, reviewtime))
    return feature_matrix

def main() -> None:
    filename = "data/dataset.csv"


    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data(
        filename=filename
    )
    IMB_features, IMB_labels, IMB_test_features, IMB_test_labels = get_imbalanced_data(
        dictionary_binary, filename=filename
    )
    
    """
    # 2a
    print(extract_word("It's a test sentence! Does it look CORRECT?"))

    # 2b
    print(len(dictionary_binary))

    # 2c
    mean = np.sum(X_train) / X_train.shape[0]
    print(mean)
    idx = np.argmax(np.sum(X_train, axis=0))
    max_word = list(dictionary_binary.keys())[idx]
    print(max_word)
    
    # 3.1b
    listC = [1000.0, 100.0, 10.0, 1.0, 0.1, 0.01, 0.001]
    metrics = ["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]
    for metric in metrics:
        print(metric)
        print(select_param_linear(X_train, Y_train, 5, metric, listC, "hinge", "l2", True))
    
    # 3.1c
    clf = LinearSVC(C=0.1, loss="hinge", penalty="l2", dual=True, random_state=445)
    clf.fit(X_train, Y_train)
    for metric in metrics:
        scores = []
        if metric == "auroc":
            pred = clf.decision_function(X_test)
        else:
            pred = clf.predict(X_test)
        scores.append(performance(Y_test, pred, metric))
        print(np.array(scores).mean())
    
    # 3.1d
    plot_weight(X_train, Y_train, "l2", [0.001, 0.01, 0.1, 1.0], "hinge", True)
    
    # 3.1e
    clf = LinearSVC(C=0.1, penalty='l2', loss='hinge', dual=True, random_state=445)
    clf.fit(X_train, Y_train)

    # Get the coefficients and corresponding words
    coefficients = clf.coef_[0] 
    word_indices = np.argsort(coefficients) 

    # Select the five most positive and five most negative coefficients
    most_positive_indices = word_indices[-5:][::-1] 
    most_negative_indices = word_indices[:5]

    # Get corresponding words
    most_positive_words = [list(dictionary_binary.keys())[i] for i in most_positive_indices]
    most_negative_words = [list(dictionary_binary.keys())[i] for i in most_negative_indices]

    # Get corresponding coefficients
    most_positive_coefficients = [coefficients[i] for i in most_positive_indices]
    most_positive_coefficients.reverse()
    most_negative_coefficients = [coefficients[i] for i in most_negative_indices]

    # Plot a bar chart
    plt.figure(figsize=(10, 7))
    plt.bar(most_negative_words, most_negative_coefficients, color='red', label='negative')
    plt.bar(most_positive_words, most_positive_coefficients, color='blue', label='positive')
    plt.ylabel('Coefficient Value')
    plt.xlabel('Word')
    plt.legend()
    plt.title('Top 5 Most Positive and Most Negative Words')
    plt.savefig('Top 5 Most Positive and Most Negative Words')
    plt.close()
    
    # 3.2a
    listC = [0.001, 0.01, 0.1, 1.0]
    print(select_param_linear(X_train, Y_train, 5, 'auroc', listC, "squared_hinge", "l1", False))
    clf = LinearSVC(penalty='l1', loss = 'squared_hinge', C=0.1, dual=False, random_state=445)
    clf.fit(X_train, Y_train)
    pred = clf.decision_function(X_test)
    print(performance(Y_test, pred, 'auroc'))

    # 3.2b
    plot_weight(X_train, Y_train, 'l1', listC, 'squared_hinge', False)
       
    # 3.3a i)
    values = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    grid_search = np.array([(c, r) for c in values for r in values])
    result = select_param_quadratic(X_train, Y_train, 5, 'auroc', grid_search)
    print(result)
    
    clf = SVC(kernel='poly', degree=2, C=result[0], coef0=result[1], gamma='auto', random_state=445)
    clf.fit(X_train, Y_train)
    pred = clf.decision_function(X_test)
    print(performance(Y_test, pred, 'auroc'))
    
    # 3.3a ii)
    ran_c = np.random.uniform(-2, 3, size=25)
    ran_r = np.random.uniform(-2, 3, size=25)
    ran_c = 10 ** ran_c
    ran_r = 10 ** ran_r
    ran_search = list(zip(ran_c, ran_r))
    result = select_param_quadratic(X_train, Y_train, 5, 'auroc', ran_search)
    print(result)

    clf = SVC(kernel='poly', degree=2, C=result[0], coef0=result[1], gamma='auto', random_state=445)
    clf.fit(X_train, Y_train)
    pred = clf.decision_function(X_test)
    print(performance(Y_test, pred, 'auroc'))

    # 4.1c)
    clf = LinearSVC(C=0.01, penalty='l2', loss='hinge', dual=True, class_weight={-1: 1, 1: 10}, random_state=445)
    clf.fit(X_train, Y_train)
    pred_1 = clf.predict(X_test)
    pred_2 = clf.decision_function(X_test)
    metrics = ["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]
    for metric in metrics:
        if metric == 'auroc':
            print(performance(Y_test, pred_2, metric))
        else:
            print(performance(Y_test, pred_1, metric))
    
    # 4.2a)
    clf = LinearSVC(C=0.01, penalty='l2', loss='hinge', dual=True, class_weight={-1: 1, 1: 1}, random_state=445)
    clf.fit(IMB_features, IMB_labels)
    pred_1 = clf.predict(IMB_test_features)
    pred_2 = clf.decision_function(IMB_test_features)
    metrics = ["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]
    for metric in metrics:
        if metric == 'auroc':
            print(performance(IMB_test_labels, pred_2, metric))
        else:
            print(performance(IMB_test_labels, pred_1, metric))
    
    # 4.3a) 
    ran_1 = np.arange(1, 20)
    ran_2 = np.arange(1, 10)
    ran = np.array([(c, r) for c in ran_1 for r in ran_2])
    #ran = ran / 10
    max_n = 0
    max_p = 0
    max_performance = -np.inf
    for (n, p) in ran:
        clf = LinearSVC(C=0.01, penalty='l2', loss='hinge', dual=True, class_weight={-1: n, 1: p}, random_state=445)
        current = cv_performance(clf, IMB_features, IMB_labels, 5, 'auroc')
        if current > max_performance:
            max_performance = current
            max_n = n
            max_p = p
    print(max_n, max_p)
    print(max_performance)
    
    # 4.3b)
    clf = LinearSVC(C=0.01, penalty='l2', loss='hinge', class_weight={-1: 9, 1: 6}, random_state=445)
    clf.fit(IMB_features, IMB_labels)
    pred_1 = clf.predict(IMB_test_features)
    pred_2 = clf.decision_function(IMB_test_features)
    metriclist = ["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]
    for metric in metriclist:
        if metric == 'auroc':
            print(performance(IMB_test_labels, pred_2, metric))
        else:
            print(performance(IMB_test_labels, pred_1, metric))

    # 4.4b
    unweighted_clf = LinearSVC(C=0.01, penalty='l2', loss='hinge', dual=True, class_weight={-1: 1, 1: 1}, random_state=445)
    weighted_clf = LinearSVC(C=0.01, penalty='l2', loss='hinge', dual=True, class_weight={-1: 9, 1: 6}, random_state=445)
    unweighted_clf.fit(IMB_features, IMB_labels)
    weighted_clf.fit(IMB_features, IMB_labels)

    fpr_unweighted, tpr_unweighted, thre = metrics.roc_curve(IMB_test_labels, unweighted_clf.decision_function(IMB_test_features))
    roc_auc_unweighted = metrics.auc(fpr_unweighted, tpr_unweighted)

    fpr_custom, tpr_custom, thre = metrics.roc_curve(IMB_test_labels, weighted_clf.decision_function(IMB_test_features))
    roc_auc_weighted = metrics.auc(fpr_custom, tpr_custom)

    # Plot the ROC curves
    plt.figure()
    plt.plot(fpr_unweighted, tpr_unweighted, color='blue', lw=2, label=f'Unweighted (AUC = {roc_auc_unweighted:.2f})')
    plt.plot(fpr_custom, tpr_custom, color='red', lw=2, label=f'Weighted (Wn=9, Wp=6)(AUC = {roc_auc_weighted:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.savefig('ROC Curve Comparison')
    plt.close()
    
    # 5.1a)
    print(count_actors_and_actresses(filename))

    # 5.1b)
    plot_actors_and_actresses(filename, 'label')
    
    # 5.1c)
    plot_actors_and_actresses(filename, 'rating')
    
    # 5.1d)
    clf = LinearSVC(C=0.1, penalty='l2', loss='hinge', random_state=445)
    clf.fit(X_train, Y_train)
    coefs = clf.coef_[0]
    coef_dict = dict(zip(dictionary_binary, coefs))
    print(coef_dict.get('actor'))
    print(coef_dict.get('actress'))
    
    # 5.2a)
    w2v = train_word2vec(filename)
    print(w2v.wv['actor'])
    
    # 5.2b)
    sim_words = w2v.wv.most_similar('plot', topn=5)
    for word, similarity in sim_words:
        print(f"- {word}: {similarity}")
    
    # 5.3a)
    A = ['her', 'woman', 'women']
    B = ['him', 'man', 'men']
    print(compute_association(filename, 'smart', A, B))
    """

    (
        multiclass_features,
        multiclass_labels,
        multiclass_vctrizer,
    ) = get_multiclass_training_data()

    # Get test set from dataset.csv
    mx_train, my_train, mx_test, my_test = get_split_multi_data(
        filename=filename, n=750
    )
    
    heldout_features = get_heldout_reviews(multiclass_vctrizer)

    # Hyperparameter selection
    c_range = [10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2]
    best_c = select_param_rbf(multiclass_features, multiclass_labels, metric="accuracy", k=5, param_range=c_range)
    c_range = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    best_c = select_param_rbf(multiclass_features, multiclass_labels, metric="accuracy", k=5, param_range=c_range)
    print(best_c)
  
    # Select SVC with parameter that maximizes accuracy
    svm_clf = SVC(kernel='rbf', C=1, gamma=1)
    #svm_clf = LinearSVC(C=0.1, loss='hinge', dual=True)
    
    # Wrap the classifier in the OneVsAllClassifier and OneVsOneClassifier
    ova_clf = OneVsRestClassifier(svm_clf)
    ovo_clf = OneVsOneClassifier(svm_clf)
    
    # Train both classifiers and run them in parallel
    ova_clf.fit(multiclass_features, multiclass_labels)
    ovo_clf.fit(multiclass_features, multiclass_labels)

    pred_my1 = ova_clf.predict(mx_test)
    pred_my2 = ovo_clf.predict(mx_test)

    print(performance(my_test, pred_my1, 'accuracy'))
    print(performance(my_test, pred_my2, 'accuracy'))

    # Make predictions
    predictions = ova_clf.predict(heldout_features)

    # Save the predictions
    generate_challenge_labels(predictions, 'lixingge')
    
    
if __name__ == "__main__":
    main()
