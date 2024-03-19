"""
EECS 445 - Winter 2024

Utility functions to help load and manipulate data.

Do not edit these functions for sections 2-5 of the project.
"""


import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
import project1

from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(filename: str) -> pd.DataFrame:
    """
    Read in a csv file and return a dataframe.

    You can access the labels by calling df["label"], the reviews by df["reviewText"], and the
    ratings by calling df["rating"].

    Args:
        filename: path to CSV file to load into a dataframe

    Returns:
        dataframe containing the data loaded from the CSV file
    """
    return pd.read_csv(filename)


def get_split_binary_data(
    filename: str = "data/dataset.csv",
    n: int = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    dict[str, int],
]:
    """
    Read in data from filename and return it using extract_dictionary and
    generate_feature_matrix. Dataset gets split into training and test sets.

    The binary labels take two values:
        -1: negative
         1: positive

    Args:
        filename: name of the file to be read from
        n: number of samples to use for training. If None, use all samples

    Returns:
        X_train: training feature matrix
        Y_train: training labels
        X_test: test feature matrix
        Y_test: test labels
        dictionary: dictionary used to create feature matrices
    """
    dataframe = load_data(filename)
    dataframe = dataframe[dataframe["label"] != 0]
    positiveDF = dataframe[dataframe["label"] == 1].copy()
    negativeDF = dataframe[dataframe["label"] == -1].copy()

    if n is not None:
        class_size = n
    else:
        class_size = 2 * positiveDF.shape[0] // 3

    X_train = (
        pd.concat([positiveDF[:class_size], negativeDF[:class_size]])
        .reset_index(drop=True)
        .copy()
    )
    dictionary = project1.extract_dictionary(X_train)
    X_test = (
        pd.concat([positiveDF[class_size:], negativeDF[class_size:]])
        .reset_index(drop=True)
        .copy()
    )
    Y_train = X_train["label"].values.copy()
    Y_test = X_test["label"].values.copy()
    X_train = project1.generate_feature_matrix(X_train, dictionary)
    X_test = project1.generate_feature_matrix(X_test, dictionary)

    return (X_train, Y_train, X_test, Y_test, dictionary)

def get_split_multi_data(
    filename: str = "data/dataset.csv",
    n: int = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    dict[str, int],
]:
    dataframe = load_data(filename)
    zeroDF = dataframe[dataframe["label"] == 0].copy()
    positiveDF = dataframe[dataframe["label"] == 1].copy()
    negativeDF = dataframe[dataframe["label"] == -1].copy()

    if n is not None:
        class_size = n
    else:
        class_size = 2 * positiveDF.shape[0] // 3

    X_train = (
        pd.concat([positiveDF[:class_size], negativeDF[:class_size], zeroDF[:class_size]])
        .reset_index(drop=True)
        .copy()
    )
    X_test = (
        pd.concat([positiveDF[class_size:], negativeDF[class_size:], zeroDF[class_size:]])
        .reset_index(drop=True)
        .copy()
    ) 
    Y_train = X_train["label"].values.copy()
    Y_test = X_test["label"].values.copy()
    X_lemm = project1.lemmatizer(X_train)
    vctrizer = TfidfVectorizer(norm='l2')
    vctrizer.fit(X_lemm)
    X_train = project1.mod_generate_feature_matrix(X_train, vctrizer)
    X_test = project1.mod_generate_feature_matrix(X_test, vctrizer)

    return (X_train, Y_train, X_test, Y_test)

def get_imbalanced_data(
    dictionary: dict[str, int],
    filename: str = "data/dataset.csv",
    ratio: float = 0.25,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
]:
    """
    Read in data from filename and return imbalanced dataset using extract_dictionary and
    generate_feature_matrix. Imbalanced dataset gets split into training and test sets.

    The binary labels take two values:
        -1: negative
         1: positive

    Args:
        dictionary: dictionary to create feature matrix from
        filename: name of the file to be read from
        ratio: ratio of negative to positive samples

    Returns:
        X_train: training feature matrix
        Y_train: training labels
        X_test: test feature matrix
        Y_test: test labels
    """
    dataframe = load_data(filename)
    dataframe = dataframe[dataframe["label"] != 0]
    positiveDF = dataframe[dataframe["label"] == 1].copy()
    negativeDF = dataframe[dataframe["label"] == -1].copy()
    negativeDF = negativeDF[: int(ratio * positiveDF.shape[0])]
    positive_class_size = 2 * positiveDF.shape[0] // 3
    negative_class_size = 2 * negativeDF.shape[0] // 3
    positiveDF = positiveDF.sample(frac=1, random_state=445)
    negativeDF = negativeDF.sample(frac=1, random_state=445)
    X_train = (
        pd.concat([positiveDF[:positive_class_size], negativeDF[:negative_class_size]])
        .reset_index(drop=True)
        .copy()
    )
    X_test = (
        pd.concat([positiveDF[positive_class_size:], negativeDF[negative_class_size:]])
        .reset_index(drop=True)
        .copy()
    )
    Y_train = X_train["label"].values.copy()
    Y_test = X_test["label"].values.copy()
    X_train = project1.generate_feature_matrix(X_train, dictionary)
    X_test = project1.generate_feature_matrix(X_test, dictionary)

    return (X_train, Y_train, X_test, Y_test)


# Note for students: altering class_size here is not allowed.
def get_multiclass_training_data(
    class_size: int = 750,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], TfidfVectorizer]:
    """
    Reads in the data from data/dataset.csv and returns it using
    extract_dictionary and generate_feature_matrix as a tuple
    (X_train, Y_train) where the labels are multiclass as follows:
        -1: negative
         0: neutral
         1: positive

    Args:
        class_size: size of each class in the training dataset

    Returns:
        X_train: training feature matrix
        Y_train: training labels
        dictionary: dictionary used to create feature matrix
    """
    fname = "data/dataset.csv"
    dataframe = load_data(fname)
    neutralDF = dataframe[dataframe["label"] == 0].copy()
    positiveDF = dataframe[dataframe["label"] == 1].copy()
    negativeDF = dataframe[dataframe["label"] == -1].copy()
    X_train = (
        pd.concat(
            [positiveDF[:class_size], negativeDF[:class_size], neutralDF[:class_size]]
        )
        .reset_index(drop=True)
        .copy()
    ) 
    Y_train = X_train["label"].values.copy()
    X_lemm = project1.lemmatizer(X_train)
    vctrizer = TfidfVectorizer(norm='l2')
    vctrizer.fit(X_lemm)
    X_train = project1.mod_generate_feature_matrix(X_train, vctrizer)

    return (X_train, Y_train, vctrizer)


def get_heldout_reviews(vctrizer: TfidfVectorizer) -> npt.NDArray[np.float64]:
    """
    Reads in the data from data/heldout.csv and returns it as a feature
    matrix based on the functions extract_dictionary and generate_feature_matrix.

    Args:
        dictionary: the dictionary created by get_multiclass_training_data

    Returns:
        X: feature matrix
    """
    fname = "data/heldout.csv"
    dataframe = load_data(fname)
    X = project1.mod_generate_feature_matrix(dataframe, vctrizer)
    return X


def generate_challenge_labels(y: npt.NDArray[np.int64], uniqname: str) -> None:
    """
    Writes your predictions to held_out_result.csv. Please make sure that you do not
    change the order of the ratings in the heldout dataset since we will use this file to
    evaluate your classifier.

    Args:
        y: predictions of multiclass classifier
        uniqname: your uniqname, which will be used to name the output file
    """
    pd.Series(np.array(y)).to_csv(uniqname + ".csv", header=["label"], index=False)


def filter_actors_and_actresses(filename: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract two dataframes from filename, containing every row which contains actor/acress
    in the review text.

    Args:
        filename: path to csv file containing dataframe

    Returns:
        df_actor: dataframe containing all rows of the original dataframe where the
                  review text contains the words 'actor' and/or 'actors' (not case
                  sensitive)
        df_actress: dataframe containing all rows of the original dataframe where the
                    review text contains the words 'actress' and/or 'actresses' (not case
                    sensitive)
    """

    df = load_data(filename)
    df_actor = df.loc[df["reviewText"].str.contains(r"\bactors?\b", case=False)]
    df_actress = df.loc[
        df["reviewText"].str.contains(r"\bactress(?:es)?\b", case=False)
    ]
    return df_actor, df_actress


def count_actors_and_actresses(filename: str) -> tuple[int, int]:
    """
    Returns the number of reviews in df_actor and df_actress from
    filter_actors_and_actresses.

    Args:
        filename: path to csv file containing dataframe

    Returns:
        number of reviews in df_actor
        number of reviews in df_actress
    """
    df_actor, df_actress = filter_actors_and_actresses(filename)
    return df_actor["reviewText"].count(), df_actress["reviewText"].count()


def plot_actors_and_actresses(filename: str, x_label: str) -> None:
    """
    Save "plot_actor_x_label.png" showing the distribution of labels or ratings across
    reviews mentioning actors and actresses.

    Args:
        filename: path to the csv file containing the dataframe
        x_label: name of the dataframe column to plot, either "label" or "rating"
    """
    df_actor, df_actress = filter_actors_and_actresses(filename)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    fig.supxlabel(x_label)
    fig.supylabel("proportion")

    ax1.set_title("Actor")
    ax2.set_title("Actress")

    num_bins = 3 if x_label == "label" else 5
    weights1 = np.ones_like(df_actor[x_label]) / float(df_actor[x_label].count())
    _, _, bars1 = ax1.hist(df_actor[x_label], bins=num_bins, weights=weights1)

    weights2 = np.ones_like(df_actress[x_label]) / float(df_actress[x_label].count())
    _, _, bars2 = ax2.hist(df_actress[x_label], bins=num_bins, weights=weights2)

    ax1.locator_params(axis="x", nbins=num_bins)
    ax2.locator_params(axis="x", nbins=num_bins)

    ax1.bar_label(bars1, fmt="%.2f")
    ax2.bar_label(bars2, fmt="%.2f")

    plt.savefig(f"plot_actor_{x_label}")
