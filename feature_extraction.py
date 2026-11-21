
from sklearn.feature_extraction.text import CountVectorizer

def create_vectorizer():

    vectorizer = CountVectorizer(
        ngram_range=(1, 3),
        lowercase=True,
        stop_words="english",
        max_features=10000,
        min_df=2
    )

    return vectorizer


def extract_features(train_text, test_text):

    vectorizer = create_vectorizer()

    # Learn vocabulary from training data only
    X_train = vectorizer.fit_transform(train_text)

    # Transform test data using the same vocabulary
    X_test = vectorizer.transform(test_text)

    return X_train, X_test, vectorizer
