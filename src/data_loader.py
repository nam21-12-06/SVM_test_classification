from sklearn.datasets import fetch_20newsgroups

def load_data():
    train_data = fetch_20newsgroups(subset="train")
    test_data = fetch_20newsgroups(subset="test")
    return train_data, test_data