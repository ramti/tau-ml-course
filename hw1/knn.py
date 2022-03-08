from sklearn.datasets import fetch_openml
import numpy.random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def knn_classify_sample(train, train_labels, query, k):
    distances = np.linalg.norm(train - query, axis=1)
    train_indices = np.argsort(distances)[:k]

    # Get the most frequenct label
    labels, count = np.unique(train_labels[train_indices], return_counts=True)
    return labels[np.argmax(count)]


def run_knn(train, train_labels, test, test_labels, train_size, k):
    # Slice the train according to train_size
    train = train[:train_size]
    train_labels = train_labels[:train_size]

    # Measure accuracy
    correct = 0
    for i, query in enumerate(test):
        pred = knn_classify_sample(train, train_labels, query, k)
        if pred == test_labels[i]:
            correct += 1
    accuracy = correct / test.shape[0]
    return accuracy


def main():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    idx = numpy.random.RandomState(0).choice(70000, 11000)
    train = data[idx[:10000], :].astype(int)
    train_labels = labels[idx[:10000]]
    test = data[idx[10000:], :].astype(int)
    test_labels = labels[idx[10000:]]

    k_list = list(range(1, 101))
    runs_different_k = np.array([run_knn(train=train,
                                         train_labels=train_labels,
                                         test=test,
                                         test_labels=test_labels,
                                         train_size=1000,
                                         k=k)
                                 for k in k_list])

    print(f"K = 10 accuracy: {runs_different_k[9]}")
    print(f"Best K: {k_list[np.argmax(runs_different_k)]}, accuracy: {np.max(runs_different_k)}")

    sns.set()
    plt.title("KNN accuracy by K")
    plt.xlabel("K")
    plt.ylabel("% Accuracy")
    plt.plot(k_list, runs_different_k, 'ro')
    plt.show()

    train_size_list = list(range(100, 5001, 100))
    runs_different_train_size = np.array([run_knn(train=train,
                                                  train_labels=train_labels,
                                                  test=test,
                                                  test_labels=test_labels,
                                                  train_size=train_size,
                                                  k=1)
                                          for train_size in train_size_list])

    print(
        f"Best train size: {train_size_list[np.argmax(runs_different_train_size)]}, accuracy: {np.max(runs_different_train_size)}")

    sns.set()
    plt.title("KNN accuracy by train size")
    plt.xlabel("Train n. sample")
    plt.ylabel("% Accuracy")
    plt.plot(train_size_list, runs_different_train_size, 'ro')
    plt.show()

if __name__ == '__main__':
    main()
