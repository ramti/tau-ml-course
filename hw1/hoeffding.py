import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

N = 200000
n = 20
BERNOULLI_PROB = 0.5


def main():
    arr = np.array([np.random.binomial(1, p=BERNOULLI_PROB) for _ in range(N * n)]).reshape(N, n)
    empirical_mean = np.sum(arr, axis=1) / arr.shape[1]

    epsilons = np.linspace(0, 1, 50)
    empirical_probability = [np.count_nonzero(np.abs(empirical_mean - 0.5) >= epsilon) / empirical_mean.size
                             for epsilon in epsilons]
    hoefdding = [2 * np.exp(-2 * n * (epsilon ** 2)) for epsilon in epsilons]

    sns.set()
    plt.title("Hoeffding bound")
    plt.xlabel("ε")
    plt.ylabel("P")

    plt.plot(epsilons, empirical_probability, 'ro')
    plt.plot(epsilons, hoefdding, 'ro', color='blue')
    plt.legend(['Empirical mean', 'Hoeffding bound'])
    plt.show()


if __name__ == '__main__':
    main()
