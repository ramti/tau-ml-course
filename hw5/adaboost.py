#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    n_samples = X_train.shape[0]
    D = np.full(n_samples, 1 / n_samples)

    hypotheses = []
    alpha_vals = []
    for t in range(T):
        h, error = WL(X_train, y_train, D)
        print(f"Iteration {t + 1}")
        a = 0.5 * np.log((1 - error) / (error + 0.00001))
        hypotheses.append(h)
        alpha_vals.append(a)

        D = distribution_update(X_train, y_train, D, h, a)
    return hypotheses, alpha_vals



##############################################
# You can add more methods here, if needed.

def apply_final_classifier(X, A, H):
    """
    Apply the final classifier on sample x.
    :param X: sample matrix of size (n_samples, n_features).
    :param A: a list of a coefficients.
    :param H: a list of hypothesis, each of item is a tuple (h_pred, h_index, h_theta).
    :return: the classifier result.
    """
    result = sum(A[i] * apply_h(X, H[i]) for i in range(len(A)))
    result[result < 0] = -1
    result[result >= 0] = 1
    return result


def apply_h(X, h):
    """
    Apply h hypothesis on sample X.
    :param X: sample matrix of size (n_samples, n_features).
    :param h: a tuple that represent a hypothesis: (h_pred, h_index, h_theta).
    :return: an array of size (n_samples) with the results of the running h on the samples.
    """
    h_pred, h_index, h_theta = h
    h_results = X[:, h_index] <= h_theta
    h_results = h_results.astype(int)

    # set the appropriate values
    h_results[h_results == 1] = h_pred
    h_results[h_results == 0] = -h_pred
    return h_results


def emp_weighted_error(X, y, classifier_results, D=None):
    """
    Calculates the empirical weighted error.
    :param X: sample matrix of size (n_samples, n_features).
    :param y: an array represents the samples labels of size (n_sample).
    :param classifier_results: an array of results of the classifier on the samples, to be compared with y.
    :param D: distribution over the sample, an array of size (n_samples). If not provided, assumes uniform.
    :return: the empirical weighted error.
    """
    n_samples = len(y)
    if D is None:
        D = np.full(n_samples, 1 / n_samples)
    return sum((classifier_results != y).astype(int) * D)


def WL(X, y, D):
    """
    Calculates the best hypothesis.
    :param X: sample matrix of size (n_samples, n_features).
    :param y: an array represents the samples labels of size (n_sample).
    :param D: distribution over the sample, an array of size (n_samples).
    :return: a tuple of:
        1) a tuple that represent a hypothesis: (h_pred, h_index, h_theta).
        2) the weighted empirical error acquired by the hypothesis.
    """
    n_features = X.shape[1]

    min_h = None
    min_error = 1
    for feature_idx in range(n_features):
        # For each feature, check all the possible thresholds and take the minimum
        for val in np.unique(X[:, feature_idx]):
            h = (1, feature_idx, val)
            error = emp_weighted_error(X, y, apply_h(X, h), D)

            # If the error is larger than 0.5, than complement hypothesis is better
            if error > 0.5:
                h = (-1, feature_idx, val)
                error = 1 - error

            # Update the minimal hypo
            if error < min_error:
                min_error = error
                min_h = h
    return min_h, min_error


def distribution_update(X, y, D, h, a):
    """
    Update the distribution according to the chosen parameters.
    """
    new_D = D * np.exp(apply_h(X, h) * a * (-y))
    new_D = new_D / sum(new_D)
    return new_D


def calc_exp_loss(X, y, A, H):
    return np.average(np.exp((-y) * apply_final_classifier(X, A, H)))

##############################################

def sectionA(X_train, y_train, X_test, y_test, T=80):
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)

    T_list = np.array(list(range(1, T + 1)))
    train_error = [emp_weighted_error(X_train, y_train, apply_final_classifier(X_train,
                                                                               alpha_vals[:(t+1)],
                                                                               hypotheses[:(t+1)]))
                    for t in range(T)]
    test_error = [emp_weighted_error(X_test, y_test, apply_final_classifier(X_test,
                                                                            alpha_vals[:(t+1)],
                                                                            hypotheses[:(t+1)]))
                    for t in range(T)]

    plt.plot(T_list, train_error, label="Train error")
    plt.plot(T_list, test_error, label="Test error")
    plt.legend()
    plt.show()


def sectionB(X_train, y_train, vocab, T=10):
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    for h_pred, h_index, h_theta in hypotheses:
        h_pred_text = "Good" if h_pred == 1 else "Bad"
        print(f"{h_pred_text}: count(\"{vocab[h_index]}\") <= {h_theta}")


def sectionC(X_train, y_train, X_test, y_test, T=80):
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)

    T_list = np.array(list(range(1, T + 1)))
    exp_loss_train = [calc_exp_loss(X_train, y_train, alpha_vals[:(t+1)], hypotheses[:(t+1)])
                      for t in range(T)]
    exp_loss_test = [calc_exp_loss(X_test, y_test, alpha_vals[:(t+1)], hypotheses[:(t+1)])
                     for t in range(T)]

    plt.plot(T_list, exp_loss_train, label="Train exp loss")
    plt.plot(T_list, exp_loss_test, label="Test exp loss")
    plt.legend()
    plt.show()


def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data

    sectionA(X_train, y_train, X_test, y_test)
    sectionB(X_train, y_train, vocab)
    sectionC(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()



