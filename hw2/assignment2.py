#################################
# Your name:
#################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        X = np.sort(np.random.uniform(size=m))
        Y = np.array([self._randomize_Y(x) for x in X])
        return np.array([X, Y]).transpose()

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        n_list = list(range(m_first, m_last + 1, step))
        results_empirical = []
        results_true = []
        for n in n_list:
            avg_empirical_error, avg_true_error = 0, 0
            for _ in range(T):
                samples = self.sample_from_D(n).transpose()
                h_intervals, error_count = intervals.find_best_interval(samples[0], samples[1], k)
                avg_empirical_error += error_count / n
                avg_true_error += self._calc_true_error(h_intervals)
            avg_empirical_error /= T
            avg_true_error /= T
            results_empirical.append(avg_empirical_error)
            results_true.append(avg_true_error)

        sns.set()
        plt.title("Experiment m range")
        plt.xlabel("n")
        plt.ylabel("Error")

        plt.plot(n_list, results_empirical, color='blue')
        plt.plot(n_list, results_true, color='red')
        plt.legend(['Avg. empirical error', 'Avg. true error'])
        plt.show()
        return np.array([results_empirical, results_true])

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        k_list = list(range(k_first, k_last + 1, step))
        results_empirical = []
        results_true = []
        for k in k_list:
            samples = self.sample_from_D(m).transpose()
            h_intervals, error_count = intervals.find_best_interval(samples[0], samples[1], k)
            results_empirical.append(error_count / m)
            results_true.append(self._calc_true_error(h_intervals))

        sns.set()
        plt.title("Experiment k range")
        plt.xlabel("k")
        plt.ylabel("Error")

        plt.plot(k_list, results_empirical, color='blue')
        plt.plot(k_list, results_true, color='red')
        plt.legend(['Empirical error', 'True error'])
        plt.show()
        return k_list[np.argmin(results_empirical)]

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        # penalty with lambda = 0.1
        penalty_func = lambda k: 2 * np.sqrt((2 * k + np.log(2 / 0.1)) / m)

        k_list = list(range(k_first, k_last + 1, step))
        results_empirical = []
        results_true = []
        results_penalty = []
        results_emp_penalty_sum = []
        for k in k_list:
            samples = self.sample_from_D(m).transpose()
            h_intervals, error_count = intervals.find_best_interval(samples[0], samples[1], k)
            empirical_error = error_count / m
            penalty = penalty_func(len(h_intervals))

            results_empirical.append(empirical_error)
            results_true.append(self._calc_true_error(h_intervals))
            results_penalty.append(penalty)
            results_emp_penalty_sum.append(empirical_error + penalty)

        sns.set()
        plt.title("Experiment k range SRM")
        plt.xlabel("k")
        plt.ylabel("Error / Penalty")

        plt.plot(k_list, results_empirical, color='blue')
        plt.plot(k_list, results_true, color='red')
        plt.plot(k_list, results_penalty, color='green')
        plt.plot(k_list, results_emp_penalty_sum, color='purple')
        plt.legend(['Empirical error', 'True error', 'Best h penalty', 'Penalty + Empirical error'])
        plt.show()
        return k_list[np.argmin(results_emp_penalty_sum)]

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        samples = self.sample_from_D(m)
        np.random.shuffle(samples)  # shuffle before splitting to train / test
        train_size = int(0.8 * len(samples))
        train, test = samples[:train_size].transpose(), samples[train_size:]
        train = train[:, np.argsort(train[0])]  # sort by the first row

        # train
        best_hypothesis_by_k = {}
        test_errors_by_k = {}
        for k in range(1, 11):
            h_intervals, _ = intervals.find_best_interval(train[0], train[1], k)
            best_hypothesis_by_k[k] = h_intervals
            test_errors_by_k[k] = self._calc_empirical_error(h_intervals, test)

        best_k = min(test_errors_by_k, key=test_errors_by_k.get)
        return best_k

    #################################
    # Place for additional methods
    #################################


    def _randomize_Y(self, x):
        """ Randomize the value of Y according to x """
        if (0.2 < x < 0.4) or (0.6 < x < 0.8):
            probs = [0.8, 0.2]
        else:
            probs = [0.1, 0.9]
        return np.random.choice([0, 1], p=probs)


    def _complement_intervals(self, intervals_lst):
        """ Returns a list of the complements intervals from 0 to 1 """
        complement = [(0, intervals_lst[0][0])]
        for i in range(1, len(intervals_lst)):
            complement.append((intervals_lst[i - 1][1], intervals_lst[i][0]))
        complement.append((intervals_lst[-1][1], 1))
        return complement

    def _intervals_intersection(self, intervals_lst, start, end):
        """ Returns a list of intervals that intersect with the interval
        marked with start and end. """
        intersection = []
        for interval_start, interval_end in intervals_lst:
            if interval_start <= start and interval_end >= end:
                intersection.append((start, end))
            elif interval_start <= start <= interval_end <= end:
                intersection.append((start, interval_end))
            elif start <= interval_start <= end <= interval_end:
                intersection.append((interval_start, end))
            elif start <= interval_start <= interval_end <= end:
                intersection.append((interval_start, interval_end))
        return intersection

    def _sum_integrals(self, start, end, intervals_list, probability):
        """
        Calculate the intersection of intervals with the interval (start, end).
        Sum the lengths of the intersected intervals and multiply with the probability:
        this is the intergral of the probability value in the intersected intervals.
        """
        intersection = self._intervals_intersection(intervals_list, start, end)
        lengths_sum = sum(item_end - item_start for item_start, item_end in intersection)
        return lengths_sum * probability

    def _calc_true_error(self, h_intervals_1):
        """  Calculates the true error of h function with intervals h_intervals_1 """
        p_intervals_1_08 = [(0, 0.2), (0.4, 0.6), (0.8, 1)]  # P(y = 1 | x) = 0.8
        p_intervals_1_01 = self._complement_intervals(p_intervals_1_08)  # P(y = 1 | x) = 0.1
        h_intervals_0 = self._complement_intervals(h_intervals_1)

        # sum all the integrals
        error = 0
        for start, end in h_intervals_1:
            # here h(x) = 1, so delta(h(x), y) = 1 iff y = 0.
            # Sum all integrals of [a, b] on P(y=0|X in [a,b]).
            error += self._sum_integrals(start, end, p_intervals_1_08, 0.2)
            error += self._sum_integrals(start, end, p_intervals_1_01, 0.9)

        for start, end in h_intervals_0:
            # here h(x) = 0, so delta(h(x), y) = 1 iff y = 1.
            # Sum all integrals of [a, b] on P(y=1|X in [a,b]).
            error += self._sum_integrals(start, end, p_intervals_1_08, 0.8)
            error += self._sum_integrals(start, end, p_intervals_1_01, 0.1)
        return error


    def _calc_empirical_error(self, h_intervals, queries):
        """
        Calculates the empirical error of h function on queries.
        :param h_intervals: list of tuples with h func intervals.
        :param queries: np array in shape (n, 2) with the sample and the label.
        :return: the empirical error.
        """
        error_count = 0
        for sample, label in queries:
            # calculate h value
            h_val = int(any(start <= sample <= end for start, end in h_intervals))
            if h_val != int(label):
                error_count += 1
        return error_count / len(queries)


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)

