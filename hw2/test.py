import numpy as np
from assignment2 import Assignment2
import intervals


def main():
    ass = Assignment2()
    #print(ass.experiment_m_range_erm(10, 100, 5, 3, 100))
    #print(ass.experiment_k_range_erm(1500, 1, 10, 1))
    #print(ass.experiment_k_range_srm(1500, 1, 10, 1))
    print(ass.cross_validation(1500))


if __name__ == "__main__":
    main()
