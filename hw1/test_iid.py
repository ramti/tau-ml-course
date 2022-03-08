import random


def main():
    n = 100000
    random_iid = [random.random() for _ in range(n)]
    print(max(random_iid))


if __name__ == '__main__':
    main()
