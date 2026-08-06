import dezero
import pathlib

if __name__ == '__main__':
    train_set = dezero.datasets.MNIST(train=True, transform=None)
    test_set = dezero.datasets.MNIST(train=False, transform=None)

    print(len(train_set))
    print(len(test_set))
    #print()