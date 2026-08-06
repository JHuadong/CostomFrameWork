import dezero

if __name__ == '__main__':
    train_set = dezero.datasets.Spiral(train=True)
    print(train_set[0])
    print(len(train_set))