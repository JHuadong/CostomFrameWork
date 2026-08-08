import cupy as cp
if __name__ == '__main__':
    x = cp.arange(6).reshape(2, 3)
    print(x)
    y = x.sum(axis=1)
    print(y)