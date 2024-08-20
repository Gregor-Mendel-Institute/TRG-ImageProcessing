from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool() as p:
        print(p.map(f, [1, 2, 3, 3, 5, 76, 7, 7, 3, 5, 7, 5, 76]))
