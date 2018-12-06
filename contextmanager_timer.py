import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def load_data():
    return

def preprocess():
    return

def feature_engineering():
    return

def modeling():
    return

def main(): 
    with timer('process loading'):
        load_data()

    with timer('process preprocessing'):
        preprocess()

    with timer('process feature-engineering'):
        feature_engineering()

    with timer('process modeling'):
        modeling()
    
    
if __name__ == "__main__":
    main()
