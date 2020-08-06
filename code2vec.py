import pickle

with open('data/java14m/java14m.dict.c2v', mode='rb') as f:
    a = pickle.load(f)
    print(a)