prepare datasets

```
$ mkdir data
$ wget https://s3.amazonaws.com/code2vec/data/java14m_data.tar.gz
$ tar -xvzf java14m_data.tar.gz
$ mv java14m data
```

for local testing

```
$ python code2vec.py --train --batch-size 2
```

for training

```
$ python code2vec.py --train
```