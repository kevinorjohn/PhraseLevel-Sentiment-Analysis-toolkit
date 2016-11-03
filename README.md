PhraseLevel-Sentiment-Analysis-toolkit
======================================

This is a Python3 Theano implemented toolkit for phrase-level sentiment analysis

Dependencey
-----------

* Numpy
* Python3
* Theano

Prerequisite
------------

Set the file path in the Makefile:
```
all:
	python3 run_model.py -i INPUT_PATH -o OUTPUT_PATH
```

The input directory requires seven numpy matrices:

1. 	A.npy
> array-like or sparse matrix, shape = [m_reviews, n_pairs]
2. 	X\_prime.npy
> array-like or sparse matrix, shape = [m_reviews, 2]
3. 	G.npy
> array-like or vector, shape = [n_pairs]
4. 	X\_zero.npy
> array-like or sparse matrix, shape = [n_pairs, 2]
5. 	W\_a.npy
> array-like or sparse matrix, shape = [n_pairs, n_pairs]
6. 	W\_b.npy
> array-like or sparse matrix, shape = [n_pairs, n_pairs]
7. 	W\_s.npy
> array-like or sparse matrix, shape = [n_pairs, n_pairs]
	

Reference
---------

> Zhang, Yongfeng, et al. ["Do users rate or review?: boost phrase-level sentiment labeling 
with review-level sentiment classification."](http://yongfeng.me/attach/bps-zhang.pdf) Proceedings of the 37th international ACM SIGIR 
conference on Research & development in information retrieval. ACM, 2014.
