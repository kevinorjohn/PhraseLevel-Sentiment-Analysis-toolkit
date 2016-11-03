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

* **A.npy:** matrix, shape = [m_reviews, n_pairs]
* **X\_prime.npy:** matrix, shape = [m_reviews, 2]
* **G.npy:** vector, shape = [n_pairs]
* **X\_zero.npy:** matrix, shape = [n_pairs, 2]
* **W\_a.npy:** 
    > matrix, shape = [n_pairs, n_pairs]
* **W\_b.npy:**
    > array-like or sparse matrix, shape = [n_pairs, n_pairs]
* **W\_s.npy:**
    > array-like or sparse matrix, shape = [n_pairs, n_pairs]


Reference
---------

> Zhang, Yongfeng, et al. ["Do users rate or review?: boost phrase-level sentiment labeling
with review-level sentiment classification."](http://yongfeng.me/attach/bps-zhang.pdf) Proceedings of the 37th international ACM SIGIR
conference on Research & development in information retrieval. ACM, 2014.
