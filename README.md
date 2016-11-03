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
	python3 run_model.py -i YOUR_INPUT_PATH -o YOUR_OUTPUT_PATH
```

The input directory requires seven numpy matrices:

* **A.npy:** array-like or sparse matrix, shape = [m_reviews, n_pairs]
* **X_prime.npy:** array-like or sparse matrix, shape = [m_reviews, 2]
* **G.npy:** array-like or vector, shape = [n_pairs]
* **X_zero.npy:** array-like or sparse matrix, shape = [n_pairs, 2]
* **W_a.npy:** array-like or sparse matrix, shape = [n_pairs, n_pairs]
* **W_b.npy:** array-like or sparse matrix, shape = [n_pairs, n_pairs]
* **W_s.npy:** array-like or sparse matrix, shape = [n_pairs, n_pairs]

Usage
-----

There are two ways to execute the program

* The simpliest is use the command line to run the program
```python
python3 run_model.py -i YOUR_INPUT_PATH -o YOUR_OUTPUT_PATH
```

* You could also amend the file path in the Makefile and do it by using make, e.g.,
```
> make
```

Reference
---------

> Zhang, Yongfeng, et al. ["Do users rate or review?: boost phrase-level sentiment labeling
with review-level sentiment classification."](http://yongfeng.me/attach/bps-zhang.pdf) Proceedings of the 37th international ACM SIGIR
conference on Research & development in information retrieval. ACM, 2014.
