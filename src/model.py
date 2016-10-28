import os
import sys
import timeit

import numpy as np
import theano
import theano.tensor as T

class PhraseLevel_Sentiment_Classification(object):
    def __init__(self, rng, n_pairs, A, X_prime, lambda_1=1):

        self.X = theano.shared(
                value = np.asarray(
                    rng.uniform(low=-0.1, high=0.1, size=(n_pairs, 2)),
                    dtype=theano.config.floatX
                    ),
                name = 'X',
                borrow = True
                )
        
        self.A = A
        self.X_prime = X_prime
        self.lambda_1 = lambda_1

        self.params = [self.X]

        # R1 represents the squared error of Review-level Sentiment Orientation
        self.R1 = self.lambda_1 * T.sum((T.dot(self.A, self.X) - self.X_prime)**2)

    def cost(self):
        return self.R1

