import os
import sys
import timeit

import numpy as np
import theano
import theano.tensor as T

class PhraseLevel_Sentiment_Classification(object):
    def __init__(self, rng, n_pairs, A, X_prime, lambda_1=1):

                # value = np.asarray(
                    # rng.uniform(low=0, high=0.5, size=(n_pairs, 2)),
                    # dtype=theano.config.floatX
                    # ),
        self.X = theano.shared(
                value = np.asarray(
                    np.random.dirichlet(np.ones(2), size=n_pairs),
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
        self.R1 = self.lambda_1 * \
                T.sqrt(T.sum((T.dot(self.A, self.X) - self.X_prime)**2))

    def cost(self):
        return self.R1

