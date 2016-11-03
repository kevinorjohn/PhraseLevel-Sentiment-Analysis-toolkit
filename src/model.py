import os
import sys
import timeit

import numpy as np
import theano
import theano.tensor as T

class PhraseLevel_Sentiment_Classification(object):
    def __init__(self, n_pairs, A, X_prime, G, X_zero, W_a, W_b, W_s,
            lambda_1=0.25, lambda_2=0.25, lambda_3=0.25, lambda_4=0.25):
        
        # Sentimet Lexicon X
        self.X = theano.shared(
                value = np.asarray(
                    np.random.dirichlet(np.ones(2), size=n_pairs),
                    dtype=theano.config.floatX
                    ),
                name = 'X',
                borrow = True
                )
        
        # input matrix
        self.A = A
        self.X_prime = X_prime
        self.G = T.nlinalg.diag(G)
        self.X_zero = X_zero
        self.W_a = W_a
        self.W_b = W_b
        self.W_s = W_s
        
        # weighting parameters
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        
        # diagonal matrices
        self.D = T.nlinalg.diag(T.sum(W_a, axis=1) + T.sum(W_b, axis=1))
        self.D_s = T.nlinalg.diag(T.sum(W_s, axis=1))

        # anti-diagonal matrix
        self.E = np.array([[0,1],[1,0]], dtype="float32")
        

        # R1 represents the error of Review-level Sentiment Orientation
        self.R1 = self.lambda_1 * \
                T.sqrt(T.sum((T.dot(self.A, self.X) - self.X_prime)**2))

        # R2 represents the error of General Sentiment Lexicon
        self.R2 = self.lambda_2 * \
                T.sqrt(T.sum((T.dot(self.G, self.X - self.X_zero))**2))

        # R3 represents the error of Linguistic Heuristic
        self.R3 = self.lambda_3 * \
                T.nlinalg.trace(
                        T.dot(T.dot(self.X.T, self.D), self.X) - 
                        T.dot(T.dot(self.X.T, self.W_a), self.X) - 
                        T.dot(T.dot(T.dot(self.X.T, self.W_b), self.X), self.E)
                        )

        # R4 represents the error of Sentential Sentiment Consistency
        self.R4 = self.lambda_4 * \
                T.nlinalg.trace(
                        T.dot(T.dot(self.X.T, self.D_s), self.X) -
                        T.dot(T.dot(self.X.T, self.W_s), self.X)
                        )
        
    def cost(self):
        return self.R1 + self.R2 + self.R3 + self.R4

