import argparse
import os
import pickle
import sys
import timeit

import numpy as np
import theano
import theano.tensor as T
from scipy.sparse import csr_matrix
from theano import sparse
from theano.tensor.shared_randomstreams import RandomStreams

from src.model import PhraseLevel_Sentiment_Classification

def load_data(input_dir):
    def load_sparse_csr(filename):
        loader = np.load(os.path.join(input_dir, filename))
        return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                shape = loader['shape'], dtype="float64")

    data = {
            'A': load_sparse_csr('A.npz'),
            'X_prime': np.load(os.path.join(input_dir, 'X_prime.npy')),
            'G': np.load(os.path.join(input_dir, 'G.npy')),
            'X_zero': np.load(os.path.join(input_dir, 'X_zero.npy')),
            'W_a': np.load(os.path.join(input_dir, 'W_a.npy')),
            'W_b': np.load(os.path.join(input_dir, 'W_b.npy')),
            'W_s': np.load(os.path.join(input_dir, 'W_s.npy')),
            }
    return data


def dump_model(args, clf):
    with open(os.path.join(args.output_dir, 'best_model.pkl'), 'wb') as fp:
        pickle.dump(clf, fp)

    np.save(os.path.join(args.output_dir, 'X'), clf.X.get_value())


def get_parser():
    parser = argparse.ArgumentParser(
            description='Phrase-Level Sentiment Classification'
            )

    parser.add_argument('-i', action='store', dest='input_dir',
            help='The input directory'
            )
    parser.add_argument('-o', action='store', dest='output_dir',
            help='The output directory'
            )

    return parser.parse_args()


def main():
    args = get_parser()
    
    # load dataset
    print('loading data...')
    data = load_data(args.input_dir)


    # build model 
    print('building the model...')

    # generate symbolic variables for input A, X_prime
    # A = T.matrix('A')
    A = sparse.csr_matrix('A')
    X_prime = T.matrix('X_prime')
    G = T.vector('G')
    X_zero = T.matrix('X_zero')
    W_a = T.matrix('W_a')
    W_b = T.matrix('W_b')
    W_s = T.matrix('W_s')

    # construct phrase-level sentiment classification
    clf = PhraseLevel_Sentiment_Classification(
            data['A'].shape[1], 
            A, X_prime,
            G, X_zero,
            W_a, W_b, 
            W_s
            )

    # cost function
    cost = clf.cost()

    # updating rule
    updates = [(clf.X, clf.X * T.sqrt(
        (clf.lambda_1 * sparse.basic.dot(clf.A.T, clf.X_prime) + 
            clf.lambda_2 * T.dot(clf.G, clf.X_zero) +
            clf.lambda_3 * T.dot(clf.W_a, clf.X) +
            clf.lambda_3 * T.dot(T.dot(clf.W_b, clf.X), clf.E) + 
            clf.lambda_4 * T.dot(clf.W_s, clf.X)
            ) /
        (clf.lambda_1 * T.dot(sparse.basic.dot(clf.A.T, A), clf.X) +
            clf.lambda_2 * T.dot(clf.G, clf.X) +
            clf.lambda_3 * T.dot(clf.D, clf.X) +
            clf.lambda_4 * T.dot(clf.D_s, clf.X) +
            10**-8
            )
        ))]

    
    train_model = theano.function(
            inputs = [
                A, X_prime,
                G, X_zero,
                W_a, W_b,
                W_s
                ],
            outputs = cost,
            updates = updates
            )

    # training
    print('training PhraseLevel Sentiment Classification...')
    patience = 2
    patience_threshold = 32
    improvement_threshold = 0.995

    start_time = timeit.default_timer()
    best_train_loss = np.inf

    epoch = 0
    n_epochs = 100
    while (epoch < n_epochs) and (patience > epoch):
        epoch += 1

        this_train_loss = train_model(
                data['A'], data['X_prime'],
                data['G'], data['X_zero'],
                data['W_a'], data['W_b'],
                data['W_s']
                )
        if this_train_loss < best_train_loss:
            if this_train_loss < best_train_loss * \
                    improvement_threshold:
                        patience = patience * 2 if patience * 2 < \
                                patience_threshold else patience + 1
                        
                        # save best model
                        dump_model(args, clf)
            best_train_loss = this_train_loss

        print('Epoch#{},patience: {},best: {},loss: {}'.format(
            epoch, patience, best_train_loss, this_train_loss)
            )
                                

    end_time = timeit.default_timer()

    print('Optimization complete with best train loss %f' %(best_train_loss))
    print(('The code for file ' +
        os.path.split(__file__)[1] +
        ' ran for %.2fmins' % ((end_time - start_time) / 60.)), file=sys.stderr)
        

if __name__ == '__main__':
    main()
