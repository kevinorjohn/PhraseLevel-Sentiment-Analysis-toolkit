import argparse
import os
import pickle
import sys
import timeit

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from src.model import PhraseLevel_Sentiment_Classification

def load_data(input_dir):
    A = np.load(os.path.join(input_dir, 'A.npy'))
    X_prime = np.load(os.path.join(input_dir, 'X_prime.npy'))

    return A, X_prime


def dump_model(args, clf):
    with open(os.path.join(args.output_dir, 'best_model.pkl'), 'wb') as fp:
        pickle.dump(clf, fp)


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
    
    # TODO: load dataset
    print('loading data...')
    A_train, X_prime_train = load_data(args.input_dir)


    # TODO : build model 
    print('building the model...')

    # generate symbolic variables for input A, X_prime
    A = T.matrix('A')
    X_prime = T.matrix('X_prime')

    # construct phrase-level sentiment classification
    rng = np.random.RandomState(1126)
    clf = PhraseLevel_Sentiment_Classification(
            rng, 
            A_train.shape[1], 
            A, X_prime
            )

    # cost function
    cost = clf.cost()

    # updating rule
    updates = [(clf.X, 
        (clf.lambda_1 * T.dot(clf.A.T, clf.X_prime)) / \
                (clf.lambda_1 * T.dot(T.dot(clf.A.T,A), clf.X))
        )]

    
    train_model = theano.function(
            inputs = [A, X_prime],
            outputs = cost,
            updates = updates
            )

    # training
    print('training PhraseLevel Sentiment Classification...')
    patience = 2
    patience_threshold = 16
    improvement_threshold = 0.995

    start_time = timeit.default_timer()
    best_train_loss = np.inf

    epoch = 0
    n_epochs = 100
    while (epoch < n_epochs) and (patience > epoch):
        epoch += 1

        this_train_loss = train_model(A_train, X_prime_train)
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
