###########################################################
#
# Reuters RCV1 text classification task
#
###########################################################


from __future__ import division
from __future__ import print_function
import numpy as np
import os
import sys
from time import time
try:
    import cPickle as pickle
except:
    import pickle

import argparse
from softmax import SGD, SAGA
from loader import load_covtype, load_mnist, load_news20, load_rcv1, load_reuters4


def load(data_name, work_directory):
    if data_name == 'covtype':
        return load_covtype(work_directory)
    elif data_name == 'mnist':
        return load_mnist(work_directory)
    elif data_name == 'news20':
        return load_news20(work_directory)
    elif data_name == 'rcv1':
        return load_rcv1(work_directory)
    elif data_name == 'reuters4':
        return load_reuters4(work_directory)
    else:
        raise ValueError('Data %s not found. Abort.' % data_name)


def main(args):
    # set seed
    np.random.seed(args.seed)

    # load dataset
    print('loading data ...')
    sys.stdout.flush()

    work_directory = os.path.join(args.file_path, args.data_name)
    data = load(args.data_name, work_directory)
    print('num_train   : %6d' % data['train_y'].shape[0])
    print('num_test    : %6d' % data['test_x'].shape[0])
    print('num_features: %6d' % data['test_x'].shape[1])
    print('num_classes : %6d' % data['train_y'].shape[1])
    sys.stdout.flush()

    # set model
    print('\ncreating a model ...')
    sys.stdout.flush()
    eta = 10 ** -args.learn_rate
    filename = '%s_%s_%d_' % (args.solver, args.method, args.seed)
    weights_path = None
    if args.weights_path:
        weights_path = os.path.join(args.weights_path, args.data_name, filename)

    if args.solver == 'sgd':
        model = SGD(epochs=args.epochs, method=args.method, eta=eta,
                    eval_every=args.eval_every, cache_path=weights_path)
    elif args.solver == 'saga':
        model = SAGA(epochs=args.epochs, method=args.method, eta=eta,
                     eval_every=args.eval_every, cache_path=weights_path)
    else:
        raise ValueError('Solver %s not found. Abort.' % args.solver)

    # training
    print('start training ...')
    sys.stdout.flush()
    start = time()
    model.fit(data['train_x'], data['train_y'], data['test_x'], data['test_y'])
    end = time()
    print('done.\n')
    print('train time: %f [sec.]' % (end - start))
    print('stopping point: %f' % model.stop_point)
    print('test score: %f' % model.best_score)
    sys.stdout.flush()

    save_path = os.path.join(args.results_path, args.data_name, filename)
    np.array(model.train_log).tofile(save_path + 'train.dat')
    #np.array(model.train_reward).tofile(save_path + 'train_reward.dat')
    np.array(model.dev_log).tofile(save_path + 'dev.dat')
    #np.array(model.dev_reward).tofile(save_path + 'dev_reward.dat')
    np.array(model.norm_log).tofile(save_path + 'norm.dat')
    np.array(model.variance_log).tofile(save_path + 'variance.dat')

    #with open(save_path + 'model.pkl', 'wb') as output_file:
    #     pickle.dump(model, output_file)

    # plot
    #print('plotting results ...')
    #plot(train_scores, test_scores)


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--learn_rate", default=1.0, type=float, help="learning rate (10**-g)")
    parser.add_argument("-m", "--method", default='full', type=str, help="method name {full, greedy, bandit, cv}")
    parser.add_argument("-s", "--solver", default='saga', type=str, help="solver name {sgd, saga}")
    parser.add_argument("-e", "--epochs", default=10, type=int, help="number of epochs")
    parser.add_argument("-w", "--weights_path", type=str, default=None, help="path to dump weights")
    parser.add_argument("-r", "--results_path", type=str, default='results', help="path to save results")
    parser.add_argument("-f", "--file_path", type=str, default='data', help="path to data root")
    parser.add_argument("-d", "--data_name", type=str, default='rcv1', help="data name {covtype, mnist, news20, rcv1}")
    parser.add_argument("--seed", type=int, default=123, help="seed")
    parser.add_argument("--eval_every", type=int, default=1000, help="evaluate every * iterations")
    args = parser.parse_args()

    for k, v in vars(args).items():
        print('%10s %s' % (str(k), str(v)))
    print()
    main(args)
