###########################################################
#
# Data loader
#
###########################################################


from __future__ import print_function
from __future__ import division
import os
import sys
import urllib
import gzip
import numpy as np
import scipy.sparse as sp
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import LabelBinarizer

SOURCE_URL = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/'


def maybe_download(filename, work_directory, source_url=SOURCE_URL):
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.urlretrieve(source_url + filename, filepath)
        print('Succesfully downloaded:', filename)
    return filepath


def one_hot_encoding(labels, unique):
    one_hot = np.zeros((len(labels), len(unique)))
    for i in range(len(labels)):
        j = unique.index(labels[i])
        one_hot[i, j] = 1.0
    return one_hot


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)


def _check_magic(magic, number):
    if magic != number:
        raise ValueError('Invalid magic number %d (must be %d)' % (magic, number))


def extract_features(filename):
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        _check_magic(magic, 2051)
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
    return data


def extract_labels(filename):
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        _check_magic(magic, 2049)
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return labels


def load_mnist(work_directory):
    #s = {'train': 60000, 'test': 10000, 'dim': 784, 'k': 10}
    data = {}
    source_url = 'http://yann.lecun.com/exdb/mnist/'
    
    # load
    #for data_name, t in zip(['train', 'test'], ['train', 't10k']):
    #    x_filename = t+'-images-idx3-ubyte.gz'
    #    y_filename = t+'-labels-idx1-ubyte.gz'
    #    x_filepath = maybe_download(x_filename, work_directory, source_url)
    #    y_filepath = maybe_download(y_filename, work_directory, source_url)
    #    data[data_name+'_x'] = extract_features(x_filepath)
    #    data[data_name+'_y'] = extract_labels(x_filepath)
    
    # one-hot encoding
    #unique = list(np.unique(data['train_y']))
    #data['train_y'] = one_hot_encoding(data['train_y'], unique)
    #data['test_y'] = one_hot_encoding(data['test_y'], unique)
    
    #assert data['train_x'].shape == (s['train'], s['dim'])
    #assert data['train_y'].shape == (s['train'], s['k'])
    #assert data['test_x'].shape == (s['test'], s['dim'])
    #assert data['test_y'].shape == (s['test'], s['k'])
    
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(work_directory, one_hot=True, 
                                      validation_size=0)
    
    data['train_x'] = mnist.train.images
    data['train_y'] = mnist.train.labels
    data['test_x'] = mnist.test.images
    data['test_y'] = mnist.test.labels
    
    return data


def load_reuters4(work_directory):
    d = np.load(os.path.join(work_directory, '4classes.npz'))
    X_train, Y_train = d['train']
    X_test0, Y_test0 = d['test0']
    #X_test1, Y_test1 = d['test1']
    #X_test2, Y_test2 = d['test2']
    #X_test3, Y_test3 = d['test3']
    #data = {
    #    'train_x': sp.vstack((X_test0, X_test1, X_test2, X_test3)),
    #    'train_y': np.vstack((Y_test0, Y_test1, Y_test2, Y_test3)),
    #    'test_x': X_train,
    #    'test_y': Y_train
    #}
    data = {
        'train_x': X_test0,
        'train_y': Y_test0,
        'test_x': X_train,
        'test_y': Y_train
    }
    return data


def load_rcv1(work_directory):
    s = {'train': 15564, 'test': 518571, 'dim': 47236, 'k': 51}
    data = {}
    
    # load
    for data_name in ['train', 'test']:
        filename = 'rcv1_'+data_name+'.multiclass.bz2'
        filepath = maybe_download(filename, work_directory)
        print('Extracting', filepath)
        sys.stdout.flush()
        x, y = load_svmlight_file(filepath)
        data[data_name+'_x'] = x
        data[data_name+'_y'] = y
    
    # one-hot encoding
    lb = LabelBinarizer()
    lb.fit(data['train_y'])
    data['train_y'] = lb.transform(data['train_y'])
    data['test_y'] = lb.transform(data['test_y'])

    #unique = list(np.unique(data['train_y']))
    #data['train_y'] = one_hot_encoding(data['train_y'], unique)
    #data['test_y'] = one_hot_encoding(data['test_y'], unique)

    # check shape
    assert data['train_x'].shape == (s['train'], s['dim']), data['train_x'].shape
    assert data['train_y'].shape == (s['train'], s['k']), data['train_y'].shape
    assert data['test_x'].shape == (s['test'], s['dim']), data['test_x'].shape
    assert data['test_y'].shape == (s['test'], s['k']), data['test_y'].shape

    return data


def load_news20(work_directory):
    s = {'train': 15935, 'test': 3993, 'dim': 62061, 'k': 20}
    data = {}
    
    # load
    for data_name, t in zip(['train', 'test'], ['', '.t']):
        filename = 'news20'+t+'.scale.bz2'
        filepath = maybe_download(filename, work_directory)
        print('Extracting', filepath)
        sys.stdout.flush()
        x, y = load_svmlight_file(filepath)
        data[data_name+'_x'] = x
        data[data_name+'_y'] = y
    
    # one-hot encoding
    lb = LabelBinarizer()
    lb.fit(data['train_y'])
    data['train_y'] = lb.transform(data['train_y'])
    data['test_y'] = lb.transform(data['test_y'])
    #unique = list(np.unique(data['train_y']))
    #data['train_y'] = one_hot_encoding(data['train_y'], unique)
    #data['test_y'] = one_hot_encoding(data['test_y'], unique)

    # reshape (pad with zero)
    data['test_x'] = sp.csr_matrix((data['test_x'].data,
                                    data['test_x'].indices,
                                    data['test_x'].indptr),
                                   shape=(s['test'], s['dim']))

    # check shape
    assert data['train_x'].shape == (s['train'], s['dim'])
    assert data['train_y'].shape == (s['train'], s['k'])
    assert data['test_x'].shape == (s['test'], s['dim'])
    assert data['test_y'].shape == (s['test'], s['k'])

    return data


def load_covtype(work_directory):
    s = {'train': 522911, 'test': 58101, 'dim': 54, 'k': 7}
    data = {}
    
    # load
    filename = 'covtype.scale01.bz2'
    filepath = maybe_download(filename, work_directory)
    print('Extracting', filepath)
    sys.stdout.flush()
    x, y = load_svmlight_file(filepath)
    
    # one-hot encoding
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    #unique = list(np.unique(y))
    #y = one_hot_encoding(y, unique)
    
    data['train_x'] = x[:s['train'], :]
    data['train_y'] = y[:s['train'], :]
    data['test_x'] = x[s['train']:, :]
    data['test_y'] = y[s['train']:, :]

    # check shape
    assert data['train_x'].shape == (s['train'], s['dim'])
    assert data['train_y'].shape == (s['train'], s['k'])
    assert data['test_x'].shape == (s['test'], s['dim'])
    assert data['test_y'].shape == (s['test'], s['k'])

    return data

