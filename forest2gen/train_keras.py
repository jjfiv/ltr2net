from keras.models import Sequential
from keras.layers import Dense, Input
from keras import backend as K
import argparse
import json
import numpy as np
from forest2gen import smart_reader, load_ranklib_model, split_points_to_generator
from ranklib2numpy import load_ranklib_file
from sklearn.metrics import average_precision_score
from collections import defaultdict


def relu6(x):
    return K.relu(x, max_value=6)


def create_dnn_model(D, layers=[500, 100], optimizer='adam', loss='mean_absolute_error', activation=relu6):
    assert(len(layers) >= 1)
    model = Sequential()
    model.add(Dense(layers[0], activation=activation, input_shape=(D,)))
    for layer in layers[1:]:
        model.add(Dense(layer, activation=activation))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=loss)
    return model


def compute_aps(pred_y, test_y, test_qids):
    pred = defaultdict(list)
    test = defaultdict(list)
    for i, qid in enumerate(test_qids):
        pred[qid].append(pred_y[i])
        test[qid].append(test_y[i])

    aps = []
    for qid, ys in pred.items():
        truth = np.array(test[qid]) > 0
        # Gov2 has a query with no relevant documents...
        if np.sum(truth) == 0:
            continue
        actual = np.array(ys)
        aps.append(average_precision_score(truth, actual))

    return np.array(aps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a DNN from a LTR Forest Model.')
    parser.add_argument('input_model')
    parser.add_argument('stats_file')
    parser.add_argument('test_data')
    parser.add_argument('output_model')
    parser.add_argument('--quiet', required=False,
                        action='store_true', default=False)
    parser.add_argument('--batch_size', required=False, type=int, default=32)
    parser.add_argument('--batch_steps', required=False, type=int, default=200)
    parser.add_argument('--num_batches', required=False, type=int, default=20)
    args = parser.parse_args()

    test_X, test_y, test_qids, names = load_ranklib_file(args.test_data)
    ensemble = load_ranklib_model(args.input_model)

    ensemble_ys = []
    for i in range(len(test_qids)):
        ensemble_ys.append(ensemble.eval(test_X[i, :]))
    ensemble_aps = compute_aps(ensemble_ys, test_y, test_qids)
    print('ensemble.mAP: {0}'.format(np.mean(ensemble_aps)))

    fstats = None
    with smart_reader(args.stats_file) as fp:
        fstats = dict((int(k), v) for k, v in json.load(fp).items())

    D = len(fstats)+1
    print("fstats.size={0}".format(len(fstats)))
    model = create_dnn_model(D, [500, 100])

    generate_fn = split_points_to_generator(ensemble, fstats)

    for b in range(args.num_batches):
        train_loss = []
        for i in range(args.batch_steps):
            X, y = generate_fn(args.batch_size)
            loss = model.train_on_batch(X, y)
            train_loss.append(loss)
        print('Train[{2},{1}].Loss: {0}'.format(np.mean(loss), i, b))
        # predict on test data:
        pred_y = model.predict(test_X)
        aps = compute_aps(pred_y, test_y, test_qids)
        print('Test.mAP: {0}'.format(np.mean(aps)))
