import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse, random, json
import numpy as np
import itertools
from tqdm import tqdm
from forest2gen import smart_reader, load_ranklib_model, split_points_to_generator
from ranklib2numpy import load_ranklib_file
from sklearn.metrics import average_precision_score
from collections import defaultdict

class ApproximateRanker(nn.Module):
    def __init__(self, dim, layers=[500,100]):
        super(ApproximateRanker, self).__init__()
        steps = []
        for i in range(len(layers)):
            if i == 0:
                steps.append(nn.Linear(dim, layers[i]))
                steps.append(nn.ReLU6())
            else:
                steps.append(nn.Linear(layers[i-1], layers[i]))
                steps.append(nn.ReLU6())
        steps.append(nn.Linear(layers[-1], 1))
        steps.append(nn.ReLU6())
        self.forward_layers = nn.Sequential(*steps)

    def forward(self, xs):
        return self.forward_layers(xs)

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

def batch(lst, n):
    l = len(lst)
    for ndx in range(0, l, n):
        yield lst[ndx:min(ndx + n, l)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a DNN from a LTR Forest Model.')
    parser.add_argument('input_model')
    parser.add_argument('stats_file')
    parser.add_argument('train_data')
    parser.add_argument('test_data')
    parser.add_argument('output_model')
    parser.add_argument('--quiet', required=False,
                        action='store_true', default=False)
    # Note that our batches are actually twice this size.
    parser.add_argument('--batch_size', required=False, type=int, default=32)
    args = parser.parse_args()

    print('load test data')
    test_X, test_y, test_qids, names = load_ranklib_file(args.test_data)
    print('load train data')
    train_X, train_y, _, _ = load_ranklib_file(args.test_data)
    print('load ranklib ensemble')
    ensemble = load_ranklib_model(args.input_model)

    training_data = []
    for i in range(len(train_y)):
        xv = train_X[i,:]
        y_forest = ensemble.eval(xv)
        training_data.append( (y_forest, xv) )
    print('collected predictions')

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
    model = ApproximateRanker(D, [500, 100])

    generate_fn = split_points_to_generator(ensemble, fstats)

    model.train()
    random.shuffle(training_data)
    progress = tqdm(batch(training_data, args.batch_size))
    for minibatch in progress:
        train_ys = torch.tensor([y for (y,_) in minibatch], dtype=torch.float)
        train_xs = torch.cat([torch.tensor(x) for (_,x) in minibatch], axis=0)
        cover_x, cover_y = generate_fn(args.batch_size)
        batch_xs = torch.cat([torch.from_numpy(cover_x), train_xs], axis=0)
        batch_ys = torch.cat([torch.from_numpy(cover_y), train_ys], axis=0)
        print(batch_xs.shape, batch_ys.shape)
        #aps = compute_aps(pred_y, test_y, test_qids)
        #print('Test.mAP: {0}'.format(np.mean(aps)))
