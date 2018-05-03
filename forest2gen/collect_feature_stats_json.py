from collections import defaultdict
import argparse
import gzip
import json
import sys


def smart_reader(path):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt')
    return open(path, 'r')


def substr_before(x, pattern):
    idx = x.find(pattern)
    if idx < 0:
        return x
    return x[:idx]


class MaxMinMean(object):
    def __init__(self):
        self.min = None
        self.max = None
        self.sum = 0.0
        self.N = 0

    def append(self, f):
        if self.min is None or self.min > f:
            self.min = f
        if self.max is None or self.max < f:
            self.max = f
        self.sum += f
        self.N += 1

    def to_dict(self):
        return {'min': self.min, 'max': self.max, 'mean': self.sum / self.N}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Read a Ranklib/RankSVM training file and collect statistics about features.')
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('--quiet', required=False,
                        action='store_true', default=False)
    args = parser.parse_args()

    stats = defaultdict(MaxMinMean)

    with smart_reader(args.input_file) as fp:
        for i, line in enumerate(fp):
            cols = substr_before(line, '#').strip().split(' ')
            lbl = cols[0]
            qid = cols[1]
            for c in cols[2:]:
                fid, fval = c.split(':')
                stats[fid].append(float(fval))
            if not args.quiet and i % 1000 == 0:
                # print progress...
                sys.stderr.write('.')
                sys.stderr.flush()
    if not args.quiet:
        sys.stderr.write('\n')
    # save output:
    keep = dict((str(k), v.to_dict()) for k, v in stats.items())
    with open(args.output_file, 'w') as out:
        json.dump(keep, out, indent=2)
