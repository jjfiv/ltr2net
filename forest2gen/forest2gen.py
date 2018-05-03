import xml.etree.ElementTree as ET
import gzip
from collections import defaultdict


class EvalNode(object):
    def __init__(self, fid, value):
        self.yes = None
        self.no = None
        self.weight = 1.0
        self.fid = fid
        self.value = value

    def eval(self, arr):
        if arr[self.fid] <= self.value:
            return self.weight * self.yes.eval(arr)
        else:
            return self.weight * self.no.eval(arr)

    def visit(self, fn):
        fn(self)
        self.yes.visit(fn)
        self.no.visit(fn)


class EvalLeaf(object):
    def __init__(self, response):
        self.response = response
        self.weight = 1.0

    def eval(self, arr):
        return self.weight * self.response

    def visit(self, fn):
        fn(self)


class EvalEnsemble(object):
    def __init__(self, trees):
        self.trees = trees
        self.weight = 1.0

    def eval(self, arr):
        return self.weight * sum([t.eval(arr) for t in self.trees])

    def visit(self, fn):
        fn(self)
        for t in self.trees:
            t.visit(fn)

    def find_splits(self):
        splits = []

        def include(node, splits):
            if type(node) == EvalNode:
                splits.append(node)
        self.visit(lambda n: include(n, splits))
        return splits

    def find_midpoints_by_fid(self):
        midpoints = defaultdict(list)
        for split in self.find_splits():
            midpoints[split.fid].append(split.value)
        return dict((fid, set(mps)) for fid, mps in midpoints.items())


def _parse_split(split):
    output = split.find('output')
    # if there's an <output> tag, we're at a leaf:
    if output is not None:
        return EvalLeaf(float(output.text))

    # otherwise, split based on this feature.
    fid = int(split.findtext('feature'))
    cond = float(split.findtext('threshold'))
    current = EvalNode(fid, cond)

    # recursively translate the whole tree.
    for child in split.findall('split'):
        pos = child.get('pos')
        recurse = _parse_split(child)
        if pos == 'left':
            current.yes = recurse
        else:
            current.no = recurse
    return current


def load_ranklib_model_reader(reader):
    comments = []
    model = []
    for line in reader:
        if line.startswith('##'):
            comments.append(line)
            continue
        else:
            model.append(line)
    ensemble = ET.fromstring('\n'.join(model))
    keep = []
    for tree in ensemble.findall('tree'):
        root_split_node = tree.find('split')
        root = _parse_split(root_split_node)
        root.weight = float(root_split_node.get("weight", default="1.0"))
        keep.append(root)
    return EvalEnsemble(keep)


def smart_reader(path):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt')
    return open(path, 'r')


def load_ranklib_model(path):
    with smart_reader(path) as fp:
        return load_ranklib_model_reader(fp)


ensemble = None
if __name__ == '__main__':
    ensemble = load_ranklib_model(
        'tree_parsing/mq07.lambdaMart.l10.kcv10.tvs90.gz')
