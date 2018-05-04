import sys
import numpy as np
from forest2gen import smart_reader


def load_ranklib_file(path, quiet=False):
    X = []
    y = []
    names = []
    qids = []
    with smart_reader(path) as fp:
        for i, line in enumerate(fp):
            data, name = line.split('#')
            names.append(name.strip())
            cols = data.strip().split(' ')
            lbl = float(cols[0].strip())
            y.append(lbl)
            qid = cols[1].split('qid:')[1].strip()
            qids.append(qid)

            D = len(cols[2:])+1
            x = np.zeros(D)

            for c in cols[2:]:
                fid, fval = c.split(':')
                x[int(fid)] = float(fval)
            X.append(x)

            if not quiet and i % 1000 == 0:
                # print progress...
                sys.stderr.write('.')
                sys.stderr.flush()
    if not quiet:
        sys.stderr.write('\n')

    return np.array(X), np.array(y), np.array(qids), np.array(names)
