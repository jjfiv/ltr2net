# ltr2net
Code and data to train a simple multilayer perceptron (feed-forward neural network) in pytorch from a regression forest learning to rank model (e.g., LambdaMART, RF, GBDT, etc.).

This code is a re-implementation of experiments done for the following SIGIR paper:

> Cohen, D.,  Foley, J.,  Zamani, H.,  Allan, J. and Croft, W. B. , ["Universal Approximation Functions for Fast Learning to Rank,"](http://ciir-publications.cs.umass.edu/pub/web/getpdf.php?id=1309) to appear in the Proceedings of the 41st International ACM SIGIR Conference on Research and Development in Information Retrieval, Ann Arbor, MI, Jul. 8-12 2018 (SIGIR’18)

## Data

The paper uses the LTR dataset from Microsoft: [MSN30k dataset](https://www.microsoft.com/en-us/research/project/mslr/).

The Gov2/MQ07 data for the paper will be up shortly. The features for MQ07/Gov2 we used are about 1.5GB compressed, and we need to figure out a good way to host that much data.
