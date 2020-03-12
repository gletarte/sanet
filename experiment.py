import argparse
import logging
import os
from os.path import join, abspath, dirname
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.init import xavier_normal_

from poutyne.framework import Experiment, ReduceLROnPlateau

from sanet.text_dataset import TextDatasetBuilder, collate_padding
from sanet.sanet import SANet

DATASETS = ["ag_news", "amazon_review_full", "amazon_review_polarity", "dbpedia", "yahoo_answers", "yelp_review_full", "yelp_review_polarity"]
WORD_VECTORS = ["glove", "custom", "random"]

RESULTS_PATH = os.environ.get('SANET_RESULTS_DIR', join(dirname(abspath(__file__)), "results"))

def launch(args=None):
    # Parser
    parser = argparse.ArgumentParser(description="Text Classification")
    parser.add_argument('--dataset', choices=DATASETS, default="yelp_review_polarity")
    parser.add_argument('--words', choices=WORD_VECTORS, default="glove")
    parser.add_argument('--len', type=int, default=1500)
    parser.add_argument('--freq', type=int, default=1)
    parser.add_argument('--exp', default="test")
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--vector', type=int, default=50)
    parser.add_argument('--blocks', type=int, default=1)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--pos', action='store_true', default=False)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    args = parser.parse_args(args)
    run(args)

def run(args):
    # Logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Initialization
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    # Fix bug in PyTorch where memory is still allocated on GPU0 when
    # asked to allocate memory on GPU1.
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    device = torch.device('cuda:%d' % args.device if torch.cuda.is_available() else 'cpu')

    # Building dataset
    dataset = TextDatasetBuilder(name=args.dataset,
                                 word_vectors=args.words,
                                 vector_size=args.vector,
                                 random_state=random_seed)
    logging.debug("Dataset built.")

    dataset.pre_process(min_freq=args.freq, max_len=args.len)
    embeddings = dataset.build_embeddings()
    logging.debug("Vocab size {}".format(len(dataset.vocab)))
    pos_enc_len = None
    if args.pos:
        pos_enc_len = args.len

    traind, validd, testd = dataset.get_train_valid_test()
    logging.debug("Split: train = {}, valid = {} and test = {}".format(len(traind), len(validd), len(testd)))

    # Creating Data Loaders
    train_loader = DataLoader(traind,
                              batch_size=args.batch,
                              shuffle=True,
                              collate_fn=collate_padding)
    valid_loader = DataLoader(validd,
                              batch_size=args.batch,
                              shuffle=False,
                              collate_fn=collate_padding)

    test_loader = DataLoader(testd,
                             batch_size=args.batch,
                             shuffle=False,
                             collate_fn=collate_padding)


    model = SANet(input_size=args.vector,
                  hidden_size=args.hidden,
                  n_classes=len(dataset.classes),
                  embeddings=embeddings,
                  n_blocks=args.blocks,
                  pos_enc_len=pos_enc_len)

    init_model(model)

    params = [p for n, p in model.named_parameters() if n != 'word_embedding.weight']
    optimizer = optim.SGD([{'params': model.word_embedding.parameters(), 'lr': args.lr * 0.1},
                           {'params': params, 'lr':args.lr, 'momentum':args.momentum}])

    # Preparing results output
    expt_path = join(RESULTS_PATH, args.dataset, args.exp)
    expt = Experiment(expt_path, model, device=device, logging=True, optimizer=optimizer, task='classifier')

    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                  mode='min',
                                  patience=2,
                                  factor=0.5,
                                  threshold_mode='abs',
                                  threshold=1e-3,
                                  verbose=True)

    expt.train(train_loader, valid_loader,
               epochs=args.epochs,
               lr_schedulers=[reduce_lr])

    expt.test(test_loader)

    print("### DONE ###")

def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            xavier_normal_(m.weight.data, 1)
            m.bias.data.zero_()


if __name__ == '__main__':
    launch()
