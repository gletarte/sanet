import itertools
import math
import json
from collections import Counter

from os.path import join, abspath, dirname, isfile

import pandas as pd
import numpy as np
from nltk import word_tokenize, sent_tokenize
from sklearn.utils import check_random_state
import torch
from torch.utils.data import TensorDataset, Dataset

DATASET_ROOT_PATH = join(dirname(abspath(__file__)), "..", "datasets")
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
CLASSES_FILE = "classes.txt"

TRAIN_TOKEN_FILE = "train_token.pkl"
TEST_TOKEN_FILE = "test_token.pkl"

STR_TO_REPLACE = [("\\n", " "), ("\\\"", "\""), ("<br />", " "), ("&lt;", "<"), ("&gt;", ">"), ("\\", " "), ("/", " "), (".", ". ")]

class TextDatasetBuilder(object):

    def __init__(self, name, word_vectors="glove", vector_size=50, random_state=42, build_inv_vocab=False):
        self.name = name
        self.word_vectors = word_vectors
        self.vector_size = vector_size
        self.rng = check_random_state(random_state)
        self.build_inv_vocab = build_inv_vocab
        self._load_dataset()
        self.classes = self._load_classes()

    def _load_classes(self):
        classes_path = join(DATASET_ROOT_PATH, self.name, CLASSES_FILE)
        if isfile(classes_path):
            classes = [line.rstrip() for line in open(classes_path)]
        else:
            classes = sorted(list(np.unique(self.train_labels)))
        return classes

    def _load_dataset(self):
        train_token_path = join(DATASET_ROOT_PATH, self.name, TRAIN_TOKEN_FILE)
        test_token_path = join(DATASET_ROOT_PATH, self.name, TEST_TOKEN_FILE)

        if not(isfile(train_token_path) and isfile(test_token_path)):
            print("Tokenized dataset not found -> Launching Tokenization...")
            self._tokenize_source()

        train = pd.read_pickle(train_token_path)
        self.train_corpus = train['text'].values
        self.train_labels = train['label'].values

        test = pd.read_pickle(test_token_path)
        self.test_corpus = test['text'].values
        self.test_labels = test['label'].values

    def _tokenize_source(self):
        for source, token in [(TRAIN_FILE, TRAIN_TOKEN_FILE), (TEST_FILE, TEST_TOKEN_FILE)]:
            df = pd.read_csv(join(DATASET_ROOT_PATH, self.name, source), header=None)
            text_index = 1
            if self.name in ["yahoo_answers"]:
                df[4] = df[1].astype(str) + " " + df[2].astype(str) + " " + df[3].astype(str)
                text_index = 4
            if self.name in ["amazon_review_polarity", "amazon_review_full", "ag_news", "dbpedia", "sogou_news"]:
                df[3] = df[1].astype(str) + ". " + df[2].astype(str)
                text_index = 3

            corpus = df[text_index].values
            labels = df[0].values - 1
            corpus = [self.tokenize(self._clean_text(text)) for text in corpus]
            df = pd.DataFrame.from_items([('label', labels), ('text', corpus)])
            df.to_pickle(join(DATASET_ROOT_PATH, self.name, token))

    def _clean_text(self, text):
        for old, new in STR_TO_REPLACE:
            text = text.replace(old, new)
        return text

    def build_vocab(self, corpus, min_freq=1):
        self.vocab = sorted(Counter(itertools.chain(*list(corpus))).items())
        self.vocab = [word for word, freq in self.vocab if freq >= min_freq and word != 'unk']
        self.vocab = {token:(i+2) for i, token in enumerate(self.vocab)}
        self.vocab['PAD'] = 0
        self.vocab['unk'] = 1

        if self.build_inv_vocab:
            self.inv_vocab = {i: token for token, i in self.vocab.items()}

    def tokenize(self, text):
        return [token.lower() for sent in sent_tokenize(text) for token in word_tokenize(sent)]

    def get_max_text_len(self, corpus):
        return np.amax([len(text) for text in corpus])

    def get_texts_len_distrib(self, corpus):
        return [len(text) for text in corpus]

    def vectorize(self, corpus, max_len=None):
        vcorpus = []
        for text in corpus:
            vtext = np.array([self.vocab[word] if word in self.vocab else self.vocab['unk'] for word in text])
            if max_len is not None:
                vtext = vtext[:max_len]
            vcorpus.append(vtext)
        return vcorpus

    def _load_glove_vectors(self):
        embeddings = {}
        path = join(DATASET_ROOT_PATH, "glove", "glove.6B." + str(self.vector_size) + "d.txt")
        with open(path, 'r', encoding='utf8') as embeddings_file:
            for line in embeddings_file:
                if len(line) > 50:
                    fields = line.strip().split(' ')
                    word = fields[0]
                    vector = np.asarray(fields[1:], dtype='float32')
                    embeddings[word] = vector
        return embeddings

    def _load_custom_vectors(self):
        embeddings = {}
        path = join(DATASET_ROOT_PATH, "custom_embeddings")
        vocab_to_int = json.load(open(join(path, f"{self.name}_vocab_to_int.json")))
        embedding_vectors = np.load(join(path, f"{self.name}_{self.vector_size}_embeddings.npy"))
        for word in vocab_to_int.keys():
            embeddings[word] = embedding_vectors[vocab_to_int[word]]
        return embeddings

    def build_embeddings(self):
        embeddings = np.random.rand(len(self.vocab), self.vector_size)

        if self.word_vectors != 'random':

            if self.word_vectors == 'glove':
                word_vectors = self._load_glove_vectors()
            elif self.word_vectors == 'custom':
                word_vectors = self._load_custom_vectors()

            for word, index in self.vocab.items():
                if word in word_vectors:
                    embeddings[index] = word_vectors[word]

        return embeddings
    def n_vocab_in_word_vectors(self):
        word_vectors = self._load_word_vectors()
        count = 0
        for word in self.vocab.keys():
            if word in word_vectors:
                count += 1
        return count

    def pre_process(self, min_freq=1, max_len=None):
        self.build_vocab(self.train_corpus, min_freq)
        self.train_corpus = self.vectorize(self.train_corpus, max_len)
        self.test_corpus = self.vectorize(self.test_corpus, max_len)

    def _split_train_valid(self, ratio):
        n_train = int(math.ceil(ratio*self.train_labels.shape[0]))
        data = list(zip(self.train_corpus, self.train_labels))
        self.rng.shuffle(data)
        self.train_corpus, self.train_labels = list(zip(*data))
        self.train_corpus = list(self.train_corpus)
        self.train_labels = np.array(self.train_labels)
        return self.train_corpus[:n_train],\
            self.train_labels[:n_train],\
            self.train_corpus[n_train:],\
            self.train_labels[n_train:]

    def _build_tensor_dataset(self, corpus, labels):
        return TensorDataset(torch.from_numpy(corpus).long(), torch.from_numpy(labels))

    def get_train_valid_test(self, ratio=0.8):
        train_corpus, train_labels, valid_corpus, valid_labels = self._split_train_valid(ratio)
        train = TextDataset(train_corpus, train_labels)
        valid = TextDataset(valid_corpus, valid_labels)
        test = TextDataset(self.test_corpus, self.test_labels)
        return train, valid, test

class TextDataset(Dataset):

    def __init__(self, corpus, labels,):
        self.corpus = corpus
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return (self.corpus[index], self.labels[index])

def collate_padding(samples):
    batch_texts, labels = list(zip(*samples))
    text_lens = [text.shape[0] for text in batch_texts]
    text_tensor = torch.zeros(max(text_lens), len(samples)).long()
    text_mask = torch.zeros(len(samples), max(text_lens)).bool()
    for i, (text, text_len) in enumerate(zip(batch_texts, text_lens)):
        text_tensor[:text_len, i] = torch.LongTensor(text)
        text_mask[i, text_len:] = 1
    labels = torch.LongTensor(np.array(labels))
    return (text_tensor, text_mask), labels
