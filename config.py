import argparse
import os


UPDATE_OPS_COLLECTION = 'resnet_update_ops' # ops group
BN_EPSILON = 0.001


class Config(object):
    def __init__(self, pattern=None):
        self.parse = argparse.ArgumentParser()
        if pattern == 'cnn':
            self.cnn_config()
        elif pattern == 'rnn':
            self.rnn_config()
        else:
            self.file_config()

    def file_config(self):
        self.parse.add_argument('--train_path', default='data/cnews.train.txt')
        self.parse.add_argument('--test_path', default='data/cnews.test.txt')
        self.parse.add_argument('--val_path', default='data/cnews.test.txt')
        self.parse.add_argument('--vocab_path', default='data/cnews.vocab.txt')
        self.parse.add_argument('--save_path', default='checkpoints/textcnn/best_validation')

    def rnn_config(self):
        self.parse.add_argument('--embedding_dim', default=64)
        self.parse.add_argument('--seq_length', default=600)
        self.parse.add_argument('--num_classes', default=10)
        self.parse.add_argument('--vocab_size', default=5000)
        self.parse.add_argument('--num_layers', default=2)
        self.parse.add_argument('--hidden_dim', default=128)
        self.parse.add_argument('--rnn', default='gru')
        self.parse.add_argument('--dropout_keep_prob', default=0.8)
        self.parse.add_argument('--learning_rate', default=1e-3)
        self.parse.add_argument('--batch_size', default=128)
        self.parse.add_argument('--num_epochs', default=10)
        self.parse.add_argument('--print_per_batch', default=100)
        self.parse.add_argument('--save_per_batch', default=10)

    def cnn_config(self):
        self.parse.add_argument('--embedding_dim', default=64)
        self.parse.add_argument('--seq_length', default=600)
        self.parse.add_argument('--num_classes', default=10)
        self.parse.add_argument('--num_filters', default=256)
        self.parse.add_argument('--kernel_size',  default=5)
        self.parse.add_argument('--vocab_size', default=5000)
        self.parse.add_argument('--hidden_dim', default=128)
        self.parse.add_argument('--dropout_keep_prob',  default=0.5)
        self.parse.add_argument('--learning_rate', default=1e-3)
        self.parse.add_argument('--batch_size', default=64)
        self.parse.add_argument('--num_epochs', default=10)
        self.parse.add_argument('--print_per_batch', default=100)
        self.parse.add_argument('--save_per_batch', default=10)