import tensorflow as tf
from config import Config


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self):
        self.config, _ = Config('cnn').parse.parse_known_args()
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')

    def inference(self, input_x):
        # embedding层
        with tf.variable_scope('embedding_layer'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedded = tf.nn.embedding_lookup(embedding, input_x)
        # 卷积层
        with tf.variable_scope('conv_layer'):
            # 使用tf.layers.conv1d原因 参考 https://www.imooc.com/article/details/id/32111
            conv = tf.layers.conv1d(embedded, self.config.num_filters, self.config.kernel_size, name='conv')
            # 最大池化层
            pool = tf.reduce_max(conv, reduction_indices=[1], name='pool')
        # 全连接层
        with tf.variable_scope('fc_layer1'):
            fc1 = tf.layers.dense(pool, self.config.hidden_dim, name='fc1')
            # dropout
            fc1 = tf.contrib.layers.dropout(fc1, self.keep_prob)
            fc1 = tf.nn.relu(fc1)
        # 全连接层
        with tf.variable_scope('fc_layer2'):
            logit = tf.layers.dense(fc1, self.config.num_classes, name='fc2')
        # softmax层
        with tf.variable_scope('softmax_layer'):
            y_ = tf.arg_max(tf.nn.softmax(logit), 1)
        return y_, logit

    def loss(self, logit, input_y):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=input_y))
        return loss

    def train(self, loss):
        opt = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(loss)
        return opt

    def accuracy(self, y_):
        correct_pred = tf.equal(tf.argmax(self.input_y, 1), y_)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return acc