import os
import time
from datetime import timedelta

import tensorflow as tf
from sklearn import metrics

from my_cnn_model import TextCNN
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab
from config import Config


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(sess, x_, y_, model, loss, acc):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.keep_prob: 1.0}
        loss_, acc_ = sess.run([loss, acc], feed_dict=feed_dict)
        total_loss += loss_ * batch_len
        total_acc += acc_ * batch_len

    return total_loss / data_len, total_acc / data_len


def train(model, file_config):
    print(file_config)
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False

    # my change
    config = model.config

    # 模型计算
    y_, logit = model.inference(model.input_x)
    loss = model.loss(logit, model.input_y)
    train_op = model.train(loss)
    acc = model.accuracy(y_)

    # tensorboard定义
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", acc)
    merged_summary = tf.summary.merge_all()
    # tensorboard文件写入对象
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(file_config.save_path):
        os.makedirs(file_config.save_path)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    # 读取训练和测试数据，TODO
    x_train, y_train = process_file(file_config.train_path, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(file_config.val_path, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    # 将图写入tensorboard
    writer.add_graph(session.graph)

    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        # 每一个batch重新定义一个迭代器
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.keep_prob: config.dropout_keep_prob}

            # tensorboard写入
            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([loss, acc], feed_dict=feed_dict)
                # TODO
                loss_val, acc_val = evaluate(session, x_val, y_val, model, loss, acc)

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=file_config.save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(train_op, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


if __name__ == '__main__':
    model = TextCNN()
    config = model.config
    file_config, _ = Config().parse.parse_known_args()
    print('Configuring CNN model...')
    if not os.path.exists(file_config.vocab_path):  # 如果不存在词汇表，重建
        build_vocab(file_config.train_path, file_config.vocab_path, config.vocab_size)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(file_config.vocab_path)
    config.vocab_size = len(words)
    train(model, file_config)



