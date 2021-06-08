#! -*- coding: utf-8 -*-
# 百度LIC2020的事件抽取赛道，非官方baseline
# 直接用RoBERTa+CRF
# 在第一期测试集上能达到0.78的F1，优于官方baseline

import json
import numpy as np
from bert4keras.backend import keras, K, search_layer
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
import pylcs
from keras.layers import Input, Dense, Embedding, LSTM, Bidirectional,GRU
from keras.optimizers import Optimizer
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
# 基本信息
train=True #参数为True代表训练 参数为False代表为预测
maxlen = 220
epochs = 10
batch_size = 16
learning_rate = 1e-5
crf_lr_multiplier = 100  # 必要时扩大CRF层的学习率
model_save='best_model_v4.weights'  #最好的成绩是模型2
# bert配置
config_path = '/home/maxin/roberta_wwm/bert_config.json'
checkpoint_path = '/home/maxin/roberta_wwm/bert_model.ckpt'
dict_path = '/home/maxin/roberta_wwm/vocab.txt'
      

def load_data(filename):
    D = []
    with open(filename) as f:
        for l in f:
            l = json.loads(l)
            arguments = {}
            for event in l['event_list']:
                for argument in event['arguments']:
                    key = argument['argument']
                    value = (event['event_type'], argument['role'])
                    arguments[key] = value
            D.append((l['text'], arguments))
    print(D)
    return D


# 读取数据
train_data = load_data('../baidu_ee/duee_train.json')
valid_data = load_data('../baidu_ee/duee_dev.json')

# 读取schema
with open('../baidu_ee/duee_event_schema.json') as f:
    id2label, label2id, n = {}, {}, 0
    for l in f:
        l = json.loads(l)
        for role in l['role_list']:
            key = (l['event_type'], role['role'])
            id2label[n] = key
            label2id[key] = n
            n += 1
    num_labels = len(id2label) * 2 + 1

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, arguments) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            labels = [0] * len(token_ids)
            for argument in arguments.items():
                a_token_ids = tokenizer.encode(argument[0])[0][1:-1]
                start_index = search(a_token_ids, token_ids)
                if start_index != -1:
                    labels[start_index] = label2id[argument[1]] * 2 + 1
                    for i in range(1, len(a_token_ids)):
                        labels[start_index + i] = label2id[argument[1]] * 2 + 2
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            #print(batch_labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='roberta'
)
#lstm_output = Bidirectional(LSTM(num_labels//2, dropout=0.2, return_sequences=True))(model.output)
output = Dense(num_labels)(model.output)
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)

model = Model(model.input, output)
model.summary()
opt =Adam(learning_rate)  #AccumOptimizer(Adam(learning_rate), 5) # 10是累积步数
#opt = tf.keras.optimizers.Adam(learning_rate)
#opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(
            #opt,
            #loss_scale='dynamic')	
	
model.compile(
    loss=CRF.sparse_loss,
    optimizer=opt,
    metrics=[CRF.sparse_accuracy]
)


def viterbi_decode(nodes, trans):
    """Viterbi算法求最优路径
    其中nodes.shape=[seq_len, num_labels],
        trans.shape=[num_labels, num_labels].
    """
    labels = np.arange(num_labels).reshape((1, -1))
    scores = nodes[0].reshape((-1, 1))
    scores[1:] -= np.inf  # 第一个标签必然是0
    paths = labels
    for l in range(1, len(nodes)):
        M = scores + trans + nodes[l].reshape((1, -1))
        idxs = M.argmax(0)
        scores = M.max(0).reshape((-1, 1))
        paths = np.concatenate([paths[:, idxs], labels], 0)
    return paths[:, scores[:, 0].argmax()]


def extract_arguments(text):
    """arguments抽取函数
    """
    tokens = tokenizer.tokenize(text)
    while len(tokens) > 510:
        tokens.pop(-2)
    mapping = tokenizer.rematch(text, tokens)
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    nodes = model.predict([[token_ids], [segment_ids]])[0]
    trans = K.eval(CRF.trans)
    labels = viterbi_decode(nodes, trans)
    arguments, starting = [], False
    for i, label in enumerate(labels):
        if label > 0:
            if label % 2 == 1:
                starting = True
                arguments.append([[i], id2label[(label - 1) // 2]])
            elif starting:
                arguments[-1][0].append(i)
            else:
                starting = False
        else:
            starting = False

    return {
        text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1]: l
        for w, l in arguments
    }


def evaluate(data):
    """评测函数（跟官方评测结果不一定相同，但很接近）
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for text, arguments in tqdm(data):
        inv_arguments = {v: k for k, v in arguments.items()}
        pred_arguments = extract_arguments(text)
        pred_inv_arguments = {v: k for k, v in pred_arguments.items()}
        Y += len(pred_inv_arguments)
        Z += len(inv_arguments)
        for k, v in pred_inv_arguments.items():
            if k in inv_arguments:
                # 用最长公共子串作为匹配程度度量
                l = pylcs.lcs(v, inv_arguments[k])
                X += 2. * l / (len(v) + len(inv_arguments[k]))
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


def predict_to_file(in_file, out_file):
    """预测结果到文件，方便提交
    """
    fw = open(out_file, 'w', encoding='utf-8')
    with open("/home/maxin/baidu_data/baidu_ee/event_schema.json", 'r', encoding='UTF-8') as x:
            event_dict = json.load(x)
    x.close()
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            arguments = extract_arguments(l['text'])
            event_list = []
            for k, v in arguments.items():
                event_list.append({
                    'event_type': v[0],
                    'trigger':v[0].split('-')[-1],
                    'trigger_start_index':0,
                    'arguments': [{
                        'argument_start_index':0,
                        'role': v[1],
                        'argument': k,
                        'class':event_dict.get(v[0])
                    }]
                })
            l['event_list'] = event_list
            l = json.dumps(l, ensure_ascii=False)
            fw.write(l + '\n')
    fw.close()


class Evaluator(keras.callbacks.Callback):
    """评估和保存模型
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights(model_save)
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


# 写好函数后，启用对抗训练只需要一行代码
#adversarial_training(model, 'Embedding-Token', 0.5)
if __name__ == '__main__':
    if train==True:
        train_generator = data_generator(train_data, batch_size)
        dev_generator = data_generator(valid_data, batch_size)
        evaluator = Evaluator()
        #model.load_weights('best_model_v2.weights')
        model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=[evaluator]
        )
        #model.load_weights(model_save)
        #model.fit_generator(
           # dev_generator.forfit(),
            #steps_per_epoch=len(dev_generator),
            #epochs=10,
        #)
        #model.save(model_save)
    else:
        model.load_weights(model_save)
        predict_to_file('/home/maxin/baidu_data/baidu_ee/duee_test1.json', 'duee.json')


else:

    model.load_weights(model_save)
    # predict_to_file('/root/baidu/datasets/ee/test1_data/test1.json', 'ee_pred.json')
