import myutils
import sys
import os.path
import json
from datetime import datetime
import random
import numpy
from keras.layers import Bidirectional, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from keras.layers import LSTM
from keras.preprocessing import sequence
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Permute, Dense, Multiply, Lambda, RepeatVector
import tensorflow.keras.backend as K

numpy.set_printoptions(suppress=True)
numpy.set_printoptions(threshold=numpy.inf)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# dwce
def dynamic_weighted_binary_crossentropy(y_true, y_pred):
    batch_size = tf.cast(tf.size(y_true), tf.float32)
    positive_weight = tf.reduce_sum(y_true) / batch_size
    negative_weight = 1.0 - positive_weight
    bce = BinaryCrossentropy(from_logits=False)
    loss = positive_weight * bce(y_true, y_pred) + negative_weight * bce(y_true, y_pred)
    return loss

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

progress = 0
count = 0
step = 5
fulllength = 200
mode2 = str(step) + "_" + str(fulllength)

mincount = 5
iterationen = 100
s = 300
w = "withString"

w2vmodel = "CodeXGLUE_word2vec_5-100-300.model"

if not (os.path.isfile(w2vmodel)):
    print("word2vec model is still being created...")
    sys.exit()

w2v_model = Word2Vec.load(w2vmodel)
word_vectors = w2v_model.wv

allblocks = []
with open('../dataset/function', 'r') as infile:
    data = json.load(infile)

now = datetime.now()
nowformat = now.strftime("%H:%M")
print("finished loading. ", nowformat)

node2vec_model = Word2Vec.load("node2vec_model.model")
node_vectors = node2vec_model.wv
print(node_vectors.key_to_index)

common_tokens = set(word_vectors.key_to_index.keys()).intersection(set(node_vectors.key_to_index.keys()))

word_vectors_common = []
node_vectors_common = []

for token in common_tokens:
    word_vectors_common.append(word_vectors[token])
    node_vectors_common.append(node_vectors[token])

word_vectors_common_array = np.array(word_vectors_common)
node_vectors_common_array = np.array(node_vectors_common)


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights

multihead_attn = MultiHeadAttention(num_heads=10, d_model=300)
X, _ = multihead_attn(word_vectors_common_array, node_vectors_common_array, node_vectors_common_array, mask=None)

merged_vector = tf.reduce_sum(X, axis=1)
# sum
# merged_vector = np.sum(np.array(word_vectors_common), axis=0) + np.sum(np.array(node_vectors_common), axis=0)
# max
# merged_vector = np.maximum(np.array(word_vectors_common), np.array(node_vectors_common))
# mean
# merged_vector = (np.array(word_vectors_common) + np.array(node_vectors_common)) / 2
#bilstm
# combined_features = np.concatenate((word_vectors_common_array, node_vectors_common_array), axis=-1)  # 形状为 (n, 600)
# combined_features = combined_features.reshape((combined_features.shape[0], combined_features.shape[1], 1))  # 形状为 (n, 600, 1)
# model = Sequential()
# model.add(Bidirectional(LSTM(150, return_sequences=False), input_shape=(combined_features.shape[1], combined_features.shape[2])))
# model.add(Dense(300, activation='relu'))
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# model.fit(combined_features, np.random.rand(combined_features.shape[0], 300), epochs=10, batch_size=32)
# merged_vector = model.predict(combined_features)
# print("Merged Vector Shape:", merged_vector.shape)

print(merged_vector)
print(len(merged_vector))

merged_vector_np = merged_vector

def get_token_vector(token, common_tokens, merged_vector_np):
    if token in common_tokens:
        token_index = list(common_tokens).index(token)
        return merged_vector_np[token_index]
    else:
        return None

if_vector = get_token_vector("if", common_tokens, merged_vector_np)
print("Vector for 'if':", if_vector)
print("Shape of the vector for 'if':", if_vector.shape if if_vector is not None else "Token not found")


for r in data:
    progress = progress + 1
    b = []
    b.append(r["func"])
    b.append(r["target"])
    allblocks.append(b)

keys = []
for i in range(len(allblocks)):
    keys.append(i)
random.shuffle(keys)

cutoff = round(0.7 * len(keys))
cutoff2 = round(0.85 * len(keys))

print("cutoff " + str(cutoff))
print("cutoff2 " + str(cutoff2))

keystrain = keys[:cutoff]
keystest = keys[cutoff:cutoff2]
keysfinaltest = keys[cutoff2:]

keystrain_data = numpy.array(keystrain)
keystest_data = numpy.array(keystest)
keysfinaltest_data = numpy.array(keysfinaltest)

numpy.save('key_data/keystrain_data.npy', keystrain_data)
numpy.save('key_data/keystest_data.npy', keystest_data)
numpy.save('key_data/keysfinaltest_data.npy', keysfinaltest_data)

TrainX = []
TrainY = []
ValidateX = []
ValidateY = []
FinaltestX = []
FinaltestY = []


print("Creating training dataset... ")
for k in keystrain:
    block = allblocks[k]
    code = block[0]
    token = myutils.getTokens(code)
    vectorlist = []

    for t in token:
        if t in node_vectors.key_to_index and t in word_vectors.key_to_index and t != " ":
            combined_vector = get_token_vector(t, common_tokens, merged_vector_np)
            vectorlist.append(combined_vector.tolist())
    TrainX.append(vectorlist)
    TrainY.append(block[1])

print("Creating validation dataset...")
for k in keystest:
    block = allblocks[k]
    code = block[0]
    token = myutils.getTokens(code)
    vectorlist = []
    for t in token:
        if t in node_vectors.key_to_index and t in word_vectors.key_to_index and t != " ":
            combined_vector = get_token_vector(t, common_tokens, merged_vector_np)
            vectorlist.append(combined_vector.tolist())
    ValidateX.append(vectorlist)
    ValidateY.append(block[1])

print("Creating finaltest dataset...")
for k in keysfinaltest:
    block = allblocks[k]
    code = block[0]
    token = myutils.getTokens(code)
    vectorlist = []
    for t in token:
        if t in node_vectors.key_to_index and t in word_vectors.key_to_index and t != " ":
            combined_vector = get_token_vector(t, common_tokens, merged_vector_np)
            vectorlist.append(combined_vector.tolist())
    FinaltestX.append(vectorlist)
    FinaltestY.append(block[1])

print("Train length: " + str(len(TrainX)))
print("Test length: " + str(len(ValidateX)))
print("Finaltesting length: " + str(len(FinaltestX)))
now = datetime.now()
nowformat = now.strftime("%H:%M")
print("time: ", nowformat)

X_train = numpy.array(TrainX, dtype="object")
y_train = numpy.array(TrainY, dtype="object")
X_test = numpy.array(ValidateX, dtype="object")
y_test = numpy.array(ValidateY, dtype="object")
X_finaltest = numpy.array(FinaltestX, dtype="object")
y_finaltest = numpy.array(FinaltestY, dtype="object")

X_train = sequence.pad_sequences(X_train, maxlen=200)
X_test = sequence.pad_sequences(X_test, maxlen=200)
X_finaltest = sequence.pad_sequences(X_finaltest, maxlen=200)
X_train = numpy.asarray(X_train).astype(numpy.float32)
y_train= numpy.asarray(y_train).astype(numpy.float32)
X_test = numpy.asarray(X_test).astype(numpy.float32)
y_test= numpy.asarray(y_test).astype(numpy.float32)
X_finaltest = numpy.asarray(X_finaltest).astype(numpy.float32)
y_finaltest= numpy.asarray(y_finaltest).astype(numpy.float32)

now = datetime.now()
nowformat = now.strftime("%H:%M")
print("numpy array done. ", nowformat)

print(str(len(X_train)) + " samples in the training set.")
print(str(len(X_test)) + " samples in the validation set.")
print(str(len(X_finaltest)) + " samples in the final test set.")

csum = 0
for a in y_train:
    csum = csum + a
print("percentage of vulnerable samples: " + str(int((csum / len(X_train)) * 10000) / 100) + "%")

testvul = 0
for y in y_test:
    if y == 1:
        testvul = testvul + 1
print("absolute amount of vulnerable samples in test set: " + str(testvul))

max_length = fulllength

dropout = 0.4
neurons = 100
optimizer = "adam"
epochs = 50
batchsize = 1024

now = datetime.now()
nowformat = now.strftime("%H:%M")
print("Starting LSTM: ", nowformat)

print("Dropout: " + str(dropout))
print("Neurons: " + str(neurons))
print("Optimizer: " + optimizer)
print("Epochs: " + str(epochs))
print("Batch Size: " + str(batchsize))
print("max length: " + str(max_length))

model = Sequential()
model.add(Bidirectional(LSTM(neurons, return_sequences=True), input_shape=(200, 600)))  # around 50 seems good
model.add(Dropout(dropout))
model.add((LSTM(neurons, activation='tanh', return_sequences=True)))
model.add(Dropout(dropout))
model.add((LSTM(neurons, activation='tanh', return_sequences=True)))
model.add(Dropout(dropout))
model.add((LSTM(neurons, activation='tanh')))
model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))

now = datetime.now()
nowformat = now.strftime("%H:%M")
print("Compiled LSTM: ", nowformat)

class_weights = class_weight.compute_class_weight('balanced', classes=numpy.unique(y_train), y=y_train)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print(class_weights_dict)

# MSE
# model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy', myutils.precision, myutils.recall, myutils.f1])

# DWSE
model.compile(loss=dynamic_weighted_binary_crossentropy, optimizer=optimizer, metrics=['accuracy', myutils.precision, myutils.recall, myutils.f1])

# history = model.fit(X_train, y_train, epochs=epochs, batch_size=batchsize, class_weight=class_weights_dict)
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batchsize)

fname = "model.h5"
model.save(fname)

for dataset in ["train", "test", "finaltest"]:
    print("Now predicting on " + dataset + " set (" + str(dropout) + " dropout)")

    if dataset == "train":
        yhat = model.predict(X_train, verbose=0)
        yhat_classes = (yhat > 0.5).astype(int)
        accuracy = accuracy_score(y_train, yhat_classes)
        precision = precision_score(y_train, yhat_classes)
        recall = recall_score(y_train, yhat_classes)
        F1Score = f1_score(y_train, yhat_classes)
        mcc = matthews_corrcoef(y_train, yhat_classes)

    if dataset == "test":
        yhat = model.predict(X_test, verbose=0)
        yhat_classes = (yhat > 0.5).astype(int)
        accuracy = accuracy_score(y_test, yhat_classes)
        precision = precision_score(y_test, yhat_classes)
        recall = recall_score(y_test, yhat_classes)
        F1Score = f1_score(y_test, yhat_classes)
        mcc = matthews_corrcoef(y_test, yhat_classes)

    if dataset == "finaltest":
        yhat = model.predict(X_finaltest, verbose=0)
        yhat_classes = (yhat > 0.5).astype(int)
        accuracy = accuracy_score(y_finaltest, yhat_classes)
        precision = precision_score(y_finaltest, yhat_classes)
        recall = recall_score(y_finaltest, yhat_classes)
        F1Score = f1_score(y_finaltest, yhat_classes)
        mcc = matthews_corrcoef(y_finaltest, yhat_classes)

    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print('F1 score: %f' % F1Score)
    print('MCC: %f' % mcc)
    print("\n")

y_pred_lstm = model.predict(X_finaltest).ravel()
fpr_lstm, tpr_lstm, thresholds_lstm = roc_curve(y_finaltest, y_pred_lstm)
auc_lstm = auc(fpr_lstm, tpr_lstm)
print (auc_lstm)
