import json
import pandas as pd
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow import keras

def preprocess(x, y, padding_size = 128, oov_token="<UNK>", vocab_file = "vocab.json", train_ratio=0.7) :  
  preprocessor = preprocessing.text.Tokenizer(oov_token=oov_token) #토큰화
  preprocessor.fit_on_texts(x)
  x = preprocessor.texts_to_sequences(x) #시퀀스로 변환
  vocab = preprocessor.word_index #단어:인덱스 dictionary
  json.dump(vocab, open(vocab_file, 'w'), ensure_ascii=False)
  x = preprocessing.sequence.pad_sequences(x, maxlen=padding_size, padding='post', truncating='post')

  index = int(len(x)*train_ratio)

  return x[:index], y[:index], x[index:], y[index:], vocab


def CNNforText( num_classes,  #클래스 갯수
          vocab_size,
          embed_size = 512, #단어 임베딩 사이즈                 
          filter_sizes = [3,4,5],
          regularizers_lambda = 0.01, #L2 regulation parameter
          dropout =  0.5,
          feature_size = 128, #문장 시퀀스 길이
          num_filters = 128 #필터 개수 (필터사이즈와 같음). mhlee 하나로 통일하자
) :
          

  inputs = keras.Input(shape=(feature_size,), name='input_data')
  embed_initer = keras.initializers.RandomUniform(minval=-1, maxval=1)
  #sequence 임베딩
  embed = keras.layers.Embedding(vocab_size, embed_size,
                                  embeddings_initializer=embed_initer,
                                  input_length=feature_size,
                                  name='embedding')(inputs)
                                  
  embed = keras.layers.Reshape((feature_size, embed_size, 1), name='add_channel')(embed)

  pool_outputs = []

  #filter 별로 모델 구성
  for filter_size in filter_sizes :
    #convolution
    filter_shape = (filter_size, embed_size)
    conv = keras.layers.Conv2D(num_filters, filter_shape, strides=(1, 1), padding='valid',
                                data_format='channels_last', activation='relu',
                                kernel_initializer='glorot_normal',
                                bias_initializer=keras.initializers.constant(0.1),
                                name='convolution_{:d}'.format(filter_size))(embed)
    #max pooling
    max_pool_shape = (feature_size - filter_size + 1, 1)
    pool = keras.layers.MaxPool2D(pool_size=max_pool_shape,
                                  strides=(1, 1), padding='valid',
                                  data_format='channels_last',
                                  name='max_pooling_{:d}'.format(filter_size))(conv)
    pool_outputs.append(pool)

  pool_outputs = keras.layers.concatenate(pool_outputs, axis=-1, name='concatenate')
  pool_outputs = keras.layers.Flatten(data_format='channels_last', name='flatten')(pool_outputs)
  pool_outputs = keras.layers.Dropout(dropout, name='dropout')(pool_outputs)

  outputs = keras.layers.Dense(num_classes, activation='softmax',
                                kernel_initializer='glorot_normal',
                                bias_initializer=keras.initializers.constant(0.1),
                                kernel_regularizer=keras.regularizers.l2(regularizers_lambda),
                                bias_regularizer=keras.regularizers.l2(regularizers_lambda),
                                name='dense')(pool_outputs)
  model = keras.Model(inputs=inputs, outputs=outputs)
  model.summary()
  return model, num_classes

import os
import time
import tensorflow as tf

def train(model, x_train, y_train, num_classes
          , batch_size = 64, epochs = 1, fraction_validation = 0.05, results_dir = "./result/", save_path = "model") :
  timestamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
  path = os.path.join(results_dir, timestamp)
  if not os.path.exists(path) :    
    path_log = os.path.join(path, 'log/')
    os.makedirs(path_log)

  model.compile(tf.optimizers.Adam(), loss='categorical_crossentropy',metrics=['accuracy'])
  #모델 구조 이미지 파일로 저장
  #keras.utils.plot_model(model, show_shapes=True, to_file=os.path.join(path, "model.jpg"))
  y_train = tf.one_hot(y_train, num_classes)
  tb_callback = keras.callbacks.TensorBoard(path_log,
                                            histogram_freq=0.1, write_graph=True,
                                            write_images=True,
                                            embeddings_freq=0.5, update_freq='batch')

  history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                                callbacks=[tb_callback], validation_split=fraction_validation, shuffle=True)
  
  #모델 저장
  keras.models.save_model(model, save_path)
  print(history.history)

  return model, path_log

def test(model, x_test, y_test, num_classes):
    y_pred_one_hot = model.predict(x=x_test, batch_size=1, verbose=1)
    y_pred = tf.math.argmax(y_pred_one_hot, axis=1)

    print('\nTest accuracy: {}\n'.format(accuracy_score(y_test, y_pred)))
    print('Classification report:')
    target_names = ['class {:d}'.format(i) for i in np.arange(num_classes)]
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

if __name__ == "__main__" :
    df = pd.read_csv("./ratings.txt",sep='\t').dropna()
    df.head(5)
    x_train, y_train, x_test, y_test, vocab = preprocess(df["document"].tolist(), df["label"].tolist())
    model, num_classes = CNNforText(len(np.unique(y_train)), len(vocab))
    model, path_log = train(model, x_train, y_train, num_classes, epochs=1)
    test(model, x_test[:50000], y_test[:50000], num_classes)

