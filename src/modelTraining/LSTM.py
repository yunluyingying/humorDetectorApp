from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import re
import numpy as np
import random
import pandas as pd
import os
import sys
import math

# clean the raw text data
def preprocessor_cleantext(text):

    text = re.sub('\[\*\*[^\]]*\*\*\]','',text)             # remove all the information in [] inclusively
    text = re.sub('<[^>]*>','', text)                       # remove all the information in <> inclusively
    text = re.sub('[\W]+',' ',text.lower())                 # convert all the letters to lower case and replace non-word with space
    text = re.sub("\d+"," ", text)                          # replace all the digits with space

    return text

# create sequence for each text data
def createWordSequence(df, max_sequence_len = 200, inputCol = 'text'):

    texts = df[inputCol].apply(preprocessor_cleantext)      # import and clean the text column
    toke = Tokenizer()                                      # chop the text into pieces and throwing away certain characters, such as punctuation
    toke.fit_on_texts(texts)                                # store all the words in index
    sequence = toke.texts_to_sequences(texts)               # convert the texts to sequence

    ave_seq = [len(i) for i in sequence]
    df = pd.DataFrame(ave_seq)

    print ("Average text length is: {} ".format(1.0 * sum(ave_seq)/len(ave_seq)))

    word_index = toke.word_index                            # dictionary to store the {word: index}
    reverse_word_index = dict(zip(word_index.values(), word_index.keys()))    # reverse dictionary for easy search {index: word}
    print("Found {} unique tokens".format(len(word_index)))
    data = pad_sequences(sequence, maxlen = max_sequence_len)# trim the sequence to the length of max_sequence_len

    return data, word_index, reverse_word_index

# create embedding matrix for all the legal words in the texts
def create_EmbeddingMatrix (word_index, word2vec_path, remove_stopwords = False):

    glove_model = {}
    print("Loading GloVe Model")

    f = open(word2vec_path)
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        vector = np.asarray(splitLine[1:], dtype = 'float32')
        glove_model[word] = vector
    f.close()

    # print(glove_model)
    print("Found {} word vectors.".format(len(glove_model)))

    if remove_stopwords:
        keys_updated = [word for word in glove_model.keys() if word not in stopwords.words('English')]
        glove_model_set = set(keys_updated)
    else:
        glove_model_set = set(glove_model.keys())

    embedding_dim = len(glove_model["hello"])                   # dimension of the embedding vector
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    for word,i in word_index.items():                           # construct embedding matrix for all the legal words in the texts
        if word in glove_model_set:
            embedding_matrix[i] = glove_model.get(word)

    return embedding_matrix

# separate the samples into the group of train (60%), validation(20%) and test(20%)
def train_test_separator(seed, N):
    idx = list(range(N))
    random.seed(seed)
    random.shuffle(idx)
    idx_train = idx[0: int(N*0.60)]
    idx_val = idx[int(N*0.60):int(N*0.80)]
    idx_test = idx[int(N*0.80):N]

    return idx_train, idx_val, idx_test

# LSTM Model
def LSTM_MODEL(input_shape, output_shape, embedding_layer):
    print("Building Model")
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape, name = 'input_layer'))
    model.add(embedding_layer)
    model.add(LSTM(256, return_sequences=True))           # returns a sequence of vectors of dimension 256
    model.add(Dropout(0.5))                               # Overcoming the overfitting problem
    model.add(BatchNormalization())                       # Speed up the learning speed
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(output_shape,activation='sigmoid'))

    model.compile(loss = 'mean_squared_error', optimizer = 'rmsprop', metrics = ['mse'])
    model.summary()
    return model

# CNN model
def CNN_MODEL(input_shape, output_shape, embedding_layer):
    print("Building Model")
    model = Sequential()
    # model.add(InputLayer(input_shape=input_shape, name = 'input_layer'))
    model.add(embedding_layer)
    model.add(ZeroPadding1D(1, input_shape=input_shape))
    model.add(Conv1D(256, 3, activation="relu"))
    model.add(ZeroPadding1D(1))
    model.add(Conv1D(256, 3, activation="relu"))
    model.add(MaxPooling1D(2, strides=2))

    model.add(ZeroPadding1D(1))
    model.add(Conv1D(128, 3, activation="relu"))
    model.add(ZeroPadding1D(1))
    model.add(Conv1D(128, 3, activation="relu"))
    model.add(MaxPooling1D(2, strides=2))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mse'])
    model.summary()
    return model

# Train Model
def train(reverse_word_index, embedding_matrix, train_data, train_label, val_data, val_label,
          test_data, test_label, nb_epoch = 50, batch_size = 128, pre_train = False, model_name = "LSTM_MODEL",
          output_path = "",scorerange = 1):

    if(output_path == ""):
        print("Please assign the output path")
        exit(0)

    max_sequence_length = train_data.shape[1]
    vocabulary_size = len(reverse_word_index) +1
    embedding_dim = embedding_matrix.shape[1]
    input_shape = train_data.shape[1:]

    embedding_layer = Embedding(vocabulary_size,
                                embedding_dim,
                                weights = [embedding_matrix],
                                input_length=max_sequence_length,
                                trainable = False,
                                name = 'embedding_layer')

    model_fun = getattr(sys.modules[__name__],model_name)
    model = model_fun(input_shape,scorerange,embedding_layer)

    if not os.path.isdir('../../inputData/cache'):
        os.mkdir('../../inputData/cache')

    weight_name = 'weights_LSTM.h5'
    weight_path = os.path.join("../../inputData/cache", weight_name)

    if pre_train:
        model.load_weights(weight_path)

    print('checkpoint')
    checkpointer = ModelCheckpoint(filepath = weight_path, verbose = 1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose = 0, mode = 'auto')
    print('Early stop at 5')
    # fit the model
    History = model.fit(train_data, train_label,
              batch_size = batch_size,
              epochs = nb_epoch,
              validation_data = [val_data, val_label],
              callbacks = [checkpointer, earlystopping])
    print("done with model training")

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # summerize history for loss

    fig_loss = plt.figure()
    plt.plot(History.history['loss'], label='train')
    plt.plot(History.history['val_loss'], label='validation')
    plt.legend()
    plt.show()
    fig_loss.savefig(os.path.join(output_path,"loss.png"))

    prediction = model.predict(test_data,batch_size= 128,verbose = 1)
    print(prediction)

    rmse = math.sqrt(mean_squared_error(test_label, prediction))
    print('Test RMSE: %.3f' % rmse)

    model.save_weights(os.path.join(output_path, "weights_LSTM.h5"))

    print("--------------------model weights were saved successfully------------------------")

if __name__ == "__main__":
    # Input Parameters
    os.environ['KERAS_BACKEND'] = 'theano'

    model_name = "LSTM_MODEL"
    max_sequence_len = 200
    pre_train = False
    dataset_path = "../../outputData/train_data.csv"
    output_path = "../../outputData/LSTM"

    print("---Load the raw data---")
    df = pd.read_csv(dataset_path)

    print("---Output the word_index file---")
    data, word_index, reverse_word_index = createWordSequence(df,max_sequence_len = max_sequence_len)

    print("---Create train-validate-test file---")
    N = df.shape[0]
    seed = 1000
    idx_train, idx_val, idx_test = train_test_separator(seed,N)

    train_data = data[idx_train, :]
    train_label = df['score'].values[idx_train]
    # np.savetxt("../outputData/top20/train.txt",train_info, fmt = '%s', delimiter=',')

    val_data = data[idx_val, :]
    val_label = df['score'].values[idx_val]
    # np.savetxt("../outputData/top20/validation.txt",validation_info,fmt = '%s', delimiter=',')

    test_data = data[idx_test,:]
    test_label = df['score'].values[idx_test]
    #print(test_label)
    # np.savetxt("../outputData/top20/test.txt",test_info,fmt = '%s', delimiter=',')

    print("---Output the embedding matrix file---")
    # ../inputData/glove.840B.300d.txt
    em = create_EmbeddingMatrix(word_index, "../../inputData/glove.840B.300d.txt", remove_stopwords=True)

    print("---Train model---")

    train(reverse_word_index, em, train_data, train_label,
           val_data, val_label, test_data, test_label,nb_epoch = 5,
           pre_train = pre_train, model_name = model_name,output_path = output_path)


    print("---End---")