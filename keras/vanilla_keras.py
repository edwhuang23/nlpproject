# sections_start for when section starts
# segment_loudness_max or segment_loudness_start for loudness of segment
# segment_start, segment_timbre, and segment_pitches are important

# Think about if a song has rock, classical, and rock, and that particular song is indicative of a rock song

# segments_pitches for the 12 pitches of segment
# segments_start to do the weighted sum of how long each segment is
# Round to tenths for segments_pitch to decrease space

# Train model on each segment and the correct genre

# For testing we will get the probability distribution for one segment and then select the genre with highest probability

# Get the genre distribution for each segment


import h5py
import pickle, os, argparse, random
from numba import jit, cuda
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Masking
from keras.callbacks import EarlyStopping


unka = [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ]
UNKA = '[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]'

def train():
    # Load data from genre.txt
    genre_dict = {} # Filename: genre
    genre_file = '../genre.txt'

    f = open(genre_file, 'r', encoding="ISO-8859-1")
    genre = f.read().splitlines()
    f.close()

    for line in genre:
        line = line.split()
        key = line[0].strip()
        value = line[1].strip()
        genre_dict[key] = value

    # Open each song file and obtain the segments_pitches for the song and organize
    # song:[ section:[[12 chroma features], [], [], ... ], ... ]
    songDir = '../songs/training/'
    
    vocab_indexed = pickle.load( open('vocab-4.0_limited_pr.pickle', "rb") )
    
    # we want to keep the 0 index free for the padding indicator
    for seg in vocab_indexed.keys():
        vocab_indexed[seg] += 1
    
    genres_indexed = pickle.load( open("genres_limited_pr.pickle", "rb") )
    trainingText = pickle.load( open("training_mod-4.0_limited_pr.pickle", "rb") )

    # This code is for combining the sections into one big list of segments (as indexed by the vocab)
    # print('COMBINING SECTIONS NOW')
    # X_train = []
    # Y_train = []
    # progress = 0
    # MAX_LENGTH = 0
    # for song in trainingText.keys():
        # if progress % 50 == 0: print(progress, 'songs combined')
        # seg_list = []
        # for section in trainingText[song]:
            # segments_indcs = [vocab_indexed[str(segment)] for segment in section]
            # seg_list = seg_list + segments_indcs
        # if len(seg_list) > MAX_LENGTH : MAX_LENGTH = len(seg_list)
        # X_train.append(seg_list)
        # Y_train.append(genres_indexed[genre_dict[song[:-3]]])
        # progress += 1
    
    # save these lists for future use
    # print('SAVING TRAIN COMBOS NOW')
    # X_train_pickle = open('X_train_limited_pr.pickle', 'wb')
    # pickle.dump(X_train, X_train_pickle)
    # X_train_pickle.close()
    
    # Y_train_pickle = open('Y_train_limited_pr.pickle', 'wb')
    # pickle.dump(Y_train, Y_train_pickle)
    # Y_train_pickle.close()
    
    # load combos
    print('LOADING COMBOS NOW')
    X_train = pickle.load( open("X_train_limited_pr.pickle", "rb") )
    Y_train = pickle.load( open("Y_train_limited_pr.pickle", "rb") )
    MAX_LENGTH = 4459
    PAD_idx = 0
    
    X_train_padded = pad_sequences(X_train, maxlen=MAX_LENGTH, padding='post', value=PAD_idx)
    
    X_test = pickle.load( open("X_test_limited_pr.pickle", "rb") )
    X_test_padded = pad_sequences(X_test, maxlen=MAX_LENGTH, padding='post', value=PAD_idx)
    Y_test = pickle.load( open("Y_test_limited_pr.pickle", "rb") )
    
    output = open('model_accuracies.txt', 'a')
    # Create RNN model
    print('LOADING MODEL NOW')
    EMBEDDING_DIM = 100
    MAX_NB_WORDS = len(vocab_indexed) + 1 # "+ 1" to account for padding indicator
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, mask_zero=True, input_length=X_train_padded.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(len(genres_indexed), activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
       
    print('ABOUT TO TRAIN')
    NUM_EPOCHS = 30
    BATCH_SIZE = 32
    for epoch in range(NUM_EPOCHS):
    # history = model.fit(X_train, Y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
        model.fit(np.array(X_train_padded), np.array(Y_train), epochs=1, batch_size=BATCH_SIZE, shuffle=True)

        # save the model
        print('SAVING MODEL NOW')
        model.save('keras_model_limited_pr-' + str(epoch+1) + 'e/')
        
        accr = model.evaluate(np.array(X_test_padded),np.array(Y_test))
        output.write('Results on test set after ' + str(epoch+1) + ' epochs:\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}\n\n'.format(accr[0],accr[1]))
        print('Results on test set after ' + str(epoch+1) + ' epochs:\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    # test the model
    print('TESTING MODEL NOW')
    #testingText = pickle.load( open("../testing_mod-4.0.pickle", "rb") )
    # X_test = []
    # Y_test = []
    # X_test = pickle.load( open("X_test_limited_pr.pickle", "rb") )
    # X_test_padded = pad_sequences(X_test, maxlen=MAX_LENGTH, padding='post', value=PAD_idx)
    # Y_test = pickle.load( open("Y_test_limited_pr.pickle", "rb") )
    
        
    # The following code preprocesses and combines the section-partitioned testing songs into a list of segments (as indexed by the vocab)
    # It also genereates a precidtion for each song according to the model
    # for song in testingText.keys():
        # seg_list = []
        # for section in testingText[song]:
            # segments_indcs = [vocab_indexed[str(segment)] if str(segment) in vocab_indexed else vocab_indexed[UNKA] for segment in section]
            # seg_list = seg_list + segments_indcs
        # X_test.append(seg_list)
        # if len(seg_list) > MAX_LENGTH:
            # print(len(seg_list))
            # MAX_LENGTH = len(seg_list)
        # #seg_list_padded = pad_sequences([seg_list], maxlen=MAX_LENGTH, padding='post', value=PAD_idx)
        # # print(seg_list[0])
        # #X_test_padded.append(seg_list_padded)
        # # pred_dist = model.predict(np.array(seg_list_padded))
        # truth = genres_indexed[genre_dict[song[:-3]]]
        # Y_test.append(truth)
        # # pred = np.argmax(pred_dist)
        # # pred_list.append(pred)
        # # print(pred)
        # if progress % 50 == 0:
            # print(progress, 'songs tested')
            # # print(pred_dist, pred, truth)
        # progress += 1

    # use the following code to see the prediciton for each song
    pred_list = []
    progress = 0
    
    for i in range(len(X_test)):
        song = X_test_padded[i]
        pred_dist = model.predict(np.array([song]))
        truth = Y_test[i]
        pred = np.argmax(pred_dist)
        pred_list.append(pred)
        print(pred, truth)
        if i % 10 == 0:
            print(i, 'songs tested')

    # save these lists for future use
    # print('SAVING TEST COMBOS NOW')
    # X_test_pickle = open('X_test_limited_pr.pickle', 'wb')
    # pickle.dump(X_test, X_test_pickle)
    # X_test_pickle.close()
    
    # Y_test_pickle = open('Y_test_limited_pr.pickle', 'wb')
    # pickle.dump(Y_test, Y_test_pickle)
    # Y_test_pickle.close()
    
    print("The accuracy of the model is %6.2f%%" % (accuracy_score(Y_test, pred_list) * 100))
    
    # Use this code to evaluate the model very quickly, but you will nto be shown any individual predictions
    # print('TESING WITH model.evaluate NOW')
    # accr = model.evaluate(np.array(X_test_padded),np.array(Y_test))
    # print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    return

def test():
    # read in genre dictionary from genre.txt
    genre_dict = {}
    genre_file = '../genre.txt'

    f = open(genre_file, 'r', encoding="ISO-8859-1")
    genre = f.read().splitlines()
    f.close()

    for line in genre:
        line = line.split()
        key = line[0].strip()
        value = line[1].strip()
        genre_dict[key] = value
    
    # song:[ section:[[12 chroma features], [], [], ... ], ... ]
    songDir = '../songs/testing/'
    
    vocab_indexed = pickle.load( open('vocab-4.0_limited_pr.pickle', "rb") )
    
    for seg in vocab_indexed.keys():
        vocab_indexed[seg] += 1
    # vocab_pickle = open('vocab_indexed_limited_pr.pickle', 'wb')
    # pickle.dump(vocab_indexed, vocab_pickle)
    # vocab_pickle.close()
    
    genres_indexed = pickle.load( open("genres_limited_pr.pickle", "rb") )
    trainingText = pickle.load( open("training_mod-4.0_limited_pr.pickle", "rb") )

    # Use the following code to combine the sections of a song into a big list of segments
    # print('COMBINING SECTIONS NOW')
    # X_train = []
    # Y_train = []
    # progress = 0
    # MAX_LENGTH = 0
    # for song in trainingText.keys():
        # if progress % 50 == 0: print(progress, 'songs combined')
        # seg_list = []
        # for section in trainingText[song]:
            # segments_indcs = [vocab_indexed[str(segment)] for segment in section]
            # seg_list = seg_list + segments_indcs
        # if len(seg_list) > MAX_LENGTH : MAX_LENGTH = len(seg_list)
        # X_train.append(seg_list)
        # Y_train.append(genres_indexed[genre_dict[song[:-3]]])
        # progress += 1
    
    # save these for future use
    # print('SAVING TRAIN COMBOS NOW')
    # X_train_pickle = open('X_train_limited_pr.pickle', 'wb')
    # pickle.dump(X_train, X_train_pickle)
    # X_train_pickle.close()
    
    # Y_train_pickle = open('Y_train_limited_pr.pickle', 'wb')
    # pickle.dump(Y_train, Y_train_pickle)
    # Y_train_pickle.close()
    
    # load combos
    print('LOADING COMBOS NOW')
    X_train = pickle.load( open("X_train_limited_pr.pickle", "rb") )
    Y_train = pickle.load( open("Y_train_limited_pr.pickle", "rb") )
    MAX_LENGTH = 4459
    PAD_idx = 0
    
    X_train_padded = pad_sequences(X_train, maxlen=MAX_LENGTH, padding='post', value=PAD_idx)
    
    X_test = pickle.load( open("X_test_limited_pr.pickle", "rb") )
    X_test_padded = pad_sequences(X_test, maxlen=MAX_LENGTH, padding='post', value=PAD_idx)
    
    Y_test = pickle.load( open("Y_test_limited_pr.pickle", "rb") )
    
    output = open('model_accuracies.txt', 'a')
    
    print('ABOUT TO TEST')
    NUM_EPOCHS = 30
    BATCH_SIZE = 32
    for epoch in range(NUM_EPOCHS):
        print('LOADING MODEL NOW')
        model = load_model('keras_model_limited_pr-' + str(epoch+1) + 'e/')
        
        print('TESTING MODEL NOW')
        accr = model.evaluate(np.array(X_train_padded),np.array(Y_train))
        output.write('Results on train set after ' + str(epoch+1) + ' epochs:\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}\n\n'.format(accr[0],accr[1]))
        print('Results on train set after ' + str(epoch+1) + ' epochs:\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
        
        accr = model.evaluate(np.array(X_test_padded),np.array(Y_test))
        output.write('Results on test set after ' + str(epoch+1) + ' epochs:\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}\n\n'.format(accr[0],accr[1]))
        print('Results on test set after ' + str(epoch+1) + ' epochs:\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    return


def main(params):
    if params.train:
        if params.gpu:
            train_func = (jit)(train)
            train_func()
        else:
            train()
    else:
        if params.gpu:
            test_func = (jit)(test)
            test_func()
        else:
            test()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vanilla_keras")
    parser.add_argument("--train", action='store_const', const=True, default=False)
    parser.add_argument("--gpu", action='store_const', const=True, default=False)
    main(parser.parse_args())