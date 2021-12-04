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
import pickle, os, argparse
from numba import jit, cuda
from sklearn import preprocessing
#from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader

class RNNTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(RNNTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, dropout=0.0)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = self.softmax(tag_space)
        return tag_scores

def train(seg_comp_mode):
    # Enter genres into dictionary
    unka = [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ]
    UNKA = '[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]'
    # for i in range(len(unka)): unka[i] = round(unka[i] / 4.0, 1) * 10 #  Maybe 4.5 or (20/3) for 12*2^11 choices
    # UNKA = str(unka)
    print(UNKA)
    # input()
    genre_dict = {}
    genre_file = 'genre.txt'

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
    trainingText = {} # filename: [ section:[ segment:[12 chroma features], [], [], ... ], ... ]
    songDir = 'songs/training/'
    duration_info = { }

    for filename in os.listdir(songDir):
        # if filename[0:-3] not in genre_dict:
            # # Delete this file and remove key from genre_dict
            # print(filename)
            # os.remove(os.path.join(songDir, filename))

        with h5py.File(os.path.join(songDir, filename), "r") as f:
            # # List all groups
            # print("Keys:", str(f.keys()))
            # # analysis = list(f.keys())[0]
            print(filename)

            # # Get the data keys
            data = f['analysis']
            song_end = f['analysis']['songs'][()][0][3]
            duration_info[filename] = []
            # song_sections = []
            first_seg_idx = 0
            for sec_end in data['sections_start']:
                if sec_end == 0.0: continue
                segments_dur = []
                last_seg_idx = first_seg_idx
                while last_seg_idx < len(data['segments_start']):
                    if data['segments_start'][last_seg_idx] < sec_end : last_seg_idx += 1
                    else : break
                last_seg_idx -= 1
                i = first_seg_idx
                while i <= last_seg_idx :
                    seg_start = data['segments_start'][i]
                    if i+1 == len(data['segments_start']) : seg_end = song_end
                    else : seg_end = data['segments_start'][i+1]
                    seg_duration = seg_end - seg_start
                    segments_dur.append(seg_duration)
                    i += 1
                # section = data['segments_pitches'][first_seg_idx: last_seg_idx]
                # # print(section)
                duration_info[filename].append(segments_dur)
                # print(data['segments_start'][last_seg_idx], sec_end)
                # # input()
                # song_sections.append(section)
                first_seg_idx = last_seg_idx + 1
                # segmentCount += len(section)
                # if first_seg_idx >= len(data['segments_start']) : break
            # if first_seg_idx < len(data['segments_start']):
                # # now to grab the last section
                # section = data['segments_pitches'][first_seg_idx:]
                # # print(section)
                # # input()
                # song_sections.append(section)
                # segmentCount += len(section)
            # trainingText[filename] = song_sections
            # f.close()

    durations_pickle = open('train_durations.pickle', 'wb')
    pickle.dump(duration_info, durations_pickle)
    durations_pickle.close()
    
    return
    # segmentFrequency = {}

    # X = []
    # Y = []
    # maxSentenceLength = 0

    # Add PAD and UNKA index
    # vocab['<PAD>'] = len(vocab)
    # tags['<PAD>'] = len(tags)
    # vocab['UNKA'] = len(vocab)
    # tags['UNKA'] = len(tags)

    # filename: [ section:[[12 chroma features], [], [], ... ], ... ]
    # tags = pickle.load( open("tags.pickle", "rb") )
    # vocab = pickle.load( open("vocab-4.0.pickle", "rb") )
    
    # Get segmentFrequency for each segment
    # for song in trainingText.keys():
        # for section in trainingText[song]:
            # for i in range(len(section)):
                # # TODO: Round to tenths for segments_pitch to decrease space of possible segments
                # section[i] = [ round(chroma / 4.0, 1) for chroma in section[i] ] #  Maybe 4.5 or (20/3) for 12*2^11 choices
                # #print(section[i])
                # seg_str = str(section[i])
                # # print(section[i])
                # # print(seg_str)
                # # input()
                # if seg_str not in segmentFrequency: segmentFrequency[seg_str] = 0
                # segmentFrequency[seg_str] += 1

    # Now we can determine the vocabulary
    # vocab_indexed = { UNKA : 0 }
    # vocab_freq = { UNKA : 0 }
    # genres_indexed = {}
    # genres_freq = {}
    # for song in trainingText.keys():
        # genre = genre_dict[song[:-3]]
        # if genre not in genres_indexed:
            # genres_indexed[genre] = len(genres_indexed)
            # genres_freq[genre] = 0
        # genres_freq[genre] += 1
        # for section in trainingText[song]:
            # # sentence = []
            # for i in range(len(section)):
                # seg_str = str(section[i])
                # if segmentFrequency[seg_str] < 3:
                    # seg_str = UNKA
                    # section[i] = unka
                # if seg_str not in vocab_indexed:
                    # vocab_indexed[seg_str] = len(vocab_indexed)
                    # vocab_freq[seg_str] = 0
                # vocab_freq[seg_str] += 1

                # sentence.append(seg_str)

            # sentenceLength = len(sentence)
            # maxSentenceLength = max(maxSentenceLength, sentenceLength)
            # print(tagList)
            # X.append(segmentList)
            # Y.append(tagList)

    # Save vocab locally to vocab.pickle and tag locally to tag.pickle
    
    vocab_indexed = pickle.load( open("vocab-4.0.pickle", "rb") )
    vocab_freq = pickle.load( open("vocab_freq-4.0.pickle", "rb") )
    genres_indexed = pickle.load( open("genres.pickle", "rb") )
    genres_freq = pickle.load( open("genres_freq.pickle", "rb") )
    trainingText = pickle.load( open("sandbox/training_mod-4.0-1.pickle", "rb") )
    
    # vocab_pickle = open('vocab-4.0.pickle', 'wb')
    # pickle.dump(vocab_indexed, vocab_pickle)
    # vocab_pickle.close()
    
    # vocab_freq_pickle = open('vocab_freq-4.0.pickle', 'wb')
    # pickle.dump(vocab_freq, vocab_freq_pickle)
    # vocab_freq_pickle.close()

    # genres_pickle = open('genres.pickle', 'wb')
    # pickle.dump(genres_indexed, genres_pickle)
    # genres_pickle.close()
    
    # genres_freq_pickle = open('genres_freq.pickle', 'wb')
    # pickle.dump(genres_freq, genres_freq_pickle)
    # genres_freq_pickle.close()
    
    # training_texts = open('training_mod-4.0.pickle', 'wb')
    # pickle.dump(trainingText, training_texts)
    # training_texts.close()

    # return
    
    # Sentence padding
    # for i in range(len(X)):
        # while len(X[i]) < maxSentenceLength:
            # X[i].append('')
            # Y[i].append('')

    # Create RNN model
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 32
    model = RNNTagger(EMBEDDING_DIM, HIDDEN_DIM, len(vocab_indexed), len(genres_indexed))
    # error = nn.CrossEntropyLoss(ignore_index=0)

    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.NLLLoss()

    # Repeatedly train RNN model
    BATCH_SIZE = 1
    NUM_EPOCHS = 2

    # train_data = []

    # for i in range(len(X)):
        # for j in range(maxSentenceLength):
            # train_data.append([X[i][j], Y[i][j]])
    
    # train_dataloader = iter(DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True))
    
    for epoch in range(NUM_EPOCHS):
        print("Epoch", epoch + 1, "/", NUM_EPOCHS)
        for song in trainingText.keys():
            genre = genre_dict[song[:-3]]
            genre_idx = genres_indexed[genre]
            genre_tensor = torch.tensor(genre_idx, dtype=torch.long)
            for i in range(len(trainingText[song])) :
                section = trainingText[song][i]
                model.zero_grad()
                segments_indcs = [vocab_indexed[str(segment)] for segment in section]
                segments_tensor = torch.tensor(segments_indcs, dtype=torch.long)
                # tag_indexes = [tags_indexed[tag] for tag in tags]
                # tag_tensor = torch.tensor(tag_indexes, dtype=torch.long)
                genre_scores = model(segments_tensor)
                print(genre_scores)
                print(len(section))
                # print(genres_indexed)
                # input()
                dist_out = distribution_compiler(genre_scores, seg_comp_mode, genres_indexed, duration_info[song][i])
                print("Printing dist_out now:")
                print(dist_out)
                # input()
                loss = loss_function(dist_out, genre_tensor)
                loss.backward()
                optimizer.step()
        # print("Epoch", epoch + 1, "/", NUM_EPOCHS)
        # for i in range(0, len(X), BATCH_SIZE):
            # X_batch, Y_batch = next(train_dataloader)
            # optimizer.zero_grad()
            # le = preprocessing.LabelEncoder()
            # outputs = model(torch.as_tensor(le.fit_transform(X_batch)))
            # print(outputs)
            # # Put into loss function

            # loss = error(outputs, torch.as_tensor(le.fit_transform(Y_batch)))
            # loss.backward()
            # optimizer.step()
    
    model_filename = 'model_' + compilation_mode + '.torch'
    torch.save(model.state_dict(), model_filename)
    return

def distribution_compiler(genre_scores, compilation_mode, genres_indexed, sec_duration_info):
    softmax = nn.Softmax(dim=-1)
    if compilation_mode == 'PV':
        genre_counts = [ 0 for genre in genres_indexed ]
        num_segments = len(genre_scores)
        for seg_dist in genre_scores:
            highest_prob = 0.0
            most_likely_genre = -1
            for genre_idx in range(len(seg_dist)):
                genre_prob = seg_dist[genre_idx]
                if genre_prob > highest_prob:
                    highest_prob = genre_prob
                    most_likely_genre = genre_idx
            genre_counts[most_likely_genre] += 1
        #genre_counts_tensor = torch.tensor(genre_counts, dtype=torch.float)
        #print(genre_counts_tensor)
        #print(torch.nn.functional.normalize(genre_counts_tensor))
        genre_dist = [ count / num_segments for count in genre_counts ]
        
        # genre_counts_tensor = torch.tensor(genre_counts, dtype=torch.float)
        # norm = genre_counts_tensor.norm(p=2, dim=-1, keepdim=True)
        # gct_norm = genre_counts_tensor.div(num_segments)
        # print(genre_counts_tensor)
        # print(genre_dist)
        # print(gct_norm)
        
        # input()
        dist_tensor = torch.tensor(genre_dist, dtype=torch.float, requires_grad=True)
        # print(dist_tensor)
        # dist_tensor_sftmax = softmax(dist_tensor)
        # print(dist_tensor_sftmax)
        # input()
        # dist_tensor.grad_fn = genre_scores.grad_fn
        # print(dist_tensor)
        return dist_tensor
    
    if compilation_mode == 'WPV':
        sec_duration = 0.0
        for seg_dur in sec_duration_info:
            sec_duration += seg_dur
        genre_counts = [ 0.0 for genre in genres_indexed ]
        # 0.0 * len(genres_indexed)
        num_segments = len(genre_scores)
        for i in range(num_segments):
            seg_dist = genre_scores[i]
            highest_prob = 0.0
            most_likely_genre = -1
            for genre_idx in range(len(seg_dist)):
                genre_prob = seg_dist[genre_idx]
                if genre_prob > highest_prob:
                    highest_prob = genre_prob
                    most_likely_genre = genre_idx
            seg_weight = sec_duration_info[i] / sec_duration
            genre_counts[most_likely_genre] += seg_weight
        return torch.tensor(genre_counts, dtype=torch.float, requires_grad=True)
        
    if compilation_mode == 'GF':
        genre_counts = [ 0.0 for genre in genres_indexed ]
        num_segments = len(genre_scores)
        for seg_dist in genre_scores:
            for genre_idx in range(len(seg_dist)):
                genre_counts[genre_idx] += seg_dist[genre_idx]
        count_tensor = torch.tensor(genre_counts, dtype=float, requires_grad=True)
        return count_tensor.div(num_segments)
        
    if compilation_mode == 'WGF':
        sec_duration = 0.0
        for seg_dur in sec_duration_info:
            sec_duration += seg_dur
        genre_counts = [ 0.0 for genre in genres_indexed ]
        num_segments = len(genre_scores)
        for i in range(num_segments):
            seg_dist = genre_scores[i]
            seg_weight = sec_duration_info[i] / sec_duration
            for genre_idx in range(len(seg_dist)):
                genre_counts[genre_idx] += seg_dist[genre_idx] * seg_weight
        return torch.tensor(genre_counts, dtype=float, requires_grad=True)

def test(seg_comp_mode, sec_comp_mode):
    # Load test songs in same format as training songs
    testingText = {} # filename: [ section:[ segment:[12 chroma features], [], [], ... ], ... ]
    songDir = 'songs/testing/'
    duration_info = { }
    segmentCount = 0
    
    for filename in os.listdir(songDir):
        if '.h5' not in filename: continue
        print(filename)
        with h5py.File(os.path.join(songDir, filename), "r") as f:
            # Get the data keys
            data = f['analysis']
            if len(data['sections_start']) == 0: 
                f.close()
                continue
            song_end = f['analysis']['songs'][()][0][3]
            duration_info[filename] = [] # [ section 1: [segment 1 duration, ... ], ... ]
            song_sections = [] # [ section 1: [ segment data], ... ] ]
            first_seg_idx = 0
            for sec_end in data['sections_start']:
                if sec_end == 0.0: continue
                segments_dur = []
                last_seg_idx = first_seg_idx
                while last_seg_idx < len(data['segments_start']):
                    if data['segments_start'][last_seg_idx] < sec_end : last_seg_idx += 1
                    else : break
                last_seg_idx -= 1
                i = first_seg_idx
                while i <= last_seg_idx:
                    seg_start = data['segments_start'][i]
                    if i+1 == len(data['segments_start']) : seg_end = song_end
                    else : seg_end = data['segments_start'][i+1]
                    seg_duration = seg_end - seg_start
                    segments_dur.append(seg_duration)
                    i += 1
                section = data['segments_pitches'][first_seg_idx:last_seg_idx]

                if len(section) > 0:
                    song_sections.append(section)
                    segmentCount += len(section)
                    duration_info[filename].append(segments_dur)
 
                first_seg_idx = last_seg_idx + 1
                if first_seg_idx >= len(data['segments_start']) : break
            if first_seg_idx < len(data['segments_start']):
                # now to grab the last section
                i = first_seg_idx
                while i < len(data['segments_start']):
                    seg_start = data['segments_start'][i]
                    if i+1 == len(data['segments_start']) : seg_end = song_end
                    else : seg_end = data['segments_start'][i+1]
                    seg_duration = seg_end - seg_start
                    segments_dur.append(seg_duration)
                    i += 1
                section = data['segments_pitches'][first_seg_idx:]

                if len(section) > 0:
                    song_sections.append(section)
                    segmentCount += len(section)
                    duration_info[filename].append(segments_dur)
            testingText[filename] = song_sections
            f.close()

    print(segmentCount)

    # Dump testing_durations
    durations_pickle = open('testing_durations.pickle', 'wb')
    pickle.dump(duration_info, durations_pickle)
    durations_pickle.close()

    # Load vocab dictionary
    vocabPickle = open("vocab-4.0.pickle", "rb")
    vocab_indexed = pickle.load(vocabPickle)
    vocabPickle.close()

    unka = [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ]
    UNKA = '[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]'

    # Replace with UNKA if not in vocab
    for song in testingText.keys():
        for section in testingText[song]:
            for i in range(len(section)):
                # Round to tenths for segments_pitch to decrease space of possible segments
                section[i] = [ round(chroma / 4.0, 1) for chroma in section[i] ] #  Maybe 4.5 or (20/3) for 12*2^11 choices
                seg_str = str(section[i])
                if seg_str not in vocab_indexed:
                    seg_str = UNKA
                    section[i] = unka

    # Dump testing_mod
    testing_mod = open('testing_mod-4.0.pickle', 'wb')
    pickle.dump(testingText, testing_mod)
    testing_mod.close()

    return

    # # Load testing_mod
    # testing_mod = open('testing_mod-4.0.pickle', "rb")
    # test_mod = pickle.load(testing_mod)
    # testing_mod.close()
    # #print(test_mod)

    # # Load testing_durations
    # testing_durations = open('testing_durations.pickle', 'rb')
    # duration_info = pickle.load(testing_durations)
    # testing_durations.close()

    # songDir = 'songs/testing/'

    # for filename in os.listdir(songDir):
    #     with h5py.File(os.path.join(songDir, filename), "r") as f:
    #         # Get the data keys
    #         data = f['analysis']
    #         if len(duration_info[filename]) == len(data['sections_start']) : continue
    #         print(filename)
    #         print(len(duration_info[filename]))
    #         print(len(data['sections_start']))
    #         print(data['sections_start'][()])
    #         print(data['segments_start'][()])

    return

    # Load genres
    genrePickle = open("genres.pickle", "rb")
    genres_indexed = pickle.load(genrePickle)
    genrePickle.close()


    # Load tag2idc dictionary
    tagsPickle = open('genres.pickle', 'rb')
    tags = pickle.load(tagsPickle)
    tagsPickle.close()

    #249
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 32
    model = RNNTagger(EMBEDDING_DIM, HIDDEN_DIM, len(vocab), len(tags))
    model.load_state_dict(torch.load('model.torch'))

    # Obtain genres
    genre_dict = {}
    genre_file = 'genre.txt'

    f = open(genre_file, 'r', encoding="ISO-8859-1")
    genre = f.read().splitlines()
    f.close()

    for line in genre:
        line = line.split()
        key = line[0].strip()
        value = line[1].strip()
        genre_dict[key] = value

    # Convert word to indices
    X_test_expanded = []
    Y_test = []
    num_sections = []

    for song in testingText.keys():
        count = 0
        for section in testingText[song]:
            count += 1
            for segment in section:
                # Round segments to tenths
                for i in range(len(segment)):
                    segment[i] = round(segment[i] / 4.0, 1) #  Maybe 4.5 or (20/3) for 12*2^11 choices

                segment_hashed = str(segment)
                if segment_hashed not in vocab:
                    X_test_expanded.append(vocab['UNKA'])
                else:
                    X_test_expanded.append(vocab[segment_hashed])
        num_sections.append(count)
        Y_test.append(genre_dict[song[:-3]])

    X_test_expanded_pred = model(torch.tensor(X_test_expanded))
    X_test_expanded_pred = torch.argmax(X_test_expanded_pred,-1).cpu().numpy()

    # Convert tags to indices
    ground_truth = [ tags[Y_test[i]] for i in range(len(Y_test)) ]

    # Convert predictions on sections to predictions on songs
    prediction = []
    X_test_expanded_pred_iter = 0

    for num in num_sections:
        votes = {}
        for i in range(num):
            if X_test_expanded_pred[X_test_expanded_pred_iter] not in votes:
                votes[X_test_expanded_pred[X_test_expanded_pred_iter]] = 1
            else:
                votes[X_test_expanded_pred[X_test_expanded_pred_iter]] += 1
            X_test_expanded_pred_iter += 1
        
        highest_count = 0
        highest_genre = ''
        for genre in votes:
            if votes[genre] > highest_count:
                highest_count = votes[genre]
                highest_genre = genre
        
        prediction.append(highest_genre)
    
    print(prediction)
    print(ground_truth)

    print(f'The accuracy of the model is {100*accuracy_score(prediction, ground_truth):6.2f}%')


def main(params):
    if params.train:
        if params.gpu:
            train_func = (jit)(train)
            train_func(params.seg_comp_mode)
        else:
            train(params.seg_comp_mode)
    else:
        if params.gpu:
            test_func = (jit)(test)
            test_func(params.seg_comp_mode, params.sec_comp_mode)
        else:
            test(params.seg_comp_mode, params.sec_comp_mode)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM POS Tagger")
    parser.add_argument("--train", action='store_const', const=True, default=False)
    parser.add_argument("--gpu", action='store_const', const=True, default=False)
    parser.add_argument('--seg_comp_mode', type=str, default='PV', help='PV, WPV, GF, or WGF')
    parser.add_argument('--sec_comp_mode', type=str, default='PV', help='PV, WPV, GF, or WGF')
    main(parser.parse_args())
