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

unka = [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ]
UNKA = '[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]'

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

def train(seg_comp_mode, sec_comp_mode):
    # Load data from genre.txt
    genre_dict = {} # Filename: genre
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
    songDir = 'songs/training/'
    
    vocab_indexed = pickle.load( open("vocab-4.0.pickle", "rb") )
    vocab_freq = pickle.load( open("vocab_freq-4.0.pickle", "rb") )
    genres_indexed = pickle.load( open("genres.pickle", "rb") )
    genres_freq = pickle.load( open("genres_freq.pickle", "rb") )
    trainingText = pickle.load( open("training_mod-4.0.pickle", "rb") )
    duration_info = pickle.load( open("training_durations.pickle", "rb") )
    
    # Create RNN model
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 32
    model = RNNTagger(EMBEDDING_DIM, HIDDEN_DIM, len(vocab_indexed), len(genres_indexed))
    # error = nn.CrossEntropyLoss(ignore_index=0)

    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.NLLLoss()

    # Repeatedly train RNN model
    NUM_EPOCHS = 10
    
    for epoch in range(NUM_EPOCHS):
        print("Epoch", epoch + 1, "/", NUM_EPOCHS)
        progress = 0
        trainingTextKeys = list(trainingText.keys())
        random.shuffle(trainingTextKeys)
        for song in trainingTextKeys:
            if progress % 50 == 0 : print("Trained on", progress, "out of", len(trainingText.keys()), "songs")
            genre = genre_dict[song[:-3]]
            genre_idx = genres_indexed[genre]
            genre_tensor = torch.tensor(genre_idx, dtype=torch.long)
            sec_dists = []
            sec_dur_info = []
            for i in range(len(trainingText[song])) :
                section = trainingText[song][i]
                model.zero_grad()
                segments_indcs = [vocab_indexed[str(segment)] for segment in section]
                segments_tensor = torch.tensor(segments_indcs, dtype=torch.long)
                genre_scores = model(segments_tensor)
                sec_dists.append(distribution_compiler(genre_scores, seg_comp_mode, genres_indexed, duration_info[song][i]))
                sec_dur_info.append(0.0)
                for seg_dur in duration_info[song][i] : sec_dur_info[i] += seg_dur
            song_dist = distribution_compiler(sec_dists, sec_comp_mode, genres_indexed, sec_dur_info)
            # print(song_dist)
            optimizer.zero_grad()
            model.zero_grad()
            loss = loss_function(song_dist, genre_tensor)
            loss.backward()
            optimizer.step()
            progress += 1
        model_filename = 'new_model_' + seg_comp_mode + '_' + sec_comp_mode + '_' + str(epoch + 1) + '.torch'
        torch.save(model.state_dict(), model_filename)
    return

def preprocessing(isTrain):
    # Enter genres into dictionary
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
    trainingText = { } # filename: [ section:[ segment:[12 chroma features], [], [], ... ], ... ]
    songDir = 'songs/training/' if isTrain else 'songs/testing/'
    segmentCount = 0
    duration_info = { }

    for filename in os.listdir(songDir):
        # if filename[0:-3] not in genre_dict:
            # # Delete this file and remove key from genre_dict
            # print(filename)
            # os.remove(os.path.join(songDir, filename))

        with h5py.File(os.path.join(songDir, filename), "r") as f:
            print(filename)
            # Get the data keys
            data = f['analysis']
            if len(data['sections_start']) == 0 :
                print(filename, "has no sections!")
                # os.remove(os.path.join(songDir, filename))
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
            trainingText[filename] = song_sections
            f.close()

    print(segmentCount)

    # Dump testing_durations
    durations_pickle = open('training_durations.pickle' if isTrain else 'testing_durations.pickle', 'wb')
    pickle.dump(duration_info, durations_pickle)
    durations_pickle.close()
    
    segmentFrequency = {}
    
    # Get segmentFrequency for each segment
    for song in trainingText.keys():
        for section in trainingText[song]:
            for i in range(len(section)):
                # Round to tenths for segments_pitch to decrease space of possible segments
                section[i] = [ round(chroma / 4.0, 1) for chroma in section[i] ] #  Maybe 4.5 or (20/3) for 12*2^11 choices
                seg_str = str(section[i])
                if seg_str not in segmentFrequency : segmentFrequency[seg_str] = 0
                segmentFrequency[seg_str] += 1

    # Now we can determine the vocabulary
    vocab_indexed = { UNKA : 0 }
    vocab_freq = { UNKA : 0 }
    genres_indexed = {}
    genres_freq = {}
    for song in trainingText.keys():
        genre = genre_dict[song[:-3]]
        if genre not in genres_indexed:
            genres_indexed[genre] = len(genres_indexed)
            genres_freq[genre] = 0
        genres_freq[genre] += 1
        for j in range(len(trainingText[song])):
            section = trainingText[song][j]
            for i in range(len(section)):
                seg_str = str(section[i])
                if segmentFrequency[seg_str] < 3:
                    seg_str = UNKA
                    trainingText[song][j][i] = unka
                if seg_str not in vocab_indexed:
                    vocab_indexed[seg_str] = len(vocab_indexed)
                    vocab_freq[seg_str] = 0
                vocab_freq[seg_str] += 1

    # Only dump if training
    if isTrain:
        vocab_pickle = open('vocab-4.0.pickle', 'wb')
        pickle.dump(vocab_indexed, vocab_pickle)
        vocab_pickle.close()
        
        vocab_freq_pickle = open('vocab_freq-4.0.pickle', 'wb')
        pickle.dump(vocab_freq, vocab_freq_pickle)
        vocab_freq_pickle.close()

        genres_pickle = open('genres.pickle', 'wb')
        pickle.dump(genres_indexed, genres_pickle)
        genres_pickle.close()
        
        genres_freq_pickle = open('genres_freq.pickle', 'wb')
        pickle.dump(genres_freq, genres_freq_pickle)
        genres_freq_pickle.close()
    
    training_texts = open('training_mod-4.0.pickle' if isTrain else 'testing_mod-4.0.pickle', 'wb')
    pickle.dump(trainingText, training_texts)
    training_texts.close()
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
        # genre_dist = [ count / num_segments for count in genre_counts ]
        dist_tensor = torch.tensor(genre_counts, dtype=torch.float, requires_grad=True)
        return dist_tensor.div(num_segments)
    
    if compilation_mode == 'WPV':
        sec_duration = 0.0
        for seg_dur in sec_duration_info:
            sec_duration += seg_dur
        genre_counts = [ 0.0 for genre in genres_indexed ]
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
            # seg_weight = sec_duration_info[i] / sec_duration
            # genre_counts[most_likely_genre] += seg_weight
            genre_counts[most_likely_genre] += sec_duration_info[i]
        count_tensor = torch.tensor(genre_counts, dtype=torch.float, requires_grad=True).div(sec_duration)
        approx1 = 0.0
        for prob in count_tensor : approx1 += prob
        return count_tensor.div(approx1)
        
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
            for genre_idx in range(len(seg_dist)):
                genre_counts[genre_idx] += seg_dist[genre_idx] * sec_duration_info[i]
        count_tensor = torch.tensor(genre_counts, dtype=float, requires_grad=True).div(sec_duration)
        approx1 = 0.0
        for prob in count_tensor : approx1 += prob
        return count_tensor.div(approx1)

def test(seg_comp_mode, sec_comp_mode, model_name):
    # read in genre dictionary from genre.txt
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
    
    # song:[ section:[[12 chroma features], [], [], ... ], ... ]
    songDir = 'songs/testing/'
    
    vocab_indexed = pickle.load( open("vocab-4.0.pickle", "rb") )
    # vocab_freq = pickle.load( open("vocab_freq-4.0.pickle", "rb") )
    genres_indexed = pickle.load( open("genres.pickle", "rb") )
    # print(len(genres_indexed), 'different genres')
    # genres_freq = pickle.load( open("genres_freq.pickle", "rb") )
    # for genre in genres_freq.keys():
        # print(genres_freq[genre], genre, 'songs in the training data')
    testingText = pickle.load( open("testing_mod-4.0.pickle", "rb") )
    duration_info = pickle.load( open("testing_durations.pickle", "rb") )
    
    # Create RNN model
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 32
    model = RNNTagger(EMBEDDING_DIM, HIDDEN_DIM, len(vocab_indexed), len(genres_indexed))
    # error = nn.CrossEntropyLoss(ignore_index=0)
    model.load_state_dict(torch.load(model_name))

    pred_list = []
    ground_truth = []
    progress = 0
    with torch.no_grad():
        for song in testingText.keys():
            if progress % 50 == 0 : print("Tested", progress, "out of", len(testingText.keys()), "songs")
            genre = genre_dict[song[:-3]]
            genre_idx = genres_indexed[genre]
            ground_truth.append(genre_idx)
            sec_dists_out = []
            sec_dur_info = []
            for i in range(len(testingText[song])) :
                section = testingText[song][i]
                segments_indcs = [vocab_indexed[str(segment)] for segment in section]
                segments_tensor = torch.tensor(segments_indcs, dtype=torch.long)
                genre_scores = model(segments_tensor)
                # print(genre_scores[0])
                # print(genre_scores)
                sec_dists_out.append(distribution_compiler(genre_scores, seg_comp_mode, genres_indexed, duration_info[song][i]))
                # print(sec_dists_out[-1])
                # input()
                sec_dur_info.append(0.0)
                for seg_dur in duration_info[song][i] : sec_dur_info[i] += seg_dur
            song_dist_out = distribution_compiler(sec_dists_out, sec_comp_mode, genres_indexed, sec_dur_info)
            pred = -1
            max_prob = -1.0
            for i in range(len(genres_indexed)):
                prob = song_dist_out[i].numpy()
                if prob > max_prob:
                    pred = i
                    max_prob = prob
            print(pred)
            pred_list.append(pred)
            progress += 1
            
    # compute accuracy
    preds_pickle = open(model_name + '.out_pickle', 'wb')
    pickle.dump(pred_list, preds_pickle)
    preds_pickle.close()
    print("The accuracy of the model is %6.2f%%" % (accuracy_score(ground_truth, pred_list) * 100))
    #print(f"The accuracy of the model is {100*accuracy_score(pred_list, ground_truth):6.2f}%")
    return


def main(params):
    if params.train:
        if params.gpu:
            train_func = (jit)(train)
            train_func(params.seg_comp_mode, params.sec_comp_mode)
        else:
            train(params.seg_comp_mode, params.sec_comp_mode)
    else:
        if params.gpu:
            test_func = (jit)(test)
            test_func(params.seg_comp_mode, params.sec_comp_mode, params.model_in)
        else:
            test(params.seg_comp_mode, params.sec_comp_mode, params.model_in)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM POS Tagger")
    parser.add_argument("--train", action='store_const', const=True, default=False)
    parser.add_argument("--gpu", action='store_const', const=True, default=False)
    parser.add_argument('--seg_comp_mode', type=str, default='PV', help='PV, WPV, GF, or WGF')
    parser.add_argument('--sec_comp_mode', type=str, default='PV', help='PV, WPV, GF, or WGF')
    parser.add_argument('--model_in', type=str, default='model_PV.torch', help='provide the file name of a model')
    main(parser.parse_args())
