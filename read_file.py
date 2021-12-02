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
import pickle, os
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

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = nn.functional.log_softmax(tag_space, dim=1)
        return tag_scores

def train():
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
    trainingText = {} # filename: [ section:[ segment:[12 chroma features], [], [], ... ], ... ]
    songDir = 'songs/'

    for filename in os.listdir(songDir):
        if filename[0:-3] not in genre_dict:
            # Delete this file and remove key from genre_dict
            print(filename)
            os.remove(os.path.join(songDir, filename))

        with h5py.File(os.path.join(songDir, filename), "r") as f:
            # List all groups
            #print("Keys:", str(f.keys()))
            analysis = list(f.keys())[0]

            # Get the data keys
            data_key = list(f[analysis])
            data = f['analysis']

            song_segments = []
            section_length = 16

            for i in range(0, len(data['segments_pitches'][()]), section_length):
                song_segments.append(data['segments_pitches'][()][i:i+section_length])

            trainingText[filename] = song_segments
            f.close()

    vocabFrequency = {}
    vocab = {}
    tags = {}
    X = []
    Y = []
    maxSentenceLength = 0

    # Add PAD and UNKA index
    vocab['<PAD>'] = len(vocab)
    tags['<PAD>'] = len(tags)
    vocab['UNKA'] = len(vocab)
    tags['UNKA'] = len(tags)

    # filename: [ section:[[12 chroma features], [], [], ... ], ... ]

    # Get vocabFrequency for UNKA
    for key in trainingText.keys():
        for section in trainingText[key]:
            for segment in section:
                # TODO: Round to tenths for segments_pitch to decrease space
                for i in range(len(segment)):
                    segment[i] = round(segment[i] / 3.0, 1)

                segment_hashed = str(segment)
                if segment_hashed not in vocabFrequency:
                    vocabFrequency[segment_hashed] = 0
                vocabFrequency[segment_hashed] += 1

    for key in trainingText.keys():
        genre = genre_dict[key[:-3]]

        for section in trainingText[key]:
            sentenceLength = 0
            vocabList = []
            tagList = []
            
            # Fill index and tag dictionary
            for segment in section:
                segment_hashed = str(segment)

                tag = genre if vocabFrequency[segment_hashed] >= 2 else 'UNKA'

                if segment_hashed not in vocab:
                    vocab[segment_hashed] = len(vocab)

                vocabList.append(segment_hashed)

                if tag not in tags:
                    tags[tag] = len(tags)

                tagList.append(tag)

                sentenceLength += 1

            maxSentenceLength = max(maxSentenceLength, sentenceLength)
            print(tagList)
            X.append(vocabList)
            Y.append(tagList)

    # Save vocab locally to vocab.pickle and tag locally to tag.pickle
    vocabPickle = open('vocab.pickle', 'wb')
    pickle.dump(vocab, vocabPickle)
    vocabPickle.close()

    tagsPickle = open('tags.pickle', 'wb')
    pickle.dump(tags, tagsPickle)
    tagsPickle.close()

    # Sentence padding
    for i in range(len(X)):
        while len(X[i]) < maxSentenceLength:
            X[i].append('')
            Y[i].append('')

    # Create RNN model
    model = RNNTagger(maxSentenceLength, 70, len(vocab), len(tags))
    error = nn.CrossEntropyLoss(ignore_index=0)

    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Repeatedly train RNN model
    BATCH_SIZE = 240
    NUM_EPOCHS = 10

    train_data = []

    for i in range(len(X)):
        for j in range(maxSentenceLength):
            train_data.append([X[i][j], Y[i][j]])
    
    train_dataloader = iter(DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True))

    for epoch in range(NUM_EPOCHS):
        print("Epoch", epoch + 1, "/", NUM_EPOCHS)
        for i in range(0, len(X), BATCH_SIZE):
            X_batch, Y_batch = next(train_dataloader)
            print(X_batch)
            print(Y_batch)
            optimizer.zero_grad()
            outputs = model(torch.tensor(X_batch))
            loss = error(outputs, Y_batch.clone().detach())
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), 'model.torch')

def test():
    # # Load test file text
    # f = open(data_file, 'r', encoding="ISO-8859-1")
    # data = f.read().split()
    # f.close()

    # TODO: Load test songs in same format as training songs

    # Load word2idc dictionary
    vocabPickle = open("vocab.pickle", "rb")
    vocab = pickle.load(vocabPickle)
    vocabPickle.close()

    # Load tag2idc dictionary
    tagsPickle = open('tags.pickle', 'rb')
    tags = pickle.load(tagsPickle)
    tagsPickle.close()

    #249
    model = RNNTagger(249, 70, len(vocab), len(tags))
    model.load_state_dict(torch.load(model_file))

    # Convert word to indices
    for i in range(len(data)):
        data[i] = vocab[data[i]]

    prediction = model(torch.tensor(data))
    prediction = torch.argmax(prediction,-1).cpu().numpy()

    # labels = []
    # f = open(label_file, 'r', encoding="ISO-8859-1")
    # truth_text = f.read().splitlines()
    # f.close()

    # TODO: Get labels for the songs

    # Get labels from label_file
    for truth_line in truth_text:
        truth_line = truth_line.split()
        for i in range(0, len(truth_line), 2):
            labels.append(truth_line[i + 1])

    # Convert tags to indices
    ground_truth = [ tags[labels[i]] for i in range(len(labels)) ]

    #print(f'The accuracy of the model is {100*accuracy_score(prediction, ground_truth):6.2f}%')


def main():
    train()
    #test()
    

if __name__ == "__main__":
    main()
