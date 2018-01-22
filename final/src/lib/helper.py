import pandas as pd
import numpy as np
import pickle
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors


class Onehot_helper:

    def encode(self, filename):
        print("Reading "+str(filename))
        mydata = pd.read_csv(filename, sep='\n', encoding='utf-8', header=None).values
        mydata = np.array([i[0].rstrip().split(" ") for i in mydata])
        self.max_seq_length = 15
        print("Making one-hot Dictionary")
        onehotDictionary = ["EOS", "BOS"]
        for sentence in mydata:
            for word in sentence:
                if not word in onehotDictionary:
                    onehotDictionary.append(word)
        decoder_inputs_data = np.zeros((len(mydata), self.max_seq_length, len(onehotDictionary)))
        decoder_target_data = np.zeros((len(mydata), self.max_seq_length, len(onehotDictionary)))
        for i in range(len(mydata)):
            decoder_inputs_data[1] = 1  # start with BOS
            for j in range(len(mydata[i])):
                word = mydata[i][j]
                decoder_inputs_data[i][j][onehotDictionary.index(word)] = 1
                decoder_target_data[i][j][onehotDictionary.index(word)] = 1
        self.onehotDictionary = onehotDictionary
        return decoder_inputs_data, decoder_target_data

    def setDecoderModel(self, decoder_model):
        self.decoder_model = decoder_model

    def get_vec_length(self):
        return len(self.onehotDictionary)

    def decode_sequence(self, states_value):
        target_seq = np.zeros((1, 1, len(self.onehotDictionary)))
        target_seq[0, 0, self.onehotDictionary.index("BOS")] = 1
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sampled_token_index = np.argmax(output_tokens[0, -1, :])  # the most possible word index
            sampled_char = self.onehotDictionary[sampled_token_index]  # translate to word
            decoded_sentence += sampled_char  # add the word to sentence

            if sampled_char == "EOS" or len(decoded_sentence) > self.max_seq_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, len(self.onehotDictionary)))
            target_seq[0, 0, self.onehotDictionary.index("EOS")] = 1

            # Update states
            states_value = [h, c]

        return decoded_sentence


class Word2vec_helper:

    def encode(self, filename):
        print("Reading "+filename)
        mydata = pd.read_csv(filename, sep='\n', encoding='utf-8', header=None).values
        mydata = [i[0].rstrip().split(" ") for i in mydata]
        '''
        print("Making Word2vec Dictionary")
        mydata.append(["BOS"])
        mydata.append(["EOS"])
        mydata = np.array(mydata)
        bag = Word2Vec(mydata, min_count=1, sg=0)
        bag.save("Dictionary.bin")
        '''
        self.bag = Word2Vec.load("Dictionary.bin")
        print("Formating data (word2vec)")
        self.max_seq_length = 15
        decoder_inputs_data = []
        decoder_target_data = []
        for sentence in mydata:
            decoder_inputs_data.append([])
            decoder_inputs_data[len(decoder_inputs_data)-1].append(self.bag["BOS"])
            decoder_target_data.append([])
            for word in sentence:
                decoder_inputs_data[len(decoder_inputs_data)-1].append(self.bag[word])
                decoder_target_data[len(decoder_target_data)-1].append(self.bag[word])
            decoder_target_data[len(decoder_target_data)-1].append(self.bag["EOS"])
            while len(decoder_inputs_data[len(decoder_inputs_data)-1]) < self.max_seq_length:
                decoder_inputs_data[len(decoder_inputs_data)-1].append(np.zeros(100))
            while len(decoder_target_data[len(decoder_target_data)-1]) < self.max_seq_length:
                decoder_target_data[len(decoder_target_data)-1].append(np.zeros(100))
        '''
        # test
        decoder_inputs = []
        decoder_target = []
        for sentence in mydata:
            decoder_inputs.append([])
            decoder_inputs[len(decoder_inputs)-1].append("BOS")
            decoder_target.append([])
            for word in sentence:
                decoder_inputs[len(decoder_inputs)-1].append(word)
                decoder_target[len(decoder_target)-1].append(word)
            decoder_target[len(decoder_target)-1].append("EOS")
            while len(decoder_inputs[len(decoder_inputs)-1]) < self.max_seq_length:
                decoder_inputs[len(decoder_inputs)-1].append(" ")
            while len(decoder_target[len(decoder_target)-1]) < self.max_seq_length:
                decoder_target[len(decoder_target)-1].append(" ")
        print(decoder_inputs[0:5])
        print(decoder_target[0:5])
        '''
        return np.array(decoder_inputs_data), np.array(decoder_target_data)

    def get_vec_length(self):
        return 100

    def setDecoderModel(self, decoder_model):
        self.decoder_model = decoder_model

    def decode_sequence(self, states_value):
        stop_condition = False
        decoded_sentence = ["BOS"]
        while not stop_condition:
            word = np.array(self.bag[decoded_sentence[len(decoded_sentence)-1]]).reshape((1, 1, 100))
            output_tokens, h, c = self.decoder_model.predict([word] + states_value)
            predict_word = self.bag.similar_by_vector(output_tokens.reshape(100), topn=5)
            print(predict_word)
            decoded_sentence.append(predict_word[0][0])

            # Exit condition: either hit max length or find stop character.
            if predict_word == "EOS" or len(decoded_sentence) > self.max_seq_length:
                stop_condition = True

            # Update states
            states_value = [h, c]

        return decoded_sentence

bag = KeyedVectors.load("lib/w2v_model/chinese_embedding.model")
max_seq_length = 13


def normalize(vector):
    vector = (vector-np.mean(vector))/np.std(vector)
    return vector


def bag_and_padding(arr, do_normalize=True):
    arr_vector = []
    for sentence in arr:
        arr_vector.append([])
        for word in sentence:
            if word in bag.wv.vocab:
                if do_normalize == True:
                    word_vector = normalize(bag[word])
                else:
                    word_vector = bag[word]
                arr_vector[len(arr_vector)-1].append(word_vector)
            else:
                arr_vector[len(arr_vector)-1].append(np.zeros(300))
        # padding
        while len(arr_vector[len(arr_vector)-1]) < max_seq_length:
            arr_vector[len(arr_vector)-1].append(np.zeros(300))
    return np.array(arr_vector)


def make_options(arr, options_number, generate_same_length = True):
    options = []
    ans = np.zeros((len(arr), options_number))
    for i in range(len(arr)):
        rand_index = [i]
        count = 0
        while len(rand_index) < options_number:
            if count < len(arr)*4:
                count += 1
                r = np.random.randint(len(arr))
                if r in rand_index or (len(arr[r]) != len(arr[i]) and generate_same_length):
                    continue
                rand_index.append(r)
            else:
                rand_index.append(-1)
        np.random.shuffle(rand_index)
        options.append([])
        for k in range(len(rand_index)):
            if rand_index[k] == i:
                ans[i][k] = 1
        for k in range(len(rand_index)):
            if rand_index[k] == -1:
                print(i)
                temp = arr[rand_index[np.argmax(ans[i])]].copy()
                np.random.shuffle(temp)
                options[i].append(temp)
            else:
                options[i].append(arr[rand_index[k]])
    return np.array(options), np.array(ans)


def voice_data_proccess():
    print("Reading voice data")
    with open("data/train.data", "rb") as file:
        voice = pickle.load(file)  # each vector length=39
    print("voice_data padding")
    voice_data = []
    for data in mydata:
        voice_data.append([])
        for vector in data:
            voice_data[len(voice_data)-1].append(vector)
        while len(voice_data[len(voice_data)-1]) < 246:
            voice_data[len(voice_data)-1].append(np.zeros(39))
    encoder_input_data = np.array(voice_data)
    np.save("data/npy/encoder_input_data.npy", encoder_input_data)
