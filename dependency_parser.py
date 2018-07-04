from data_utils import DataUtils
import numpy as np
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Embedding, LSTM, Input, Add, Concatenate
from keras.utils import plot_model
import keras.backend as K

class DependencyParser(object):
    def __init__(self):
        pass

    def __create_xy(self, parse_tree_file, data_size, seq_len, test=False):
        sentences, words, tags = DataUtils.parse_dependency_tree(parse_tree_file)
        word_int = DataUtils.create_int_dict(words)
        tag_int = DataUtils.create_onehot_vectors(tags)

        self.seq_len = seq_len
        self.distinct_words = len(words)
        self.distinct_tags = len(tags)

        data_len = 0
        for i in range(len(sentences)):
            data_len += int(np.ceil(len(sentences[i])/seq_len))*seq_len*seq_len

        forward = np.zeros((2,data_len,seq_len,),dtype="int32")
        backward = np.zeros((2,data_len,seq_len,),dtype="int32")
        probability = np.zeros((data_len,),dtype="float32")
        tags = np.zeros((data_len,18))

        idx = 0
        for sentence in sentences:
            parts = [sentence[i:i+seq_len] for i in range(0,len(sentence),seq_len)]
            for part in parts:
                part_len = len(part)
                word_forward = np.zeros((seq_len,seq_len),dtype="int32")
                word_backward = np.zeros((seq_len,seq_len),dtype="int32")

                for jdx in range(part_len):
                    word_forward[jdx][seq_len-jdx-1:] = [word_int[part[i]["word"]] for i in range(jdx+1)]
                    word_backward[jdx][seq_len-part_len+jdx:] = [word_int[part[part_len-i-1]["word"]] for i in range(part_len-jdx)]

                for jdx in range(part_len):
                    for zdx in range(part_len):
                        tags[idx] = tag_int[part[jdx]["tag"]]
                        forward[0][idx] = word_forward[jdx]
                        forward[1][idx] = word_forward[zdx]
                        backward[0][idx] = word_backward[jdx]
                        backward[1][idx] = word_backward[zdx]
                        probability[idx] = 1.0 if part[jdx]["head"] == part[zdx]["word"] else 0.0
                        idx += 1

                        if idx%int(data_len/100) == 0:
                            DataUtils.update_message(str(int(idx/data_len*100)))
        if test:
            forward = [np.array(forward[0][5000:10000]),np.array(forward[1][5000:10000])]
            backward = [np.array(backward[0][5000:10000]),np.array(backward[1][5000:10000])]
            probability = np.array(probability[5000:10000])
            tags = np.array(tags[5000:10000])
        else:
            forward = [np.array(forward[0][:5000]),np.array(forward[1][:5000])]
            backward = [np.array(backward[0][:5000]),np.array(backward[1][:5000])]
            probability = np.array(probability[:5000])
            tags = np.array(tags[:5000])

        return [forward[0], backward[0], forward[1], backward[1]], [tags,probability]

    def create_xy_train(self, parse_tree_file, data_size=1, seq_len=10):
        DataUtils.message("Prepearing Training Data...", new=True)

        x_train, y_train = self.__create_xy(parse_tree_file, data_size, seq_len)

        self.x_train = x_train
        self.y_train = y_train

    def create_xy_test(self, parse_tree_file, data_size=1, seq_len=10):
        DataUtils.message("Prepearing Validation Data...", new=True)

        x_test, y_test = self.__create_xy(parse_tree_file, data_size, seq_len, test=True)

        self.x_test = x_test
        self.y_test = y_test

    def save(self, note=""):
        DataUtils.message("Saving Model...", new=True)
        self.model.save(DataUtils.get_filename("DP", note)+".h5")

    def load(self, file):
        DataUtils.message("Loading Model...", new=True)
        self.model = load_model(file)

    def plot(self, note=""):
        DataUtils.message("Ploting Model...", new=True)
        plot_model(self.model, to_file=DataUtils.get_filename("DP", note)+".png", show_shapes=True, show_layer_names=False)

    def create(self):
        DataUtils.message("Creating The Model...", new=True)

        input_forward = Input(shape=(self.seq_len,))
        input_backward = Input(shape=(self.seq_len,))

        head_forward = Input(shape=(self.seq_len,))
        head_backward = Input(shape=(self.seq_len,))

        word_embedding = Embedding(self.distinct_words, 128, input_length=self.seq_len, trainable=True)
        input_forward_embedding = word_embedding(input_forward)
        input_backward_embedding = word_embedding(input_backward)

        head_forward_embedding = word_embedding(head_forward)
        head_backward_embedding = word_embedding(head_backward)

        lstm_forward = LSTM(128)
        lstm_backward = LSTM(128)

        input_forward_lstm = lstm_forward(input_forward_embedding)
        input_backward_lstm = lstm_backward(input_backward_embedding)
        input_lstm = Concatenate()([input_forward_lstm,input_backward_lstm])

        head_forward_lstm = lstm_forward(head_forward_embedding)
        head_backward_lstm = lstm_backward(head_backward_embedding)
        head_lstm = Concatenate()([head_forward_lstm,head_backward_lstm])

        tag_output = Dense(18, activation="softmax")(input_lstm)

        input_hidden = Dense(100, activation=None)
        input_forward_hidden = input_hidden(input_lstm)

        head_hidden = Dense(100, activation=None)
        head_forward_hidden = head_hidden(head_lstm)

        sum_hidden = Add()([input_forward_hidden, head_forward_hidden])
        tanh_hidden = Activation("tanh")(sum_hidden)

        arc_output = Dense(1, activation=None)(tanh_hidden)

        model = Model(inputs=[input_forward,input_backward,head_forward,head_backward],outputs=[tag_output,arc_output])

        def nll1(y_true, y_pred):
            # keras.losses.binary_crossentropy give the mean
            # over the last axis. we require the sum
            return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

        model.compile(loss=['categorical_crossentropy',nll1], optimizer="adam", metrics=['accuracy'])
        self.model = model

    def train(self, epochs, batch_size=32):
        DataUtils.message("Training...", new=True)
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size)

    def validate(self, batch_size=32):
        DataUtils.message("Validation...", new=True)
        return self.model.evaluate(self.x_test, self.y_test)

    def summary(self):
        self.model.summary()

if __name__ == "__main__":
    parse_tree_file = "data/penn-treebank.conllx"
    embedding_file = "embeddings/GoogleNews-vectors-negative300-SLIM.bin"

    model = DependencyParser()
    model.create_xy_train(parse_tree_file, 1, 30)
    model.create_xy_test(parse_tree_file, 1, 30)
    model.create()
    model.train(5)

    DataUtils.message(model.validate())