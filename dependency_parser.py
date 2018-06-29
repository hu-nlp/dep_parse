from data_utils import DataUtils
import numpy as np
from keras.engine import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Layer, RepeatVector, Masking, Concatenate, Add, Reshape, Activation, Dropout, LSTM, TimeDistributed, Bidirectional, Embedding, Input
from keras.utils import plot_model
import keras.backend as K

class BiLSTM(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.f_lstm = LSTM(output_dim, dropout=0.35, recurrent_dropout=0.1, return_sequences=True)
        self.b_lstm = LSTM(output_dim, dropout=0.35, recurrent_dropout=0.1, return_sequences=True)
        self.bilstm = Concatenate()
        super(BiLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BiLSTM, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        forward_output = self.f_lstm(x[0])
        backward_output = self.b_lstm(x[1])
        backward_output = K.reverse(backward_output, [1])
        backward_output = Reshape((input_shape[0][1],self.output_dim))(backward_output)
        bilstm_output = self.bilstm([forward_output, backward_output])
        return bilstm_output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim*2)

class DependencyParser(object):
    def __init__(self):
        pass

    def __create_xy(self, dependency_tree, embedding_file, data_size, look_back, test=False):
        sentences, words, tags = DataUtils.parse_dependency_tree(dependency_tree)
        word_emb = DataUtils.load_embeddings(embedding_file)
        tag_int = DataUtils.create_int_dict(tags)

        data_size = int(len(sentences)*min(data_size, 1))

        if test:
            sentences.reverse()

        if look_back == 0:
            for sentence in sentences[:data_size]:
                look_back = max(look_back, len(sentence))

        self.look_back = look_back
        self.distinct_words = len(words)
        self.distinct_tags = len(tags)

        word_full_forward = []
        word_full_backward = []
        word_instance_forward = []
        word_instance_backward = []

        tag_full_forward = []
        tag_full_backward = []
        tag_instance_forward = []
        tag_instance_backward = []

        head = []

        progress = 0

        for sentence in sentences[:data_size]:
            parts = [sentence[i:i+look_back] for i in range(0,len(sentence),look_back)]
            for part in parts:
                word_temp = np.zeros((2,look_back,300))
                tag_temp = np.zeros((2,look_back,),dtype="int32")

                head_instance = np.zeros((look_back,1),dtype="float32")

                for idx in range(len(part)):
                    word = part[idx]["word"]
                    word_temp[0][look_back-len(part)+idx] = word_emb[word] if word in word_emb else word_emb["UNK"]
                    word_temp[1][look_back-idx-1] = word_emb[word] if word in word_emb else word_emb["UNK"]

                    tag = part[idx]["tag"]
                    tag_temp[0][look_back-len(part)+idx] = tag_int[tag]
                    tag_temp[1][look_back-idx-1] = tag_int[tag]

                word_instance = np.zeros((2,look_back,300))
                tag_instance = np.zeros((2,look_back,),dtype="int32")

                for jdx in range(len(part)):
                    word_instance[0][look_back-jdx-1:] = word_temp[0][look_back-len(part):look_back-len(part)+jdx+1]
                    word_instance[1][look_back-len(part)+jdx:] = word_temp[1][look_back-len(part):look_back-jdx]

                    tag_instance[0][look_back-jdx-1:] = tag_temp[0][look_back-len(part):look_back-len(part)+jdx+1]
                    tag_instance[1][look_back-len(part)+jdx:] = tag_temp[1][look_back-len(part):look_back-jdx]

                    head_instance = np.zeros((look_back,1), dtype="float32")

                    for zdx in range(len(part)):
                        head_instance[zdx] = 1 if part[jdx]["head"] == part[zdx]["word"] else 0
                    if len(word_full_forward) == 0:
                        word_full_forward = [word_temp[0]]
                        word_full_backward = [word_temp[1]]
                        word_instance_forward = [word_instance[0]]
                        word_instance_backward = [word_instance[1]]

                        tag_full_forward = [tag_temp[0]]
                        tag_full_backward = [tag_temp[1]]
                        tag_instance_forward = [tag_instance[0]]
                        tag_instance_backward = [tag_instance[1]]

                        head = [head_instance]
                    else:
                        word_full_forward = np.append(word_full_forward, [word_temp[0]], axis=0)
                        word_full_backward = np.append(word_full_backward, [word_temp[1]], axis=0)
                        word_instance_forward = np.append(word_instance_forward, [word_instance[0]], axis=0)
                        word_instance_backward = np.append(word_instance_backward, [word_instance[1]], axis=0)

                        tag_full_forward = np.append(tag_full_forward, [tag_temp[0]], axis=0)
                        tag_full_backward = np.append(tag_full_backward, [tag_temp[1]], axis=0)
                        tag_instance_forward = np.append(tag_instance_forward, [tag_instance[0]], axis=0)
                        tag_instance_backward = np.append(tag_instance_backward, [tag_instance[1]], axis=0)

                        head = np.append(head, [head_instance], axis=0)

            DataUtils.update_message(str(progress)+"/"+str(data_size))
            progress += 1

        word_data = [(word_full_forward, word_full_backward), (word_instance_forward, word_instance_backward)]
        tag_data = [(tag_full_forward, tag_full_backward), (tag_instance_forward, tag_instance_backward)]

        return [word_data[0][0], word_data[0][1], tag_data[0][0], tag_data[0][1], word_data[1][0], word_data[1][1], tag_data[1][0], tag_data[1][1]], head

    def create_xy_test(self, dependency_tree, embedding_file, data_size=1, look_back=0, mode="create", load=None):
        DataUtils.message("Prepearing Test Data...", new=True)

        if mode == "create" or mode == "save":
            word_test, head_test, tag_test = self.__create_xy(dependency_tree, embedding_file, data_size, look_back, test=True)

        if mode == "save":
            DataUtils.save_array(DataUtils.get_filename("DP_W","TEST"+"_"+str(look_back)), word_test)
            DataUtils.save_array(DataUtils.get_filename("DP_H","TEST"+"_"+str(look_back)), head_test)
            DataUtils.save_array(DataUtils.get_filename("DP_T","TEST"+"_"+str(look_back)), tag_test)

        if mode == "load" and load is not None:
            word_test = DataUtils.load_array(load[0])
            head_test = DataUtils.load_array(load[1])
            tag_test = DataUtils.load_array(load[2])

        self.word_test = np.array(word_test)
        self.head_test = np.array(head_test)
        self.tag_test = np.array(tag_test)

    def create_xy_train(self, dependency_tree, embedding_file, data_size=1, look_back=0, mode="create", load=None):
        DataUtils.message("Prepearing Training Data...", new=True)

        if mode == "create" or mode == "save":
            train_x, train_y = self.__create_xy(dependency_tree, embedding_file, data_size, look_back, test=False)

        self.train_x = train_x
        self.train_y = train_y

    def save(self, note=""):
        DataUtils.message("Saving Model...", new=True)
        directory = "weights/"

        DataUtils.create_dir(directory)

        file = DataUtils.get_filename("DP", note)+".h5"

        self.model.save(directory+file)

    def load(self, file):
        DataUtils.message("Loading Model...", new=True)
        self.model = load_model(file)

    def plot(self, note=""):
        DataUtils.message("Ploting Model...", new=True)
        directory = "plot/"

        DataUtils.create_dir(directory)

        file = DataUtils.get_filename("DP", note)+".png"

        plot_model(self.model, to_file=directory+file, show_shapes=True, show_layer_names=False)

    def create(self):
        DataUtils.message("Creating The Model...", new=True)
        word_full_forward = Input(shape=(self.look_back,300))
        word_full_backward = Input(shape=(self.look_back,300))

        tag_full_forward_input = Input(shape=(self.look_back,))
        tag_full_backward_input = Input(shape=(self.look_back,))
        tag_emb = Embedding(self.distinct_tags, 30, input_length=self.look_back, trainable=True)

        tag_full_forward = tag_emb(tag_full_forward_input)
        tag_full_backward = tag_emb(tag_full_backward_input)

        full_forward = Concatenate()([word_full_forward, tag_full_forward])
        full_backward = Concatenate()([word_full_backward, tag_full_backward])

        bilstm = BiLSTM(300)([full_forward, full_backward])
        dense_output = TimeDistributed(Dense(600, activation="linear"))(bilstm)


        word_instance_forward = Input(shape=(self.look_back,300))
        word_instance_backward = Input(shape=(self.look_back,300))

        tag_instance_forward_input = Input(shape=(self.look_back,))
        tag_instance_backward_input = Input(shape=(self.look_back,))

        tag_instance_forward = tag_emb(tag_instance_forward_input)
        tag_instance_backward = tag_emb(tag_instance_backward_input)

        instance_forward = Concatenate()([word_instance_forward, tag_instance_forward])
        instance_backward = Concatenate()([word_instance_backward, tag_instance_backward])

        f_ilstm = LSTM(300, dropout=0.35, recurrent_dropout=0.1)
        b_ilstm = LSTM(300, dropout=0.35, recurrent_dropout=0.1)

        forward_ioutput = f_ilstm(instance_forward)
        backward_ioutput = b_ilstm(instance_backward)
        bilstm_ioutput = Concatenate()([forward_ioutput, backward_ioutput])

        dense_ioutput = Dense(600, activation="linear")(bilstm_ioutput)
        repeat_ioutput = RepeatVector(self.look_back)(dense_ioutput)

        sum_output = Add()([dense_output,repeat_ioutput])

        hidden = TimeDistributed(Dense(600, activation="tanh"))(sum_output)
        output = TimeDistributed(Dense(1, activation="softmax"))(hidden)

        model = Model(inputs=[word_full_forward, word_full_backward, tag_full_forward_input, tag_full_backward_input, word_instance_forward, word_instance_backward, tag_instance_forward_input, tag_instance_backward_input], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

        self.model = model

    def train(self, epochs, batch_size=32):
        DataUtils.message("Training...", new=True)
        self.model.fit(self.train_x, self.train_y, epochs=epochs, batch_size=batch_size)

    def validate(self, batch_size=16):
        DataUtils.message("Validation...")
        return self.model.evaluate([self.word_test,self.tag_test], self.head_test, batch_size=batch_size)

    def predict(self, x):
        return self.model.predict(x)

    def summary(self):
        self.model.summary()

if __name__ == "__main__":
    train_file = "data/penn-treebank.conllx"
    embedding_file = "embeddings/GoogleNews-vectors-negative300-SLIM.bin"
    epochs = 10
    look_back = 10 #0 means the largest window

    model = DependencyParser()
    model.create_xy_train(train_file, embedding_file, 0.001, look_back = look_back)
    model.create()
    model.summary()
    model.train(epochs)
    model.plot("TEST 2")