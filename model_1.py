import pandas as pd
import pickle
import numpy as np
from nltk.corpus import stopwords
import keras
from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.models import Model
from keras.utils import plot_model
from sklearn.model_selection import train_test_split


from nltk import word_tokenize, WordNetLemmatizer

df = pd.read_csv(r'C:\data\kaggle\quora_duplicates\train.csv')

df = df.dropna()

positives = df["is_duplicate"][df["is_duplicate"] == 1]
negatives = df["is_duplicate"][df["is_duplicate"] == 0]
print("positive count in training set: ", positives.count())
print("negative count in training set: ", negatives.count())

lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')


def create_lexicon(dfin):
    lexicon = []

    counter = 1
    content = ""

    for row in dfin.itertuples():
        try:
            counter += 1
            if (counter / 50.0).is_integer():
                question1 = row[4]
                content += question1
                question2 = row[5]
                content += " " + question2
                words = word_tokenize(content)
                words = [lemmatizer.lemmatize(i) for i in words]
                words = [i for i in words if i not in stopwords]
                lexicon = list(set(lexicon + words))
                print(counter, len(lexicon))
        except Exception as e:
            print(str(e))

    with open('lexicon.pickle', 'wb') as f:
        pickle.dump(lexicon, f)


# create_lexicon(df)


def convert_to_vec(dfin, fout, lexicon_pickle):
    with open(lexicon_pickle, 'rb') as f:
        lexicon = pickle.load(f)

    maxFeaturesLen = 140
    counter = 0
    allfeatures = []
    fc = 0
    for row in dfin.itertuples():
        counter += 1
        current_words = word_tokenize(row[4])
        current_words = [lemmatizer.lemmatize(i) for i in current_words]

        features = np.zeros(140)
        features.fill(20000)
        cc = 0
        for word in current_words:
            if word in lexicon:
                index_value = lexicon.index(word)
                # OR DO +=1, test both
                if cc < maxFeaturesLen:
                    features[cc] = index_value
                    cc += 1

        allfeatures.append(features)
        fc += 1

        print(counter)

    with open(fout, 'wb') as fo:
        pickle.dump(allfeatures, fo)


# convert_to_vec(df, 'train_features.pickle', 'lexicon.pickle')

def build_model():
    with open('train_features.pickle', 'rb') as f:
        train_features1 = pickle.load(f)

    with open('train_features2.pickle', 'rb') as f2:
        train_features2 = pickle.load(f2)

    with open('lexicon.pickle', 'rb') as f:
        lexicon = pickle.load(f)

    lexlen = len(lexicon)

    # dataset creation

    labels = np.asarray(df['is_duplicate'].tolist())

    q1array = np.asarray(train_features1)
    q2array = np.asarray(train_features2)

    q1array_train, q1array_test = train_test_split(q1array, test_size=0.3)
    q2array_train, q2array_test = train_test_split(q2array, test_size=0.3)

    q1array_test, q1array_val = train_test_split(q1array_test, test_size=0.5)
    q2array_test, q2array_val = train_test_split(q2array_test, test_size=0.5)

    labels_train, labels_test = train_test_split(labels, test_size=0.3)
    labels_test, labels_val = train_test_split(labels_test, test_size=0.5)

    # model creation

    questions1_input = Input(shape=(140,), dtype='int32', name='questions1')
    questions2_input = Input(shape=(140,), dtype='int32', name='questions2')

    x1 = Embedding(output_dim=512, input_dim=lexlen, input_length=140)(questions1_input)

    lstm_out1 = LSTM(64, go_backwards=True, activation='sigmoid', dropout=0.5)(x1)
    auxiliary_output1 = Dense(1, activation='sigmoid', name='aux_output1')(lstm_out1)

    x2 = Embedding(output_dim=512, input_dim=lexlen, input_length=140)(questions2_input)
    lstm_out2 = LSTM(64, go_backwards=True, activation='sigmoid', dropout=0.5)(x2)

    auxiliary_output2 = Dense(1, activation='sigmoid', name='aux_output2')(lstm_out2)

    x = keras.layers.concatenate([lstm_out1, lstm_out2])

    xd = Dense(32, activation='sigmoid')(x)

    main_output = Dense(1, activation='sigmoid', name='main_output')(xd)

    model = Model(inputs=[questions1_input, questions2_input],
                  outputs=[main_output, auxiliary_output1, auxiliary_output2])

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  loss_weights=[1., 0.2, 0.2])

    model.fit([q1array_train, q2array_train], [labels_train, labels_train, labels_train],
              epochs=5, batch_size=512)
    # serialize model to JSON
    model_json = model.to_json()
    with open("model_multi_out_sig.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_multi_out_sig.h5")
    print("Saved model to disk")

    # load json and create model
    # json_file = open('model_multi_out.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(loaded_model_json)
    # # load weights into new model
    # model.load_weights("model_multi_out.h5")
    # print("Loaded model from disk")
    # plot_model(model, to_file='model_multi_out.png')

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  loss_weights=[1., 0.2, 0.2], metrics=[keras.metrics.binary_accuracy])
    scores = model.evaluate([q1array_test, q2array_test], [labels_test, labels_test, labels_test], batch_size=1024)
    print("%s: %.2f%%" % (model.metrics_names[4], scores[4] * 100))


build_model()
