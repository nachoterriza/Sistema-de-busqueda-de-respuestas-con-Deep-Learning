from keras.layers import Input, TimeDistributed, RepeatVector
from keras.layers.core import Activation, Dense, Dropout, Permute
from keras.layers.embeddings import Embedding
from keras.layers.merge import add, concatenate, dot
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.callbacks import ModelCheckpoint,  EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras import metrics
from keras import optimizers
import pickle
import numpy as np
import en_core_web_md
from numpy import argmax
#we load previously saved data.
input_train = pickle.load(open('input_train.pickle', 'rb'))
input_test = pickle.load(open('input_test.pickle', 'rb'))
id2word = pickle.load(open('id2word.pickle', 'rb'))
vocab_size = len(id2word)

nlp = en_core_web_md.load(disable=['parser', 'tagger', 'ner'])

EMBEDDINGS_SIZE=300 #vector's length
LATENT_SIZE = 300
#we split the data into stories, questions and answers.
XsTrain, XqTrain, YTrain = input_train
XsTest, XqTest, YTest = input_test

max_len = 0
for y in YTrain:
    if len(y) > max_len:
        max_len = len(y)
for y in YTest:
    if len(y) > max_len:
        max_len = len(y)   

def inputs_generator(input_data, vocab_size, max_len): #generate each piece of the input (check table 6.1 in the memory) .
    Xs, Xq, Y = input_data
    while True:    
        for i in range(0, len(Xs)):
                stories, questions, Yx, Yy = [], [], [], []
                for j in range(1, len(Y[i])):
                    codification = np.zeros(vocab_size)
                    #split into input and output pair
                    in_seq, out_seq = Y[i][:j], Y[i][j]
                    if out_seq != 0 and out_seq != 2:
                        in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
                        codification[out_seq] = 1
                        stories.append(Xs[i])
                        questions.append(Xq[i])
                        Yx.append(in_seq)
                        Yy.append(codification)
                yield [np.array(stories, dtype=np.float16), np.array(questions, dtype=np.float16), np.array(Yx, dtype=np.float16)], np.array(Yy, dtype=np.float16)


def define_model(vocab_size, max_len): #check figure 6.3 in the memory
    #context vector
    story_inputs = Input(shape=(len(XsTrain[0]), ))
    question_inputs = Input(shape=(len(XqTrain[0]),))
    
    story_encoder = Embedding(input_dim=vocab_size, output_dim=EMBEDDINGS_SIZE, input_length=len(XsTrain[0]))(story_inputs)
    story_encoder = Dropout(0.3)(story_encoder)
    print('story encoder done!')
    
    #question encoder memory
    question_encoder = Embedding(input_dim=vocab_size, output_dim=EMBEDDINGS_SIZE, input_length=len(XqTrain[0]))(question_inputs)
    question_encoder = Dropout(0.3)(question_encoder)
    print('question encoder done!')

    
    match = dot([story_encoder, question_encoder], axes=[2,2])
    
    #encode story into vector space of question
    story_encoder_c = Embedding(input_dim=vocab_size, output_dim=len(XqTrain[0]), input_length=len(XsTrain[0]))(story_inputs)
    story_encoder_c = Dropout(0.3)(story_encoder_c)
    print('story encoder c done!')

    #combine match and story vectors
    response = add([match, story_encoder_c])
    response = Permute((2, 1))(response)
    print('response done!')

    #combine response and queztion vectors
    answer = concatenate([response, question_encoder], axis=-1)

    encoder = Bidirectional(LSTM(LATENT_SIZE), merge_mode="sum", name="encoder_lstm")(answer)
    encoder = Dropout(0.5)(encoder)
    print('encoder done!')
    
    #sequence model
    answers_inputs = Input(shape=(max_len,))
    se1 = Embedding(vocab_size, EMBEDDINGS_SIZE, mask_zero=True)(answers_inputs)
    se2 = Dropout(0.5)(se1)
    se3 = Bidirectional(LSTM(LATENT_SIZE), merge_mode="sum", name="encoder2_lstm")(se2)
    print('sequence model done!')
    
    #decoder model
    decoder = add([encoder, se3])
    decoder = Dense(LATENT_SIZE, activation='relu')(decoder)
    outputs = Dense(vocab_size, activation='softmax')(decoder)
    print('output done!')
    
   
    # tie it together [image, seq] [word]
    model = Model(inputs=[story_inputs, question_inputs, answers_inputs], outputs=outputs)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0), metrics=['acc'])
    # summarize model
    model.summary()
    return model


model = define_model(vocab_size, max_len)
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

BATCH_SIZE=5
NUM_EPOCHS=1000

model.fit_generator(inputs_generator(input_train, vocab_size, max_len), steps_per_epoch=len(XsTrain)/BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=inputs_generator(input_train, vocab_size, max_len), validation_steps=len(XsTest)/BATCH_SIZE, callbacks=[checkpoint], verbose=2)

with open('max_answer_len.pickle', 'wb') as handle:
    pickle.dump(max_len, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("finished")