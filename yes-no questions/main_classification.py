from keras.layers import Input
from keras.layers.core import Dense, Dropout, Permute
from keras.layers.embeddings import Embedding
from keras.layers.merge import add, concatenate, dot
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from pickle import load

#we load the previously saved data for the input
input_train = load(open('input_train.pickle', 'rb'))
input_test = load(open('input_test.pickle', 'rb'))

#we split the data we just loaded into stories and questions
XsTrain, XqTrain = input_train
XsTest, XqTest = input_test

#we load the previously saved data for the output
output_train = load(open('output_train.pickle', 'rb'))
output_test = load(open('output_test.pickle', 'rb'))

#we load the previously saved data for the vocabulary size
vocab_size = load(open('vocab_size.pickle', 'rb'))

EMBEDDINGS_SIZE=64
LATENT_SIZE=32

#inputs
story_input = Input(shape=(len(XsTrain[0]),))
question_input = Input(shape=(len(XqTrain[0]),))
print('input done!')

#story encoder memory
story_encoder = Embedding(input_dim=vocab_size, output_dim=EMBEDDINGS_SIZE, input_length=len(XsTrain[0]))(story_input)
story_encoder = Dropout(0.3)(story_encoder)
print('story encoder done!')

#question encoder memory
question_encoder = Embedding(input_dim=vocab_size, output_dim=EMBEDDINGS_SIZE, input_length=len(XqTrain[0]))(question_input)
question_encoder = Dropout(0.3)(question_encoder)
print('question encoder done!')

#match between story and question
match = dot([story_encoder, question_encoder], axes=[2,2])
print('match done!')

#encode story into vector space of question
story_encoder_c = Embedding(input_dim=vocab_size, output_dim=len(XqTrain[0]), input_length=len(XsTrain[0]))(story_input)
story_encoder_c = Dropout(0.3)(story_encoder_c)
print('story encoder c done!')

#combine match and story vectors
response = add([match, story_encoder_c])
response = Permute((2, 1))(response)
print('response done!')

#combine response and question vectors
answer = concatenate([response, question_encoder], axis=-1)
answer = LSTM(LATENT_SIZE)(answer)
answer = Dropout(0.3)(answer)

answer = Dense(round(LATENT_SIZE * 0.5), activation='relu')(answer)
output = Dense(1, activation='sigmoid')(answer)

print('output done!')

model = Model(inputs=[story_input, question_input], outputs=output)
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=["accuracy"])
print('model compiled!')

# summarize model
model.summary()

BATCH_SIZE=5
NUM_EPOCHS=300

#we create a function to tell the model when and how to save itself
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit([XsTrain, XqTrain], [output_train], batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=([XsTest, XqTest], [output_test]), callbacks=[checkpoint], verbose=2)
print('model fitted!')
