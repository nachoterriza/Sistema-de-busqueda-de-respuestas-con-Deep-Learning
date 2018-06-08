from numpy import argmax
from pickle import load
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
import en_core_web_md
import numpy as np


nlp = en_core_web_md.load(disable=['parser', 'tagger', 'ner'])
#we load the relevant data
data_test = load(open('dataTest.pickle', 'rb'))
data_train = load(open('dataTrain.pickle', 'rb'))
id2word = load(open('id2word.pickle', 'rb'))
max_answer_len = load(open('max_answer_len.pickle', 'rb'))
model = load_model('model.h5')
vocab_size = len(id2word)
word2id = {v:k for k, v in id2word.items()}

def parsed_segment(X):
    segment=[]
    for w in X.split():
        try:
            segment.append(word2id[w])
        except KeyError:
            segment.append(word2id['UNK'])
    return segment

# generate an answer for a story and a question
def generate_answer(model, story, question, max_length, id2word):
    # seed the generation process
    in_text = 'STARTSEQ'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = parsed_segment(in_text)
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([np.array([story]), np.array([question]), np.array(sequence)], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        try:
            word = id2word[yhat]
        except KeyError:
            # stop if we cannot map the word
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'ENDSEQ':
            break
    return in_text

# remove start/end sequence tokens from a answer
def cleanup_answer(answer):
	# remove start of sequence token
	index = answer.find('STARTSEQ ')
	if index > -1:
		answer = answer[len('STARTSEQ '):]
	# remove end of sequence token
	index = answer.find(' ENDSEQ')
	if index > -1:
		answer = answer[:index]
    # remove end of sequence token
	index = answer.find('ENDSEQ')
	if index > -1:
		answer = answer[:index]
	return answer

def print_question(question):
    s = ''
    for w in question:
        if(w != 0):
            s = s + id2word[w] + ' '
    print(s)
	
# evaluate the skill of the model
def evaluate_model(model, data, max_length, id2word):
    actual, predicted = list(), list()
    # step over the whole set
    stories, questions, answers = data
    for story, question, answer in zip(stories, questions, answers):
        #generate description
        yhat = generate_answer(model, story, question, max_length, id2word)
        # clean up prediction
        yhat = cleanup_answer(yhat)
        print_question(question)
        print(yhat)
        print('')
        # store actual and predicted
        actual.append(answer.split())
        predicted.append(yhat.split())
	# calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
    
evaluate_model(model, data_train, max_answer_len, id2word)