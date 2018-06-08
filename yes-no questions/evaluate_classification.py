from pickle import load
from keras.models import load_model
import numpy as np
from keras import backend as K
import keras.losses


#we load the previously saved pickles and the recently trained model
model = load_model('model.h5')
input_test = load(open('input_test.pickle', 'rb'))
output_test = load(open('output_test.pickle', 'rb'))
id2word = load(open('id2word.pickle', 'rb'))

def print_question(question, id2word): #this function decodes the question for printing
    s = ''
    for w in question:
        if(w != 0):
            s = s + id2word[w] + ' '
    print(s)


def evaluate_model(model, data, output_test, id2word):
    predicted =  list()
    # step over the whole set
    n = 0
    stories, questions = data
    for story, question in zip(stories, questions):
        #generate description
        yhat = model.predict([np.array([story]), np.array([question])], verbose=0)
        # clean up prediction
        print_question(question, id2word)
        if yhat[0][0] > 0.5:
            print('Yes')
            #we keep count of questions predicted as yes
            n += 1
            predicted.append(1)
        else:
            print('No')
            predicted.append(0)
        print('')
        
        # store actual and predicted
    print(n)   
    correct = 0
    wrong = 0
    for i in range(0, len(predicted)):
        if predicted[i] == output_test[i]:
            correct += 1
        else:
            wrong += 1
    
    print('Correct answers: ', correct)
    print('Wrong answers: ', wrong)
    
evaluate_model(model, input_test, output_test, id2word)
