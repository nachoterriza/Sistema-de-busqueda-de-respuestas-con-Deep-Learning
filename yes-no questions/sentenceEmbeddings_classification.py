from keras.preprocessing.sequence import pad_sequences
from unicodedata import normalize
import collections
import numpy as np
import os
import pickle
import json
import re
import string
import en_core_web_md
np.set_printoptions(threshold=np.nan)

DATA_DIR="data"
TRAIN_FILE=os.path.join(DATA_DIR, "BioASQ-trainingDataset5b.json")
nlp = en_core_web_md.load(disable=['parser', 'tagger', 'ner'])

def clean_story(stories, re_print):
    clean_stories = []
    for story in stories:
       clean_sent = ''
       for sent in story:
            # normalize unicode characters
            sent = normalize('NFD', sent).encode('ascii', 'ignore')
            sent = sent.decode('UTF-8')
            for word in nlp(sent):
                # convert to lowercase
                word = word.text.lower()
                # remove non-printable chars form each token
                word = re_print.sub('', word)
                    
                #remove all empty strings
                if(word != ''):
                    clean_sent = clean_sent + ' ' + word
                    
       clean_stories.append(clean_sent)
        

    stories = clean_stories
    return stories


def clean_body(x, re_print):
    clean_x = []
    for sent in x:
        clean_sent = []
        # normalize unicode characters
        sent = normalize('NFD', sent).encode('ascii', 'ignore')
        sent = sent.decode('UTF-8')
        for word in nlp(sent):
            # convert to lowercase
            word = word.text.lower()
            # remove non-printable chars form each token
            word = re_print.sub('', word)

            #remove all empty strings
            if(word != ''):    
                clean_sent.append(word)
        clean_x.append(' '.join(clean_sent))
    x = clean_x
    return x
	
def clean_answer(x):
    clean_x = []
    for sent in x:
        # we append 1 if the answer was yes, 0 otherwise
        if sent[0].lower() == 'y':
            clean_x.append(1)
        else:
            clean_x.append(0)
    return clean_x


def get_data(infile):
    stories, body, answers = [], [], []
    with open(infile, 'r', encoding="utf8") as file:
        data = json.load(file)
    for question in data['questions']:
        if question['type'] == 'yesno': #here we choose the type of question
            story = []
            for snippet in question['snippets']:
                story.append(snippet['text']) #get each story and concatenate
            stories.append(story)
            body.append(question['body'])
            answers.append(question['exact_answer'])

    # we take 20% for the test, the rest for the training
    test_stories = stories[:round(len(stories)/5)]
    test_body = body[:round(len(body)/5)]
    test_answers = answers[:round(len(answers)/5)]
    stories = stories[round(len(stories)/5):]
    body = body[round(len(body)/5):]
    answers = answers[round(len(answers)/5):]
    
    print('Number training questions:' , len(body))
    print('Number testing questions:' , len(test_body))
    
    print(len(test_stories))
    print(len(stories))
    
    # prepare regex for char filtering and fix the case of the word to lowercase
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    
    stories = clean_story(stories, re_print)
    test_stories = clean_story(test_stories, re_print)
    
    body = clean_body(body, re_print)
    test_body = clean_body(test_body, re_print)   
    
    answers = clean_answer(answers)
    test_answers = clean_answer(test_answers)
    
    return stories, body, answers, test_stories, test_body, test_answers

stories, body, answers, test_stories, test_body, test_answers = get_data(TRAIN_FILE)
data_train = stories, body
data_test = test_stories, test_body
print('get_data done!')

def get_wordfreqs_len(x, word_freqs, sum_number):
    words = []
    max_x_len = 0
    for word in nlp(x):
        word_freqs[word.text] += sum_number
        words.append(word.text)
    if len(words) > max_x_len:
        max_x_len = len(words)
    return word_freqs, max_x_len

def get_max(x, y):
    if(x > y):
        return x
    else:
        return y
        
def build_vocab(train_data, test_data): #here we calculate the vocabulary frequency and the maximum length of the inputs (stories, questions and answers)
    word_freqs = collections.Counter()
    max_story_len = 0
    max_question_len = 0
    for stories, questions in [train_data, test_data]:
        for i in range(0, len(stories)): #we know that the lenght of questions, stories and answers is the same
            
            word_freqs, max_segment_len = get_wordfreqs_len(stories[i], word_freqs, 1)
            max_story_len = get_max(max_segment_len, max_story_len)
            
            word_freqs, max_segment_len = get_wordfreqs_len(questions[i], word_freqs, len(stories[i]))
            max_question_len = get_max(max_segment_len, max_question_len)
            
    return word_freqs, max_story_len, max_question_len

word_freqs, STORIES_SEQUENCE_LEN, QUESTIONS_SEQUENCE_LEN= build_vocab(data_train, data_test)
VOCAB_SIZE = len(word_freqs) + 4  #We pick 80% of the words in the vocabulary
print('vocab_size done!')

#The ids are useful for making the sentence embedding in the model
word2id = {}
word2id['PAD'] = 0
word2id['UNK'] = 1
word2id['STARTSEQ'] = 2
word2id['ENDSEQ'] = 3
for v, (k, _) in enumerate(word_freqs.most_common(VOCAB_SIZE - 4)):
    word2id[k] = v + 4
id2word = {v:k for k, v in word2id.items()}

def parsed_segment(X):
    segment=[]
    for w in nlp(X):
        try:
            segment.append(word2id[w.text])
        except KeyError:
            segment.append(word2id['UNK'])
    return segment

def parsed_data(data, word2id): #here we change the words into ids
    Xs, Xq = [], []
    stories, questions = data
    for story, question in zip(stories, questions):
            Xs.append(parsed_segment(story))
            Xq.append(parsed_segment(question))
    Xs = pad_sequences(Xs, maxlen=STORIES_SEQUENCE_LEN)
    Xq = pad_sequences(Xq, maxlen=QUESTIONS_SEQUENCE_LEN)
    return Xs, Xq

XsTrain, XqTrain= parsed_data(data_train, word2id)
XsTest, XqTest = parsed_data(data_test, word2id)
print('parsed data done!')

input_train = np.array(XsTrain), np.array(XqTrain)
input_test = np.array(XsTest), np.array(XqTest)
output_train = np.array(answers)
output_test = np.array(test_answers)

#we save the relevant data in pickle files
with open('input_train.pickle', 'wb') as handle:
    pickle.dump(input_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('input_test.pickle', 'wb') as handle:
    pickle.dump(input_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('output_train.pickle', 'wb') as handle:
    pickle.dump(output_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
with open('output_test.pickle', 'wb') as handle:
    pickle.dump(output_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('vocab_size.pickle', 'wb') as handle:
    pickle.dump(VOCAB_SIZE, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('id2word.pickle', 'wb') as handle:
    pickle.dump(id2word, handle, protocol=pickle.HIGHEST_PROTOCOL)
