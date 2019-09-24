
from my_utils import *
from keras.models import load_model


word_to_index, index_to_word, word_to_vec = load_glove_embedding('my_data/glove.6B.50d.txt')
model = load_model('sentence_cls_model.h5')
maxLen = 13
print('Loading the pre-trained LSTM model...')

while(True):
    
    print("Do you want to test the type of a sentence? (Y/N)")
    ipt = input()
    if(ipt == 'Y'):
        
        print("Please input a sentence with less than 13 words:")
        sentence_input = input()
        if(len(sentence_input.split()) > 13):
            print("Sorry. Sentence is too long. Please try again:")
            continue
        sentence_test = np.array([sentence_input])
        sentence_test_indices = sentences_to_indices(sentence_test, word_to_index, maxLen)
        mapping = {0: 'S', 1: 'I', 2: 'Q'}
        cls_num = np.argmax(model.predict(sentence_test_indices))
        print(mapping[cls_num])
    elif(ipt == 'N'):
        break
    else:
        continue