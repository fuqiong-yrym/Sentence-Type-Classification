import numpy as np
from my_utils import *
np.random.seed(0)
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
np.random.seed(1)

X_train, Y_train = read_csv('my_data/myTrainDataset.csv')
X_test, Y_test = read_csv('my_data/myTestDataset.csv')
num_classes = 3
maxLen = 13


Y_oh_train = to_categorical(Y_train, num_classes)
Y_oh_test = to_categorical(Y_test, num_classes)

word_to_index, index_to_word, word_to_vec = load_glove_embedding('my_data/glove.6B.50d.txt')

def glove_embedding_layer(word_to_vec, word_to_index):
    vocab_size = len(word_to_index) + 1
    embedding_dim = word_to_vec['bus'].shape[0]
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, index in word_to_index.items():
        embedding_matrix[index] = word_to_vec[word]
    
    embedding_layer = Embedding(vocab_size, embedding_dim, trainable=True)
    embedding_layer.build((None,))
    embedding_layer.set_weights([embedding_matrix])
    
    return embedding_layer

embedding_layer = glove_embedding_layer(word_to_vec, word_to_index)

def Sentence_type_classification(input_shape, word_to_vec, word_to_index):
    sentence_indices = Input(input_shape, dtype = 'int32')
    embedding_layer = glove_embedding_layer(word_to_vec,word_to_index)
    embeddings = embedding_layer(sentence_indices)
    
    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(3)(X)
    X = Activation('softmax')(X)
    
    model = Model(sentence_indices, X)
    
    return model

model = Sentence_type_classification((maxLen,), word_to_vec, word_to_index)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
model.fit(X_train_indices, Y_oh_train, epochs=60, batch_size=32, shuffle=True)

print('Training done!')
print('====================================================')
print()
print('Start testing!-------------')
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
loss, acc = model.evaluate(X_test_indices, Y_oh_test)
print('Test accuracy: ', acc)
print()

print('These are the mislabelled examples in the test set')

if(acc != 1.0):
    X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
    pred = model.predict(X_test_indices)
    
    for i in range(len(X_test)):
        num = np.argmax(pred[i])
        if(num != Y_test[i]):
            print('Expected :', Y_test[i], ' prediction: ', X_test[i],  num)
print('0 - statement.   1 - imperative.   2 - question')
print()
print('Saving the model...')
#model.save('sentence_cls_model.h5')
print('Model saved!')







