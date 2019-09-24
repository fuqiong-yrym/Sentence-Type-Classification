When designing artificial intelligence based communication tools such as chatbot we often need to process sentences to aid comprehension and to yield automatic response. In this respository, a LSTM based learning model is devised to classify sentence types. The output categories include Statement, Imperative and Question.

# Why LSTM is selected?
Intuitively, there are at least two basic properties that could be used to distinguish the types of sentences. One is the meaning of each word and the other is the order of the words in the sentence. The effective model that I can think of that could fully take account for these two properties is, LSTM with word embedding.

While word embedding could account for the meaning of the word, the order of the words in the sentence almost determine its type.  For example, the followings are some basic and simple rules that differentiate the sentence types:
   1. The imperative sentence starts with a verb;
   2. The question sentence usually starts with Is, Can, Do, Where, When, What, How, Why;
   3. The statements usually starts with a noun.
 
LSTM is well known for its ability to selectively remember the useful features long time ago, which is the exactly desired attribute for sentence classification. Think about how the statement sentence composes itself: subject + verb + preposition + location/time. If LSTM could remember such a structure, it is able to predict that the sentence with such structure is a statement. As mentioned above, questions usually starts with question words such as how, what, when. When encountering them, the trained LSTM will tell itself: “ ok, I see the first word is a question word, this sentence is probably a question, I will remember this. But wait, I see a verb at a later position, ok, now it is a statement instead of a question. ” With the ability of memorizing features long time ago, LSTM is a promising model for sentence classification.

# Datasets:
The folder of My_data contains the two .csv files: training set containing 208 examples and test set containing 30 examples. The datasets are manually created by myself.

# GloVe model:
The [GloVe model](https://nlp.stanford.edu/data/) is used as the word embedding model. 

# Training, testing and predicting
 1. Input the following in the command line to train the model
 
     ```bash
     python Sentence_cls_train.py
     ```
     
     This will train a LSTM based on the training dataset, output the training and testing accuracy, output the mislabelled examples and save the model as sentence_cls_model.h5

  
 2. Run the following to load the pretrained model and to predict the type of a sentence:

     ```bash
     python run_me.py
     ```  

   This will ask you to input a sentence and the model will output the predicted type. 

# Required library:
Numpy, TensorFlow, Keras

# Constraints: 
1. The sentence must have less than or equal to 13 words. This could be longer if we set the maximum length to be a bigger number.
2. There is no error handling if the word in the sentence is not included in the GloVe word embedding, which contains 40000 words. If the word is not found, the program will simply crash with a key error. This could be improved by adding <UNK> or by selecting a larger corpus.

