import tensorflow as tf
from tensorflow import keras
import numpy as np

#Import the data from keras
data = keras.datasets.imdb

#Separating the data into train data and test data
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

#Getting the word index from the word list from keras
word_index = data.get_word_index() 

word_index={k:(v+3) for k, v in word_index.items()}
# Setting the following code with the index from 0 to 3
word_index{“<PAD>”} =0
word_index{“<START>”}=1
word_index{“<UNK>”}=2
word_index{"<UNUSED>"}=3

#Turning the text into numerical values using the word index
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#decodes the text
def decode_review(text):
	return " ".join([reverse_word_index.get(i,”?”) for i in text])
#Setting the training data and making sure the text used is always 250 words long
train_data=keras.preprocessing.sequence.pad_sequences(train_data,value= word_index[“<PAD>”], padding=“post”,maxlen=250)

#Setting the testing data and making sure the text used is always 250 words long
test_data=keras.preprocessing.sequence.pad_sequences(test_data,value= word_index[“<PAD>”], padding=“post”,maxlen=250)

#Creating the neural network
    
model=keras.sequential()

#Embedding layers look at the words surrounding each other to determine the context of the use of the word and group similar words.

model.add(keras.layers.Embedding(88000, 16))
#First hidden layer
model.add(keras.layers.GlobalAveragePoolingID())
#Second Hidden layer
model.add(keras.layers.Dense(16, activation=“relu”))
#Output layer. 0 indicates it is a negative review and 1 is a positive review.
model.add(keras.layers.Dense(1, activation=“sigmoid")

#Compiling the model
model.compile(optimizer=“adam”, loss=“binary_crossentropy", metrics=["accuracy”])

#Separating validation data and training data
x_val=train_data[:10000]
x_train=train_data[10000:]

y_val=train_labels[:10000]
y_train=train_labels[10000:]

#Fitting the model using the data we have
fitmodel=model.fit(x_train,y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

#Checking the accuracy
results=model.evaluate(test_data, test_labels)

print(results)


#To save the model we can use
model.save("fitmodel.h5”)
