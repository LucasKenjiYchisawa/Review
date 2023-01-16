import tensorflow as tf
from tensorflow import keras
import numpy as np

#To load the model we can use

model=keras.models.load_model("fitmodel.h5")


#To change a text to a form this model can "read" we have to use this code

def review_encode(s):
	encoded = [1]

	for word in s:
		if word.lower() in word_index:
			encoded.append(word_index[word.lower()])
		else:
			encoded.append(2)

	return encoded
#Opening the text file, treating it to remove characters that are not included in the word index, and then encoding it.
    
with open("test.txt", encoding="utf-8") as f:
	for line in f.readlines():
		nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
		encode = review_encode(nline)
		encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
		predict = model.predict(encode)#Using the model to predict
		print(line)#Print the review
		print(encode)
		print(predict[0])#Print the prediction whether it is a positive or negative review.

