#import libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer

#LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) architecture that is particularly effective for sequence prediction problems. It is designed to capture long-term dependencies in sequential data, making it well-suited for tasks such as natural language processing, time series forecasting, and sentiment analysis.
def build_model():
    model=Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=500))
    model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())
    return model