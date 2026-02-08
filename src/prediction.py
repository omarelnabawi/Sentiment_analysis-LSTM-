#importing necessary libraries
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from prepocessing import preprocess_data

#Required for loading the tokenizer
# Required for loading the tokenizer
model = load_model("models/model_epoch(1).h5")
x_train, y_train, x_test, y_test, tokenizer = preprocess_data()

def predict_sentiment(model, tokenizer, text):
    # Preprocess the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=500)

    # Predict the sentiment
    prediction = model.predict(padded_sequence)
    sentiment = 'positive' if prediction[0][0] >= 0.5 else 'negative'

    return sentiment

print(predict_sentiment(model, tokenizer, "This movie was fantastic! I really enjoyed it."))
print(predict_sentiment(model, tokenizer, "This movie was terrible. I hated it."))