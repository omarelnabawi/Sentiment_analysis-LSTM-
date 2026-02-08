# Import necessary libraries
from data_loader import load_data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_data():
    # Load the dataset
    df=load_data()
    #print(df.value_counts('sentiment'))

    # Encode the labels
    le=LabelEncoder()
    df['sentiment']=le.fit_transform(df['sentiment'])
    #print(le.classes_)

    #Split the data into training and testing sets
    X_train,X_test,y_train,y_test=train_test_split(df['review'],df['sentiment'],test_size=0.25,random_state=42)

    #Tokenization and padding
    tokenizer=Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    x_train=pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=500)
    x_test=pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=500)
    

    return x_train, y_train, x_test, y_test, tokenizer
#x_train, y_train, x_test, y_test=preprocess_data()
#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)