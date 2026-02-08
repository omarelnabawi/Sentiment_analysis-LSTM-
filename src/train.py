from prepocessing import preprocess_data
from model import build_model
import os
x_train,y_train,x_test,y_test,tokenizer=preprocess_data()
model=build_model()
i=1
model.fit(x_train, y_train, epochs=i, batch_size=64, validation_data=(x_test, y_test))
loss, accuracy=model.evaluate(x_test, y_test)
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists(f'models/model_epoch({i}).h5'):
    model.save(f'models/model_epoch({i}).h5')
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)