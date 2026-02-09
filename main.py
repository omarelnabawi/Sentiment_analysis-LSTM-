import sys
import os
from src.train import model_train
from src.prediction import predict_sentiment


k = predict_sentiment(model_train(i=5),'i enjoy')
print(k)
