import json
import os
kaggle_dict=json.load(open("Data/kaggle.json"))

os.environ["KAGGLE_USERNAME"]=kaggle_dict["username"]
os.environ["KAGGLE_KEY"]=kaggle_dict["key"]

