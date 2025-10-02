import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/SMSSpamCollection", sep="\t", header=None, names=["label","text"])
print(df.head())