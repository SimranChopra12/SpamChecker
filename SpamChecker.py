import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split # split the dataset into parts to first train the model, and then train it 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report, confusion_matrix
import joblib


df = pd.read_csv("data/SMSSpamCollection", sep="\t", header=None, names=["label","text"])
#print(df.head())

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text"])
#print(X)


# Step 3 : train the model

y = df["label"].map({"ham":0, "spam":1}) 
# converting to score - ham=0 and spam=1

X_train, X_test, y_train, y_test =  train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
# trains the model
# iterates 1000 times to make sure it learns
# .fit is the command for model to learn
# clf is the classifier - the actual model

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
# Accuracy = (Number of correct predictions) / (Total predictions)


print("classification report: \n")

print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
print(confusion_matrix(y_test, y_pred))

joblib.dump(clf, "spam_classifier.joblib")
print("Model saved as spam_classifier.joblib")
# saving the model so we dont have to train again