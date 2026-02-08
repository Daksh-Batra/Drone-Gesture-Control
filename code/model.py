import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
df = pd.read_csv("dataset.csv")

X=df.drop("label",axis=1)
y=df["label"]
y_encoded=[]
#map={"forward":0, "backward":1, "left":2, "right":3, "up":4, "down":5}
#for label in y:
#    y_encoded.append(map[label])
y_encoded=le.fit_transform(y)

X_train, X_test , y_train, y_test = train_test_split(X, y_encoded,test_size=0.2, random_state=0, stratify=y_encoded)

model=RandomForestClassifier(n_estimators=300, random_state=0)

model.fit(X_train, y_train)
y_pred=model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)
print(classification_report(y_test, y_pred, target_names=le.classes_))

joblib.dump(model, "hand_gesture_model.pkl") #to avoid retrainign of model each time
joblib.dump(le, "label_encoder.pkl") #to save which label is for which number