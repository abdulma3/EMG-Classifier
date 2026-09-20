import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('emg_features.csv')
X = df[['MAV', 'RMS', 'ZC', 'WL', 'STD']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=['rest', 'light_flex', 'hard_flex']))
print("\nFull Report:\n", classification_report(y_test, y_pred))

# More reliable accuracy estimate than a single train/test split
scores = cross_val_score(model, X, y, cv=5)
print(f"\nCross-val accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

# Which features actually drove the model's decisions
print("\nFeature importances (MAV, RMS, ZC, WL, STD):")
print(model.feature_importances_)

joblib.dump(model, 'emg_model.pkl')
print("\nModel saved to emg_model.pkl")