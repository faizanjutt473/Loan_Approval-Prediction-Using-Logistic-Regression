'''
LOGISTIC REGRESSION CODE - LINE BY LINE
EXPLANATION:


1. LIBRARIES IMPORT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

python language libraries for data manipulation, visualization, and machine learning.

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc



 2. HEADER AND SEED
print("="*60)
print(" LOGISTIC REGRESSION - PROFESSIONAL PROJECT")
print("="*60)
np.random.seed(42)


 3. DATA GENERATION
 
 X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    weights=[0.7, 0.3],
    random_state=42
)
feature_names = ['Age', 'Income', 'Credit_Score', 'Loan_Amount', 'Years_Experience']
df = pd.DataFrame(X, columns=feature_names)
df['Target'] = y
print(df.head())


4. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


5. FEATURE SCALING


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


6. MODEL TRAINING


model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)


7. MAKING PREDICTIONS


y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]


8. MODEL EVALUATION


print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))
print(cm)


9. VISUALIZATION

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',...)
plt.plot(fpr, tpr,...)
plt.show()



10. FEATURE IMPORTANCE

coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': model.coef_[0]})
coef_df['Absolute_Importance'] = np.abs(coef_df['Coefficient'])
coef_df = coef_df.sort_values('Absolute_Importance', ascending=False)
print(coef_df)





'''
