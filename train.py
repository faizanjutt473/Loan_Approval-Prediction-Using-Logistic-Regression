import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc



# ================================
# LOGISTIC REGRESSION - PROFESSIONAL PROJECT
# ================================


print("="*60)
print(" LOGISTIC REGRESSION - PROFESSIONAL PROJECT")
print("="*60)

# For reproducibility
np.random.seed(42)

print("\n[1] Generating Synthetic Dataset...")
X, y = make_classification(
    n_samples=1000, # Total number of samples
    n_features=5, # Total number of features
    n_informative=3, # Number of informative features
    n_redundant=1, # Number of redundant features
    n_classes=2, # Binary classification
    weights=[0.7, 0.3], # Imbalanced classes: 70% Class 0, 30% Class 1
    random_state=42
)
feature_names = ['Age', 'Income', 'Credit_Score', 'Loan_Amount', 'Years_Experience']
df = pd.DataFrame(X, columns=feature_names)
df['Target'] = y
print(df.head())
print(f"\nDataset Shape: {df.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training 

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Model Prediction
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Model Evaluation

print("\n[5] Evaluating Model Performance...")
print("-"*60)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)


#VISUALIZATION

plt.figure(figsize=(12, 5))

#Confusion Matrix Heatmap

plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Rejected', 'Approved'],
            yticklabels=['Rejected', 'Approved'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

#ROC Curve

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


#FEATURE IMPORTANCE

print("\n[6] Feature Importance Based on Coefficients:")
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': model.coef_[0]})
coef_df['Absolute_Importance'] = np.abs(coef_df['Coefficient'])
coef_df = coef_df.sort_values('Absolute_Importance', ascending=False)
print(coef_df)

print("\n" + "="*60)
print(" MODEL TRAINING COMPLETED SUCCESSFULLY")
print("="*60)






