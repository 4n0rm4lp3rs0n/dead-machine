import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

df = pd.read_csv('./MDT32_final_project_dataset.csv')

le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])
df = df.drop(['Product ID', 'UDI'], axis = 1)

df['Failure Type'] = df['Failure Type'].astype(str)
df.loc[df['Machine failure'] == 0, 'Failure Type'] = 'No Failure'

le_failure = LabelEncoder()
df['Failure Type'] = le_failure.fit_transform(df['Failure Type'])
reverse_failure_map = {v: k for k, v in enumerate(le_failure.classes_)}

X = df.drop(columns=['Machine failure', 'Failure Type'])
y = df[['Machine failure', 'Failure Type']]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

multi_clf = MultiOutputClassifier(RandomForestClassifier(random_state=42))
multi_clf.fit(x_train, y_train)

y_pred = multi_clf.predict(x_test)

# Separate predictions
machine_failure_pred = y_pred[:, 0]
failure_type_pred = y_pred[:, 1]
machine_failure_true = y_test['Machine failure']
failure_type_true = y_test['Failure Type']


# Evaluate Machine Failure
print("== Machine Failure Classification ==")
print(classification_report(machine_failure_true, machine_failure_pred))

# Evaluate Failure Type only where machine actually failed
mask = machine_failure_true == 1
print("\n== Failure Type Classification (on failures only) ==")
print(classification_report(
    failure_type_true[mask],
    failure_type_pred[mask],
    target_names=le_failure.classes_
))

cm_fail = confusion_matrix(machine_failure_true, machine_failure_pred)
disp_fail = ConfusionMatrixDisplay(confusion_matrix=cm_fail, display_labels=["No Failure", "Failure"])
disp_fail.plot(cmap='Blues')
plt.title("Confusion Matrix - Machine Failure")
plt.show()

mask = machine_failure_true == 1
cm_ftype = confusion_matrix(failure_type_true[mask], failure_type_pred[mask])

disp_ftype = ConfusionMatrixDisplay(confusion_matrix=cm_ftype, display_labels=le_failure.classes_)
disp_ftype.plot(cmap='Oranges', xticks_rotation=45)
plt.title("Confusion Matrix - Failure Type (on failed machines)")
plt.show()

fail_probs = multi_clf.estimators_[0].predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(machine_failure_true, fail_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Machine Failure')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print("Machine Failure Accuracy:", accuracy_score(machine_failure_true, machine_failure_pred))
print("Machine Failure ROC AUC:", roc_auc_score(machine_failure_true, fail_probs))

print("\nFailure Type Accuracy (on failures only):",
      accuracy_score(failure_type_true[mask], failure_type_pred[mask]))