import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
from tabulate import tabulate

# Read data
data = pd.read_csv('MDT32_final_project_dataset.csv')

# Clean column names
data.columns = data.columns.str.strip()

# Create Combined Target
data['Combined Target'] = data.apply(
    lambda row: 'No Failure' if row['Machine failure'] == 0 else row['Failure Type'], axis=1
)

# Encode categorical column
le = LabelEncoder()
data['Type'] = le.fit_transform(data['Type'])

# Define feature columns in explicit order
feature_cols = [
    'Type',
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

# Drop unnecessary columns
data = data[feature_cols + ['Combined Target']]

# Handle missing values
if data.isnull().sum().sum() > 0:
    data[feature_cols] = data[feature_cols].fillna(data[feature_cols].mean(numeric_only=True))

# Split features and target
X = data[feature_cols]
y = data['Combined Target']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to balance data
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Train Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_balanced, y_train_balanced)

# Predict and evaluate
y_pred = rf_model.predict(X_test_scaled)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
cm_table = [[class_name] + list(cm[i]) for i, class_name in enumerate(rf_model.classes_)]
headers = [''] + list(rf_model.classes_)
print("\n=== Confusion Matrix ===")
print(tabulate(cm_table, headers=headers, tablefmt='fancy_grid', floatfmt='.0f'))

# Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
report_table = []
for label in report.keys():
    if label not in ['accuracy', 'macro avg', 'weighted avg']:
        report_table.append([
            label,
            report[label]['precision'],
            report[label]['recall'],
            report[label]['f1-score'],
            report[label]['support']
        ])
report_table.append([
    'macro avg',
    report['macro avg']['precision'],
    report['macro avg']['recall'],
    report['macro avg']['f1-score'],
    report['macro avg']['support']
])
report_table.append([
    'weighted avg',
    report['weighted avg']['precision'],
    report['weighted avg']['recall'],
    report['weighted avg']['f1-score'],
    report['weighted avg']['support']
])
print("\n=== Classification Report ===")
print(tabulate(report_table, headers=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'], tablefmt='fancy_grid', floatfmt='.2f'))

# Save model and preprocessing objects
joblib.dump(rf_model, 'rf_combined_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'labels.pkl')

print("\nFinished")