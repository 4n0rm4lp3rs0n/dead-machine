import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
from tabulate import tabulate

# Đọc dữ liệu từ file CSV
data = pd.read_csv('MDT32_final_project_dataset.csv')

# Tạo cột Combined Target
data['Combined Target'] = data.apply(
    lambda row: 'No Failure' if row['Machine failure'] == 0 else row['Failure Type'], axis=1
)

# Tiền xử lý dữ liệu

le = LabelEncoder()
data['Type'] = le.fit_transform(data['Type'])

# Loại bỏ các cột không cần thiết
data = data.drop(['UDI', 'Product ID', 'Machine failure', 'Failure Type'], axis=1)

# Kiểm tra giá trị thiếu
if data.isnull().sum().sum() > 0:
    data = data.fillna(data.mean())

# Tách đặc trưng (X) và mục tiêu (y)
X = data.drop('Combined Target', axis=1)
y = data['Combined Target']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# scaler.transform fucks the pd.DataFrame to Numpy array, which fucks the web

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# print (type(X_test_scaled))

# Xử lý mất cân bằng lớp bằng SMOTE
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)


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

y_pred = rf_model.predict(X_test_scaled)


cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
cm_table = [[class_name] + list(cm[i]) for i, class_name in enumerate(rf_model.classes_)]
headers = [''] + list(rf_model.classes_)
print("\n=== Confusion Matrix ===")
print(tabulate(cm_table, headers=headers, tablefmt='fancy_grid', floatfmt='.0f'))


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

# Lưu mô hình và scaler
joblib.dump(rf_model, 'rf_combined_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'labels.pkl')
print("\nFinished")