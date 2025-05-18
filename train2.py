import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv('./ai4i2020.csv')

# Basic data exploration
print("DataFrame Info:")
print(df.info())
print("\nDataFrame Description:")
print(df.describe().T)
print("\nMachine Failure Value Counts:")
print(df["Machine failure"].value_counts())
print("\nColumn Names:")
print(df.columns)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nType Value Counts:")
print(df["Type"].value_counts())

# EDA
# Convert 'Type' column to numerical using Label Encoding
le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])
print("\nEncoded Type Value Counts:")
print(df['Type'].value_counts())

# Drop unnecessary columns
# df = df.drop(['Product ID', 'UDI', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
df = df.drop(['Product ID', 'UDI'], axis=1)

# Correlation heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.show()

# Histogram plotting function
def draw_histograms(dataframe, features, rows, cols):
    fig = plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax = fig.add_subplot(rows, cols, i+1)
        dataframe[feature].hist(bins=20, ax=ax, facecolor='Blue')
        ax.set_title(feature, fontsize=20, color='darkgreen')
    fig.tight_layout()
    plt.show()

# Plot histograms
draw_histograms(df, df.columns, 5, 3)

# Univariate analysis: Type vs. Machine Failure
plt.figure(figsize=(8, 6))
sns.countplot(x='Type', hue='Machine failure', data=df)
plt.title('Univariate Analysis: Type vs. Machine Failure')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()

# Tool wear distribution for machine failures
plt.figure(figsize=(8, 6))
sns.histplot(x='Tool wear [min]', hue='Machine failure', data=df[df['Machine failure'] == 1])
plt.title('Tool Wear Distribution for Machine Failures')
plt.xlabel('Tool Wear [min]')
plt.ylabel('Frequency')
plt.show()

# Machine failure count plot
sns.countplot(x="Machine failure", hue="Machine failure", data=df)
plt.show()

# Prepare features and target
X = df.drop(["Machine failure"], axis=1)
y = df["Machine failure"]

# Oversampling to balance the dataset
oversample = RandomOverSampler(sampling_strategy='minority', random_state=1)
X1, y1 = oversample.fit_resample(X, y)

# Verify oversampling
print("\nOversampled Target Value Counts:")
print(pd.DataFrame(y1.value_counts()))

# Split data into train and test sets
X1.columns = [col.replace('[', '').replace(']', '').replace('<', '').replace('>', '').strip() for col in X1.columns]
x_train, x_test, y_train, y_test = train_test_split(X1, y1, test_size=0.4, random_state=42)

# Model class for training and evaluation
class Model:
    scores = {'Model': [], 'Accuracy': [], 'CV_Score': [], 'auc': [], 'MSE': [], 'R2_Score': []}

    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name

    def predict(self):
        self.model.fit(x_train, y_train)
        pred = self.model.predict(x_test)
        cv_score = np.mean(cross_val_score(self.model, x_test, y_test, cv=5))
        auc = roc_auc_score(y_test, pred)
        fpr, tpr, thresholds = roc_curve(y_test, pred)

        self.performance(pred, cv_score, auc)
        self.plot_roc_curve(fpr, tpr)

    def performance(self, pred, cv_score, auc):
        accuracy = accuracy_score(pred, y_test)

        Model.scores['Model'].append(self.model_name)
        Model.scores['Accuracy'].append(accuracy)
        Model.scores['CV_Score'].append(cv_score)
        Model.scores['auc'].append(auc)
        Model.scores['MSE'].append(mean_squared_error(y_test, pred))
        Model.scores['R2_Score'].append(r2_score(y_test, pred))

        print(f'\n{self.model_name} Results:')
        print(f'Accuracy Score: {accuracy}')
        print(f'Mean Cross Validation Score: {cv_score}\n')
        print(f'Classification Report\n{classification_report(pred, y_test)}')
        print(f'R2 Score: {r2_score(y_test, pred)}')
        print(f'Mean Squared Error: {mean_squared_error(y_test, pred)}')

        self.confusion_matrix(pred)

    def confusion_matrix(self, pred):
        cm = confusion_matrix(y_test, pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu")
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.show()

    def plot_roc_curve(self, fpr, tpr):
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve - {self.model_name}')
        plt.legend()
        plt.show()

# Train and evaluate models
models = [
    (LogisticRegression(random_state=20), 'Logistic Regression'),
    (KNeighborsClassifier(), 'KNN'),
    (DecisionTreeClassifier(random_state=20), 'Decision Tree'),
    (GaussianNB(), 'Gaussian NB'),
    (SVC(random_state=20), 'SVC'),
    (GradientBoostingClassifier(random_state=20), 'GradientBoostingClassifier'),
    (RandomForestClassifier(random_state=20), 'Random Forest'),
    (CatBoostClassifier(random_state=20, verbose=0), 'CatBoostClassifier'),
    (XGBClassifier(random_state=20, use_label_encoder=False, eval_metric='logloss'), 'XGBoostClassifier')
]
#moar models

for model, name in models:
    model_instance = Model(model, name)
    model_instance.predict()
    

# Display performance summary
performance = pd.DataFrame(Model.scores)
performance.sort_values(by='Accuracy', ascending=False, inplace=True)
performance.reset_index(drop=True, inplace=True)
print("\nPerformance Summary:")
print(performance)