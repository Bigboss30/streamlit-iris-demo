from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pandas as pd

# Load the dataset
data = pd.read_csv('heart.csv')

# Define features and target variable
X = data.drop('HeartDisease', axis=1)  # Features (all columns except 'HeartDisease')
y = data['HeartDisease']  # Target variable

# Create a preprocessing function
def create_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']),
            
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
        ])
    
    return preprocessor

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the preprocessor
preprocessor = create_preprocessor()


#1
# Logistic Regression Model
model_lr = Pipeline(steps=[('preprocessor', preprocessor), 
                           ('classifier', LogisticRegression())])

# Train and evaluate
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
print("Logistic Regression:\n", classification_report(y_test, y_pred_lr))

#2
# Decision Tree Model
model_dt = Pipeline(steps=[('preprocessor', preprocessor), 
                           ('classifier', DecisionTreeClassifier())])

# Train and evaluate
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
print("Decision Tree:\n", classification_report(y_test, y_pred_dt))

#3
# Random Forest Model
model_rf = Pipeline(steps=[('preprocessor', preprocessor), 
                           ('classifier', RandomForestClassifier())])

# Train and evaluate
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
print("Random Forest:\n", classification_report(y_test, y_pred_rf))

#4
# SVM Model
model_svm = Pipeline(steps=[('preprocessor', preprocessor), 
                            ('classifier', SVC())])

# Train and evaluate
model_svm.fit(X_train, y_train)
y_pred_svm = model_svm.predict(X_test)
print("SVM:\n", classification_report(y_test, y_pred_svm))

#5
# KNN Model
model_knn = Pipeline(steps=[('preprocessor', preprocessor), 
                            ('classifier', KNeighborsClassifier())])

# Train and evaluate
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)
print("KNN:\n", classification_report(y_test, y_pred_knn))