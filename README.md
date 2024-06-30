# Regression-Analysis-and-Tax-Fraud-Risk-Prediction
we perform regression analysis on a dataset provided by Zameen.com to predict property prices based on various attributes. This analysis will support decision-making in property transactions by providing accurate price predictions.

#Regression Analysis

Introduction

In this project, we perform regression analysis on a dataset provided by Zameen.com to predict property prices based on various attributes. This analysis will support decision-making in property transactions by providing accurate price predictions.


Data Pre-Processing

Steps:

Validation and Correction: Ensured consistent data formats.

Checked for and unified inconsistent data formats.
Verified correct data types for each column.
Handling Missing Values: Addressed missing values using appropriate techniques.

Used mean imputation for numerical features.
Used mode imputation for categorical features.
Outlier Management: Identified and handled potential outliers.

Used z-score analysis and IQR method to detect outliers.
Applied capping for outliers beyond a certain threshold.
Code Sample:

python

Copy code
# Handling missing values

df.fillna(df.mean(), inplace=True)

# Outlier treatment

from scipy import stats

df = df[(np.abs(stats.zscore(df.select_dtypes(include=[np.number]))) < 3).all(axis=1)]

Exploratory Data Analysis (EDA)

Steps:


Correlation Structure: Analyzed correlation between variables.

Used a correlation matrix and heatmap to visualize correlations.

Correlation Analysis:

No significant correlation between the number of properties listed by an agent and average property price.

Code Sample:



python

Copy code

import seaborn as sns

import matplotlib.pyplot as plt



# Correlation matrix

corr = df.corr()

sns.heatmap(corr, annot=True, cmap='coolwarm')

plt.title('Correlation Matrix')

plt.show()

Feature Engineering

Steps:



Price per Square Meter: Computed a new column for price per square meter.



df['price_per_sqm'] = df['price'] / df['area']

Temporal Features: Derived features from date_added.



Extracted month, quarter, and day of the week.

Standardization: Standardized numerical variables using StandardScaler.



from sklearn.preprocessing import StandardScaler

Encoding: Encoded categorical variables using OneHotEncoder.



pd.get_dummies(df, columns=['category'], drop_first=True)

Code Sample:



python

Copy code

# Creating price per square meter

df['price_per_sqm'] = df['price'] / df['area']



# Temporal features

df['month'] = pd.to_datetime(df['date_added']).dt.month

df['day_of_week'] = pd.to_datetime(df['date_added']).dt.dayofweek



# Standardization

scaler = StandardScaler()

df[['price', 'area']] = scaler.fit_transform(df[['price', 'area']])



# Encoding

df = pd.get_dummies(df, columns=['category'], drop_first=True)

Model Training

Steps:



Data Splitting: Split data into training and testing sets (80:20 ratio).




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Model Selection: Used Linear Regression as the base model.



from sklearn.linear_model import LinearRegression

Hyperparameter Tuning: Performed using GridSearchCV.


from sklearn.model_selection import GridSearchCV

Code Sample:


python


Copy code

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model training


model = LinearRegression()
model.fit(X_train, y_train)



# Hyperparameter tuning


from sklearn.model_selection import GridSearchCV





parameters = {'fit_intercept': [True, False]}

grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=5)

grid_search.fit(X_train, y_train)
Model Evaluation

Metrics:



Mean Absolute Error (MAE)



from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
Mean Squared Error (MSE)




from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
Root Mean Squared Error (RMSE)



rmse = np.sqrt(mse)


Mean Absolute Percentage Error (MAPE)



mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

Code Sample:



python

Copy code

# Predictions
y_pred = model.predict(X_test)



# Evaluation metrics

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100



print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, MAPE: {mape}%")

Bonus: Dimensionality Curse

Steps:


Cross-Validation: Used K-fold cross-validation to assess model performance.



from sklearn.model_selection import cross_val_score

Regularization: Applied Ridge and Lasso regression.



from sklearn.linear_model import Ridge, Lasso
Code Sample:



python

Copy code

# Cross-validation

scores = cross_val_score(model, X, y, cv=5)



# Regularization
ridge = Ridge(alpha=1.0)


lasso = Lasso(alpha=0.1)
ridge.fit(X_train, y_train)

lasso.fit(X_train, y_train)

Predicting Tax Fraud Risk Using Decision Trees

Objective

To build a decision tree model to predict tax fraud risk, classifying individuals into "Risky" or "Good" based on various attributes.



Data Pre-Processing

Steps:



Validation and Correction: Ensured consistent data formats.



Verified and corrected any inconsistent formats and data types.

Handling Missing Values: Managed missing values effectively.



Used mean or mode imputation depending on the feature type.

Code Sample:



python

Copy code

# Handling missing values

df.fillna(df.mean(), inplace=True)


Feature Engineering

Steps:



Categorical Variables: Converted categorical variables into dummy variables.



df = pd.get_dummies(df, columns=['Undergrad', 'Marital_Status', 'Urban'], drop_first=True)

Target Variable Transformation: Transformed taxable income into binary classification.



df['Risk'] = np.where(df['Taxable_Income'] <= 30000, 0, 1)

Feature Scaling: Scaled features using MinMaxScaler.



from sklearn.preprocessing import MinMaxScaler

Code Sample:



python

Copy code

# Encoding categorical variables

df = pd.get_dummies(df, columns=['Undergrad', 'Marital_Status', 'Urban'], drop_first=True)



# Target variable transformation

df['Risk'] = np.where(df['Taxable_Income'] <= 30000, 0, 1)



# Feature scaling


scaler = MinMaxScaler()

df[['Work_Experience']] = scaler.fit_transform(df[['Work_Experience']])

Model Development

Steps:



Data Splitting: Split data into training and testing sets (70:30 ratio).



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

Decision Tree Model: Used Decision Tree for classification.



from sklearn.tree import DecisionTreeClassifier

Code Sample:



python

Copy code

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# Decision tree model
clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

Model Evaluation

Metrics:



Accuracy



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

Precision, Recall, and F1 Score



from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred)

Code Sample:



python

Copy code

# Predictions

y_pred = clf.predict(X_test)



# Evaluation metrics

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(y_test, y_pred)



print(f"Accuracy: {accuracy}")

print(report)


Model Improvement

Steps:




Hyperparameter Tuning: Used GridSearchCV for tuning.



param_grid = {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}

Feature Selection: Performed feature importance analysis.



Used feature importance scores to select key features.

Code Sample:



python

Copy code

# Hyperparameter tuning

param_grid = {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)

grid_search.fit(X_train, y_train)



# Feature selection

feature_importance = clf.feature_importances_



