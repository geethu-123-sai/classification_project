# Data Cleaning & Preprocessing Notebook
# =================================
# Competition Project: Student Performance Classification

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Dataset
# NOTE: Update path if required
raw_df = pd.read_csv("data/raw_data.csv")

# 3. Basic Inspection
print("Shape of dataset:", raw_df.shape)
raw_df.head()

# 4. Dataset Info & Null Check
raw_df.info()
print("\nMissing Values:\n", raw_df.isnull().sum())

# 5. Drop Irrelevant Columns
# student_id is an identifier and does not add predictive value
df = raw_df.drop(columns=['student_id'])

# 6. Validate Numerical Ranges
# Ensure values are within logical bounds
assert df['age'].between(15, 30).all(), "Age out of expected range"
assert df['class_attendance'].between(0, 100).all(), "Attendance out of range"
assert df['exam_score'].between(0, 100).all(), "Exam score out of range"

# 7. Outlier Detection (IQR Method)
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

numerical_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours', 'exam_score']

for col in numerical_cols:
    df = remove_outliers_iqr(df, col)

print("Shape after outlier removal:", df.shape)

# 8. Target Engineering: Convert exam_score to Performance Class
# Low: 0-40, Medium: 41-70, High: 71-100

def performance_label(score):
    if score <= 40:
        return 'Low'
    elif score <= 70:
        return 'Medium'
    else:
        return 'High'

# Create new target column
df['performance'] = df['exam_score'].apply(performance_label)

# Drop original numeric target
df.drop(columns=['exam_score'], inplace=True)

# 9. Target Distribution Check
print(df['performance'].value_counts(normalize=True))

# 10. Data Type Verification
print("\nFinal Data Types:\n", df.dtypes)

# 11. Save Cleaned Dataset
cleaned_path = "data/cleaned_data.csv"
df.to_csv(cleaned_path, index=False)

print(f"Cleaned data saved to {cleaned_path}")

# 12. Final Preview
df.head()
