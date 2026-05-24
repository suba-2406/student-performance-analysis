import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("StudentPerformanceFactors.csv")

# Show first 5 rows
print(df.head())

# Check missing values
print(df.isnull().sum())

# Visualization
sns.scatterplot(x="Hours_Studied", y="Exam_Score", data=df)

plt.title("Hours Studied vs Exam Score")

print(df.info())
print(df.describe())

df["Teacher_Quality"] = df["Teacher_Quality"].fillna(df["Teacher_Quality"].mode()[0])

df["Parental_Education_Level"] = df["Parental_Education_Level"].fillna(df["Parental_Education_Level"].mode()[0])

df["Distance_from_Home"] = df["Distance_from_Home"].fillna(df["Distance_from_Home"].mode()[0])

print(df.isnull().sum())

numeric_df = df.select_dtypes(include='number')

corr = numeric_df.corr()

plt.figure(figsize=(12,8))

sns.heatmap(corr, annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")


plt.figure(figsize=(8,5))

plt.hist(df["Attendance"], bins=10)

plt.title("Attendance Distribution")

plt.xlabel("Attendance")

plt.ylabel("Count")

plt.figure(figsize=(8,5))

sns.countplot(x="Motivation_Level", data=df)

plt.title("Motivation Levels")

plt.figure(figsize=(8,5))

sns.boxplot(x="Internet_Access", y="Exam_Score", data=df)

plt.title("Internet Access vs Exam Score")

df.to_csv("Cleaned_StudentPerformance.csv", index=False)

plt.show()