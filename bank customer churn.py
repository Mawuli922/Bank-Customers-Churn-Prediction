import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import joblib
from sklearn.feature_selection import SelectKBest, mutual_info_classif
# Read the csv file into a dataframe
churn_df = pd.read_csv("botswana_bank_customer_churn.csv")

# Inspect the  first 5 rows of dataframe
print(churn_df.head())

# Inspect basic information of every column in the dataframe
print(churn_df.info())

# The only columns with missing data are the Churn Reason and Churn Date Columns

print(churn_df["Churn Flag"].value_counts(dropna=False))

print(churn_df["Churn Reason"].value_counts(dropna=False))

# Since the churn flag 0 value count is the same as the value count of NaN in Churn Reason

# It is assumed that the number of rows without churn reason represent all customers without churn
# Replace empty values in churn reason with string "Null"
churn_df["Churn Reason"] = churn_df["Churn Reason"].fillna("Null")

print(pd.crosstab(churn_df["Churn Flag"], churn_df["Churn Reason"]))


for column in churn_df.columns:
    print(churn_df[column].value_counts(dropna= False))

churn_df["Income"].hist(bins=40)
plt.xlabel("Annual Income")
plt.ylabel("Count")
plt.title("Histogram of Bank Customers Income")
plt.show()


churn_df["Balance"].hist(bins=40)
plt.xlabel("Bank Account Balance")
plt.ylabel("Count")
plt.title("Histogram of Bank Customers' Balance")
plt.show()

churn_df.groupby(by="Churn Flag")["Balance"].mean().plot(kind="bar")
plt.xlabel("Churn Status")
plt.ylabel("Average Bank Balance")
plt.title("Average Bank Balance by Churn Status")
plt.show()

sns.boxplot(churn_df, x="Marital Status", y="Income", hue="Gender")
plt.xlabel("Marital Status")
plt.ylabel("Income")
plt.title("Distribution of Income by Marital Status and Gender")
plt.show()
churn_df["Gender_Numeric"] = churn_df["Gender"].apply(lambda x: 0 if x== "Male" else 1)

X = churn_df[["Income", "Balance", "Credit Score", "Credit History Length", "Outstanding Loans",
              "NumOfProducts", "NumComplaints", "Customer Tenure", "Gender_Numeric"]]

y = churn_df["Churn Flag"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

def select_features(X_train, y_train, X_test, k):
    selector = SelectKBest(score_func=mutual_info_classif,k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected, selector

classifiers = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}


k_values = list(range(1, X_train.shape[1]+1))

for clf_name, clf in classifiers.items():
    best_k = 0
    best_score = 0

    for k in k_values:
        X_train_selected, X_test_selected, selector = select_features(X_train, y_train, X_test, k)
        clf.fit(X_train_selected, y_train)
        y_pred = clf.predict(X_test_selected)
        score = accuracy_score(y_test, y_pred)

        if score > best_score:
            best_score= score
            best_k = k

    print(f"Best k for {clf_name}: {best_k} with accuracy: {best_score:.4f}")

    X_train_selected, X_test_selected, selector = select_features(X_train, y_train, X_test, best_k)
    clf.fit(X_train_selected, y_train)
    y_pred = clf.predict(X_test_selected)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix for {clf_name} with {best_k} features")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()




