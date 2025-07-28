import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------------------------
# 1. Load dataset
# -------------------------------------------------
df = pd.read_csv(r"C:\Users\saraye tel\OneDrive\Desktop\ARCH_Roadmap\Phase_1\heart.csv")

# -------------------------------------------------
# 2. One‑hot‑encode categorical features
# -------------------------------------------------
df = pd.get_dummies(
    df,
    columns=["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
    drop_first=True
)

# -------------------------------------------------
# 3. Separate features and label
# -------------------------------------------------
X = df.drop("target", axis=1).values
y = df["target"].values

# -------------------------------------------------
# 4. Train‑test split (stratified)
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# -------------------------------------------------
# 5. Scale numeric features (recommended for KNN / SVM)
# -------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# -------------------------------------------------
# 6. Define models to compare
# -------------------------------------------------
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="linear", probability=True),
    "Naive Bayes": GaussianNB()
}

accuracy_scores = {}

# -------------------------------------------------
# 7. Train, predict, evaluate
# -------------------------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = acc

    cm = confusion_matrix(y_test, y_pred)

    print(f"\nModel: {name}")
    print(f"Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    # Confusion‑matrix heatmap
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title(f"Confusion Matrix – {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# 8. Accuracy comparison bar chart
# -------------------------------------------------
plt.figure(figsize=(6,4))
sns.barplot(x=list(accuracy_scores.keys()),
            y=list(accuracy_scores.values()),
            palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
for i, v in enumerate(accuracy_scores.values()):
    plt.text(i, v + 0.03, f"{v:.2f}", ha="center")
plt.tight_layout()

# ✅ Save the plot BEFORE plt.show()
plt.savefig("model_accuracy_comparison.png", dpi=300)
plt.show()
