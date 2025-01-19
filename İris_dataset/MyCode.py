import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Veri setini yükle
df = pd.read_csv('IRIS.csv')

# Başlangıç incelemesi
df.head(10)
df.shape
df.describe()
df.info()

# Korelasyon grafiği
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.title("Correlation Graph")
plt.show()

# Sütun adlarını düzenleme
df.rename(columns={
    'sepal length (cm)': 'sepal_length',
    'sepal width (cm)': 'sepal_width',
    'petal length (cm)': 'petal_length',
    'petal width (cm)': 'petal_width'
}, inplace=True)

# Scatter plot
sns.scatterplot(
    x=df['sepal_length'],
    y=df['sepal_width'],
    hue=df['species'],
    palette="Set1"
)
plt.title("Scatter Plot by Species")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend(title="Species")
plt.show()

# 'species' sütununu sayısal verilere dönüştürme
df['species'] = df['species'].astype('category').cat.codes

# Özellikler ve etiketleri ayırma
X = df.drop("species", axis=1)
y = df["species"]

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelleri tanımlama
models = {
    "Lojistik Regresyon": LogisticRegression(max_iter=200),
    "Karar Ağacı": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Modellerin doğruluk oranlarını hesaplama
results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append({
        "Model": model_name,
        "Doğruluk Oranı (%)": accuracy * 100
    })

# Sonuçları DataFrame'e dönüştürme
results_df = pd.DataFrame(results)
print(results_df)

# Cross-validation ile doğruluk oranlarını hesaplama
cv_results = []

for model_name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    mean_accuracy = np.mean(cv_scores) * 100
    cv_results.append({
        "Model": model_name,
        "Doğruluk Oranı (%)": mean_accuracy
    })

# Cross-validation sonuçlarını DataFrame'e dönüştürme
cv_results_df = pd.DataFrame(cv_results)
print(cv_results_df)

# Doğruluk oranları grafiği
plt.figure(figsize=(10, 6))
bars = plt.bar(cv_results_df['Model'], cv_results_df['Doğruluk Oranı (%)'], color=['pink', 'green', 'red', 'purple', 'yellow', 'orange'])
plt.xlabel("Modeller")
plt.ylabel("Doğruluk Oranı (%)")
plt.title("Modellerin Doğruluk Oranları")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.2f}", ha='center', va='bottom')

plt.xticks(rotation=45)
plt.show()
