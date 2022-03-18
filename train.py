import pandas as pd
import numpy as np
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib

heart_df = pd.read_csv('data/heart.csv')

x = heart_df.iloc[:, :-1]
y = heart_df.iloc[:, -1]


def data_enhancement(data):
    gen_data = data.copy()

    for cp in data['cp'].unique():
        cp_data = gen_data[gen_data['cp'] == cp]
        trtbps_std = cp_data['trtbps'].std()
        age_std = cp_data['age'].std()
        chol_std = cp_data['chol'].std()
        thalachh_std = cp_data['thalachh'].std()

        for i in gen_data[gen_data['cp'] == cp].index:
            if np.random.randint(2) == 1:
                gen_data['trtbps'].values[i] += trtbps_std / 10
            else:
                gen_data['trtbps'].values[i] -= trtbps_std / 10
            if np.random.randint(2) == 1:
                gen_data['age'].values[i] += age_std / 10
            else:
                gen_data['age'].values[i] -= age_std / 10
            if np.random.randint(2) == 1:
                gen_data['chol'].values[i] += chol_std / 10
            else:
                gen_data['chol'].values[i] -= chol_std / 10
            if np.random.randint(2) == 1:
                gen_data['thalachh'].values[i] += thalachh_std / 10
            else:
                gen_data['thalachh'].values[i] -= thalachh_std / 10
    return gen_data


gen = data_enhancement(heart_df)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)

extra_sample = gen.sample(gen.shape[0] // 3)
x_train = pd.concat([x_train, extra_sample.drop(['output'], axis=1)])
y_train = pd.concat([y_train, extra_sample['output']])

x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)

results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})

classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Skl GBM": GradientBoostingClassifier(n_estimators=100),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(kernel='rbf')
}

for model_name, model in classifiers.items():
    start_time = time.time()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    total_time = time.time() - start_time

    results = results.append({"Model": model_name,
                              "Accuracy": metrics.accuracy_score(y_test, pred) * 100,
                              "Bal Acc.": metrics.balanced_accuracy_score(y_test, pred) * 100,
                              "Time": total_time},
                             ignore_index=True)

results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
results_ord.index += 1
results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')
print(results_ord)

model = classifiers[f'{results_ord.iloc[0, 0]}']
joblib.dump(model, 'model.pkl')
