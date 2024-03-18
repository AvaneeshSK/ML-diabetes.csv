# Diabetes
# https://www.kaggle.com/datasets/saurabh00007/diabetescsv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

from sklearn.svm import LinearSVC

df = pd.read_csv('Machine Learning 2\\PredictDiabetes\\diabetes.csv')
print(df)

# lets compare MinMaxScaler vs Raw
df_minmax = df.copy()
min_max_cols = df.columns[:-1]

ct = ColumnTransformer([
    ('minmax', MinMaxScaler(), min_max_cols)
], remainder='passthrough')

transformed_data = ct.fit_transform(df_minmax)
df_minmax = pd.DataFrame(data=transformed_data)
df_minmax.rename(columns={8:'Outcome'}, inplace=True)

X_minmax = df_minmax.drop(columns=['Outcome'])
y_minmax = df_minmax['Outcome']

X = df.drop(columns=['Outcome'])
y = df['Outcome']

X_train_minmax, X_test_minmax, y_train_minmax, y_test_minmax = train_test_split(X_minmax, y_minmax, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# modelling
minmax_score = np.mean(cross_val_score(
    estimator=LinearSVC(),
    X=X_minmax,
    y=y_minmax,
    cv=3,
    verbose=2,
    n_jobs=-1,
    scoring='f1'
))
raw_score = np.mean(cross_val_score(
    estimator=LinearSVC(),
    X=X,
    y=y,
    cv=3,
    verbose=2,
    n_jobs=-1,
    scoring='f1'
))

fig, ax = plt.subplots()
bars = ax.bar(['MinMaxScaled Data', 'Raw Data'], [minmax_score, raw_score], color=['salmon', 'lightblue'])
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set(xlabel='Data MinMax vs Raw', ylabel='R2 Score')
for p in bars.patches:
    ax.annotate(p.get_height(), (p.get_x() + p.get_width() / 2, p.get_height()), xytext=(10, 10), textcoords='offset points', ha='center', va='center')

model_minmax = LinearSVC()
model_raw = LinearSVC()

model_minmax.fit(X_train_minmax, y_train_minmax)
model_raw.fit(X_train, y_train)

print(f'r2 Non Cross Fold Score (MinMaxed Data): {model_minmax.score(X_test_minmax, y_test_minmax)}')
print(f'r2 Non Cross Fold Score (Raw Data): {model_raw.score(X_test, y_test)}')


minmax_predictions = model_minmax.predict(X_test_minmax).flatten()
raw_predictions = model_raw.predict(X_test).flatten()

# plot confusion matrix
cf_mat_minmax = confusion_matrix(y_pred=minmax_predictions, y_true=y_test_minmax)
cf_mat_raw = confusion_matrix(y_pred=raw_predictions, y_true=y_test)
fig, (ax_1, ax_2) = plt.subplots(ncols=2)
sns.heatmap(data=cf_mat_minmax, annot=True, ax=ax_1)
sns.heatmap(data=cf_mat_raw, annot=True, ax=ax_2)

# plot roc-auc curve
minmax_roc_auc = roc_auc_score(y_true=y_test_minmax, y_score=minmax_predictions)
raw_roc_auc = roc_auc_score(y_true=y_test, y_score=raw_predictions)
fpr_minmax, tpr_minmax, threshold_minmax = roc_curve(y_true=y_test_minmax, y_score=minmax_predictions)
fpr_raw, tpr_raw, threshold_raw = roc_curve(y_true=y_test, y_score=raw_predictions)

disp1 = RocCurveDisplay(fpr=fpr_minmax, tpr=tpr_minmax, roc_auc=minmax_roc_auc, estimator_name='LinearSVC_minmaxed_data')
disp2 = RocCurveDisplay(fpr=fpr_raw, tpr=tpr_raw, roc_auc=raw_roc_auc, estimator_name='LinearSVC_raw_data')

fig, (ax_roc1, ax_roc2) = plt.subplots(ncols=2)
disp1.plot(ax=ax_roc1)
disp2.plot(ax=ax_roc2)

plt.show()
