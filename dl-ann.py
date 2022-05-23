#%%
import os
import shutil

path_source, path_dest, file_name = "/home/lyoh001/Downloads", "/home/lyoh001/vscode/AzureDL/data", "data.csv"
[
    shutil.move(
        os.path.join(path_source, f),
        os.path.join(path_dest, file_name),
    )
    for f in os.listdir(path_source)
    if ".csv" in f
]
os.chdir(path_dest)
os.system("git add .")
os.system("git commit -m 'Commit'")
os.system("git push")
#%%
import os

import pandas as pd

path_source, file_name = "/home/lyoh001/Downloads", "data.csv"
df = pd.read_excel(os.path.join(path_source, (source_file_name := next(iter(f for f in os.listdir(path_source))))))
df.to_csv(os.path.join(path_source, file_name), index=False)
os.remove(os.path.join(path_source, source_file_name))
#%%
import chardet

with open(os.path.join(path_dest, file_name), "rb") as data:
    print(chardet.detect(data.read()))
#%%
import chardet
import requests

response = requests.get(
    "https://raw.githubusercontent.com/lyoh001/AzureML/main/data.csv"
)
chardet.detect(response.content)["encoding"]
#%%
pip_list = !pip list
packages = ["dabl", "imblearn", "keras-tuner", "mysql-connector-python", "numpy", "pandas", "sklearn", "tensorflow"]
for package in packages:
    if not pip_list.grep(package):
        !pip3 install {package}

print("Package installations completed.")
#%%
import calendar
import datetime
import os
import shutil
import warnings
from pickle import dump

import chardet
import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import pandas as pd
import requests
import scipy.stats as stat
import seaborn as sns
import statsmodels.api as sm
import tensorflow as tf
from dabl import SimpleClassifier, SimpleRegressor, clean, plot
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from IPython.display import display
from keras_tuner.tuners import BayesianOptimization, Hyperband, RandomSearch
from mysql.connector import errorcode
from sklearn import set_config
from sklearn.compose import make_column_transformer
from sklearn.datasets import (load_breast_cancer, load_diabetes, load_iris,
                              load_wine)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (SelectPercentile, VarianceThreshold,
                                       chi2, f_classif)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                                   StandardScaler)
from sklearn.utils import class_weight
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.python.client import device_lib

%matplotlib inline
set_config(display="diagram", print_changed_only=False)
pd.set_option("display.float_format", lambda f: "%.2f" % f)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
plt.rcParams["figure.figsize"] = [18, 7]
plt.style.use("dark_background")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore")

time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

print(device_lib.list_local_devices())
print(tf.config.list_physical_devices("GPU"))
print(tf.test.gpu_device_name())
print(os.system("cat /proc/cpuinfo | grep 'model name'"))
print(os.system("cat /proc/meminfo | grep 'MemTotal'"))
print(os.system("nvidia-smi"))
#%%
# try:
#     conn = mysql.connector.connect(
#         user="operations",
#         password=os.environ["OPENAI_API_KEY"],
#         host="vickk73mysqlserver.mysql.database.azure.com",
#         port=3306,
#         database="",
#         ssl_ca="{ca-cert filename}",
#         ssl_disabled=False,
#     )
#     print("Connection established")
# except mysql.connector.Error as e:
#     if e.errno == errorcode.ER_ACCESS_DENIED_ERROR:
#         print("Something is wrong with the user name or password")
#     elif e.errno == errorcode.ER_BAD_DB_ERROR:
#         print("Database does not exist")
#     else:
#         print(e)
# else:
#     tables = {table: pd.read_sql(f"""SELECT * FROM {table};""", conn) for table in pd.read_sql("SHOW TABLES", conn).iloc[:, 0]}
#     for table_name, table_dataframe in tables.items():
#         print(f"-------------------------------------------------------\nTable: {table_name}")
#         display(table_dataframe)
#     conn.close()
#     print(f"Discovered table(s): {len(tables)}.\nDB Connection closed.")
#%%
# for table_name, table_dataframe in tables.items():
#     print(f"-------------------------------------------------------\nTable: {table_name}\nColumns: {table_dataframe.columns}\nShape: {table_dataframe.shape}")
# df.rename({"": ""}, axis=1, inplace=True)
# df = pd.merge(tables[""], tables[""], on="")
# df = pd.merge(df, tables[""], on="")
# display(df)
#%%
df = load_breast_cancer(as_frame=True)["frame"]
df = load_iris(as_frame=True)["frame"]
df = load_wine(as_frame=True)["frame"]
df = load_diabetes(as_frame=True)["frame"]
url = "https://raw.githubusercontent.com/lyoh001/AzureML/main/data.csv"
df = pd.read_csv(url, delimiter=",", encoding=chardet.detect(requests.get(url).content)["encoding"], thousands=",")
display(df)
#%%
RANDOM_STATE = 11
SEARCH = ["hyperband", "random", "bayesian"][0]
EPOCHS = 500
MAX_TRIALS = 20
DUPLICATES = 0
SCALER = 1
CLASSIFICATION = 0

print("-------------------------------------------------------")
print(f"Current Shape: {df.shape}.")
print("-------------------------------------------------------")
print(f"Duplicates Percentage: {df.duplicated().sum() / df.shape[0] * 100:.2f}%")
if DUPLICATES:
    print(f"Duplicates have been kept {df.shape}.")
else:
    df.drop_duplicates(inplace=True)
    print(f"Duplicates have been removed {df.shape}.")
display(df.sample(3))
y_label = "target"
#%%
# df[y_label] = df[y_label].map({k: i for i, k in enumerate(df[y_label].unique(), 0)})
df[y_label].value_counts()
#%%
sns.heatmap(df.corr(), cmap="Blues", fmt=".2f", annot=True, linewidths=1)
plt.title("Correlation Matrix")
plt.xticks(rotation=45)
plt.show()
#%%
corr_matrix = df.corr()[y_label].sort_values().drop(y_label)
sns.barplot(x=corr_matrix.values, y=corr_matrix.index).set_title("Correlation Matrix")
plt.show()
#%%
fig, ax = plt.subplots(nrows=1, ncols=2)
corr_matrix = df.corr()[y_label].abs().sort_values(ascending=False).drop(y_label)
sns.barplot(x=corr_matrix.values, y=corr_matrix.index, ax=ax[0]).set_title("Horizontal Correlation Matrix")
sns.barplot(x=corr_matrix.index, y=corr_matrix.values, ax=ax[1]).set_title("Vertical Correlation Matrix")
plt.xticks(rotation=45)
plt.show()
#%%
def correlation(X, threshold):
    col_corr = set()
    df_corr = X.corr().abs()
    for i, _ in enumerate(df_corr.columns):
        for j in range(i):
            if (df_corr.iloc[i, j] >= threshold) and (
                df_corr.columns[j] not in col_corr
            ):
                col_corr.add(df_corr.columns[i])
    return col_corr

col_drop = correlation(df.drop(y_label, axis=1), 0.85)
df.drop(col_drop, inplace=True, axis=1)
print("-------------------------------------------------------")
print(f"Current Shape: {df.shape}.")
print("-------------------------------------------------------")
print(f"Highly correlated cols have been removed: {len(col_drop)}.")
print(f"Highly correlated cols: {col_drop}.")
#%%
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="Blues")
plt.title("Missing Values")
plt.xticks(rotation=45)
plt.show()
#%%
sns.displot(
    data=df.isnull().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    height=9.25,
)
plt.axvline(0.3, color="white")
plt.title("Missing Values")
plt.show()
#%%
"""
%a: Weekday, abbreviated: Mon, Tues, Sat
%A: Weekday, full name: Monday, Tuesday, Saturday
%w: Weekday, decimal. 0=Sunday: 1, 2, 6
%d: Day of month, zero-padded: 01, 02, 21
%b: Month, abbreviated: Jan, Feb, Sep
%B: Month, full name: January, February, September
%m: Month number, zero-padded: 01, 02, 09
%y: Year, without century, zero-padded: 02, 95, 99
%Y: Year, with century: 1990, 2020
%H: Hour (24 hour), zero padded: 01, 22
%I: Hour (12 hour) zero padded: 01, 12
%p: AM or PM: AM, PM
%M: Minute, zero-padded: 01, 02, 43
%S: Second, zero padded: 01, 32, 59
%f: Microsecond, zero-padded: 000001, 000342, 999999
%z: UTC offset ±HHMM[SS[.ffffff]]: +0000, -1030, -3423.234
%Z: Time zone name: ITC, EST, CST
%j: Day of year, zero-padded: 001, 365, 023
%U: Week # of year, zero-padded. Sunday first day of week: 00, 01, 51
%W: Week # of year, zero-padded. Monday first day of week: 00, 02, 51
%c: Appropriate date and time: Monday Feb 01 21:30:00 1990
%x: Appropriate Date: 02/01/90
%X: Appropriate Time: 21:22:00
"""
# df["date"] = pd.to_datetime(df[""], format="%Y-%m-%d %H:%M:%S")
# df["year"] = df["date"].dt.year
# df["month"] = df["date"].dt.month
# df["dayofweek"] = df["date"].dt.dayofweek
# df["day"] = df["date"].dt.days
# df["day"] = (df[""] - df[""]).dt.days
# df["hours"] = (df[""] - df[""]).dt.total_seconds() / 60
# df["end date"] = df["date"].map(lambda d: datetime.datetime.strptime(f"{d.year}-{d.month}-{calendar.monthrange(d.year, d.month)[-1]}", "%Y-%m-%d"))
# def isfloat(n):
#     try:
#         float(n)
#         return True
#     except:
#         return False
# df[df[""].map(lambda n: not isfloat(n))]
# df = pd.concat([df1, df2], axis=0, ignore_index=True)
# df = pd.merge(df1, df2, on="", how="outer")
# df.rename({"": ""}, axis=1, inplace=True)
# df.replace({"": 0, "": 1, "unknown": np.nan}, inplace=True)
# df[""] = df[""].map(lambda x: {"": 0, "": 1}.get(x, np.nan))
# df[""] = df[""].map(pd.to_numeric, errors="coerce")
# df[""] = df[""].astype(float)
# df[""] = df[""].astype(str).str.lower()
# df[""] = df[""].astype(str).str.replace("", "")
# df[""] = df[""].astype(str).str.split().str.get(0)
# df[""] = df[""].astype(str).str.strip()
# df[["", ""]] = df[""].astype(str).str.split(" ", expand=True)
# df.drop(df[df[""] < 0].index, inplace=True)
# df.drop([""], inplace=True, axis=1)
# df.dropna(subset=[y_label], inplace=True)
# df[].value_counts()
#%%
col_cat_oe = []
preprocessor_cat_oe = make_pipeline(
    (SimpleImputer(strategy="most_frequent")),
    (OrdinalEncoder(categories=[["", ""]])),
)
col_cat = [col for col in df.columns if np.issubsctype(df[col].dtype, np.object0) and col != y_label]
col_num = [col for col in df.columns if np.issubsctype(df[col].dtype, np.number) and col != y_label]
col_cat_ohe = [col for col in col_cat if col not in col_cat_oe]
col_num_disc = [col for col in col_num if df[col].nunique() < 25]
col_num_cont = [col for col in col_num if col not in col_num_disc]

df_info = pd.DataFrame(
    {
        "column": [col for col in df.columns],
        "dtype": [f"{df[col].dtype}" for col in df.columns],
        "na": [f"{df[col].isna().sum()}" for col in df.columns],
        "na %": [f"{round(df[col].isna().sum() / df[col].shape[0] * 100)}%" for col in df.columns],
        "outliers": [f"{((df[col] < (df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))) | (df[col] > (df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))))).sum()}" if col in col_num else "n/a" for col in df.columns],
        "outliers %": [f"{round((((df[col] < (df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))) | (df[col] > (df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))))).sum()) / df[col].shape[0] * 100)}%" if col in col_num else "n/a" for col in df.columns],
        "kurtosis": [f"{df[col].kurtosis(axis=0, skipna=True):.2f}" if col in col_num else "n/a" for col in df.columns],
        "skewness": [f"{df[col].skew(axis=0, skipna=True):.2f}" if col in col_num else "n/a" for col in df.columns],
        "corr": [f"{round(df[col].corr(other=df[y_label]) * 100)}%" if col in col_num else "n/a" for col in df.columns],
        "nunique": [f"{df[col].nunique()}" for col in df.columns],
        "unique": [sorted(df[col].unique()) if col in col_num else df[col].unique() for col in df.columns],
    }
).sort_values(by="dtype", ascending=False)
display(df_info)
print(f"Current Shape: {df.shape}.")
print("-------------------------------------------------------")
print(f"total na %: {df.isnull().sum().sum() / np.product(df.shape) * 100:.2f}%")
print("-------------------------------------------------------")
print(f"col_cat_oe ({len(col_cat_oe)}): {col_cat_oe}")
print(f"col_cat_ohe ({len(col_cat_ohe)}): {col_cat_ohe}")
print(f"col_num_disc ({len(col_num_disc)}): {col_num_disc}")
print(f"col_num_cont ({len(col_num_cont)}): {col_num_cont}")
print("-------------------------------------------------------")
print(f"total cols for preprocessor: {len(col_cat_oe) + len(col_cat_ohe) + len(col_num_disc) + len(col_num_cont)}")
#%%
MAX_COLS = 100
for col in col_cat_ohe:
    indice = df[col].value_counts()[:MAX_COLS].index
    df_temp = df[col].map(lambda value: value if value in indice else "-")
    # df[col] = df_temp
    print(df_temp.value_counts())
    print(f"unique values: {df_temp.nunique()}")
    print("-------------------------------------------------------")
try:
    display(df.describe(exclude="number").T.style.background_gradient(cmap="Blues"))
except Exception:
    pass
#%%
for col in col_cat_ohe:
    sr_temp = df.groupby(col)[y_label].count() / df.shape[0]
    df_temp = sr_temp[sr_temp > 0.01].index
    df[col] = np.where(df[col].isin(df_temp), df[col], "-")
#%%
for col in col_cat:
    fig, ax = plt.subplots(nrows=1, ncols=2)
    sns.countplot(x=df[col], ax=ax[0], hue=df[y_label] if CLASSIFICATION else None).set_xlabel(f"{col}")
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)
    ax[1].pie(x=df[col].value_counts(), autopct="%.1f%%", shadow=True, labels=df[col].value_counts().index)
    ax[1].set_title(col)
plt.show()
#%%
for col in col_num_disc + ([y_label] if CLASSIFICATION else []):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    sns.lineplot(x=df[col], y=df[y_label], ax=ax[0]).set_xlabel(f"{col}")
    sns.distplot(x=df[col], ax=ax[1], rug=True).set_xlabel(f"{col}")
plt.show()
#%%
OUTLIERS = ["keep", "cap", "drop", "log", "log1p", "reciprocal", "sqrt", "exp", "boxcox", "boxcox1"][0]
col_outlier = [col for col in col_num_cont + ([] if CLASSIFICATION else [y_label]) if col in df.columns]
q1, q3 = df[col_outlier].quantile(0.25), df[col_outlier].quantile(0.75)
iqr = q3 - q1
lower_range, upper_range = q1 - (1.5 * iqr), q3 + (1.5 * iqr)
condition = ~((df[col_outlier] < lower_range) | (df[col_outlier] > upper_range)).any(axis=1)
print(f"Current Shape: {df.shape}.")
print("-------------------------------------------------------")
print(f"Scanning for outliers in {col_outlier}.")
print(f"Outliers Percentage: {(df.shape[0] - df[condition].shape[0]) / df.shape[0] * 100:.2f}%")
print("-------------------------------------------------------")
print(f"upper_range:\n{upper_range}")
print("-------------------------------------------------------")
print(f"lower_range:\n{lower_range}")
print("-------------------------------------------------------")
if OUTLIERS == "keep":
    print(f"Outliers have been kept {df.shape}.")
elif OUTLIERS == "cap":
    for col in col_outlier:
        df[col] = np.where(df[col] < lower_range[col], lower_range[col], df[col])
        df[col] = np.where(df[col] > upper_range[col], upper_range[col], df[col])
    print(f"Outliers have been capped {df.shape}.")
elif OUTLIERS == "drop":
    df = df[condition]
    print(f"Outliers have been removed {df.shape}.")
elif OUTLIERS == "log":
    for col in col_outlier:
        df[col] = np.log(df[col])
    print(f"Outliers have been log transformed {df.shape}.")
elif OUTLIERS == "log1p":
    for col in col_outlier:
        df[col] = np.log1p(df[col])
    print(f"Outliers have been log1p transformed {df.shape}.")
elif OUTLIERS == "reciprocal":
    for col in col_outlier:
        df[col] = (1 / df[col])
    print(f"Outliers have been reciprocal transformed {df.shape}.")
elif OUTLIERS == "sqrt":
    for col in col_outlier:
        df[col] = (df[col] ** 0.5)
    print(f"Outliers have been sqrt transformed {df.shape}.")
elif OUTLIERS == "exp":
    for col in col_outlier:
        df[col] = (df[col] ** (1/1.2))
    print(f"Outliers have been exp transformed {df.shape}.")
elif OUTLIERS == "boxcox":
    for col in col_outlier:
        df[col] = stat.boxcox(df[col])[0]
    print(f"Outliers have been boxcox transformed {df.shape}.")
elif OUTLIERS == "boxcox1":
    for col in col_outlier:
        df[col] = stat.boxcox(df[col] + 1)[0]
    print(f"Outliers have been boxcox1 transformed {df.shape}.")
#%%
for col in col_num_cont + ([] if CLASSIFICATION else [y_label]):
    fig, ax = plt.subplots(nrows=1, ncols=3)
    sns.distplot(x=df[col], ax=ax[0], rug=True).set_xlabel(f"{col}")
    sns.boxplot(x=df[col], ax=ax[1], notch=True).set_xlabel(f"{col}")
    sm.qqplot(data=df[col], ax=ax[2], xlabel=col, ylabel="")
    print("-------------------------------------------------------")
    print(f"{col}\nSkew: {df[col].skew(axis=0, skipna=True):.2f}\nKurtosis: {df[col].kurtosis(axis=0, skipna=True):.2f}")
    plt.show()
#%%
for col in col_num_cont + ([] if CLASSIFICATION else [y_label]):
# for col in []:
    for trans in ["keep", "log", "log1p", "reciprocal", "sqrt", "exp", "boxcox", "boxcox1"]:
        try:
            fig, ax = plt.subplots(nrows=1, ncols=3)
            if trans == "keep":
                x = df[col]
            elif trans == "log":
                x = np.log(df[col])
            elif trans == "log1p":
                x = np.log1p(df[col])
            elif trans == "reciprocal":
                x=(1 / df[col])
            elif trans == "sqrt":
                x=(df[col] ** 0.5)
            elif trans == "exp":
                x=(df[col] ** (1/1.2))
            elif trans == "boxcox":
                x=pd.Series(stat.boxcox(df[col])[0], name=col)
            else:
                x=pd.Series(stat.boxcox(df[col] + 1)[0], name=col)
            sns.distplot(x=x, ax=ax[0], rug=True).set_xlabel(f"{col}")
            sns.boxplot(x=x, ax=ax[1], notch=True).set_xlabel(f"{col}")
            sm.qqplot(data=x, ax=ax[2], xlabel=col, ylabel="")
        except Exception:
            pass
        finally:
            for i in range(3):
                ax[i].set_title(f"{trans.title()} Transformation")
            print("-------------------------------------------------------")
            print(f"{col} {trans.title()} Transformation\nSkew: {x.skew(axis=0, skipna=True):.2f}\nKurtosis: {x.kurtosis(axis=0, skipna=True):.2f}")
            plt.show()
#%%
plt.title("Boxplots for Numeric Columns")
sns.boxplot(data=df[[col for col in col_num_cont]], orient="h", notch=True)
plt.show()
#%%
display(df.describe().T.style.background_gradient(cmap="Blues").format("{:.2f}"))
display(df.quantile([0.01, 0.99]).T.style.background_gradient(cmap="Blues").format("{:.2f}"))
#%%
df_plot = df.groupby(col_cat + [y_label]).size().reset_index().rename({0: "Count"}, axis=1)
display(df_plot)
sns.barplot(x=y_label, y="Count", hue=None, data=df_plot)
plt.show()
#%%
# for col in col_num:
#     if CLASSIFICATION:
#         fig, ax = plt.subplots(nrows=1, ncols=1)
#         sns.swarmplot(x=y_label, y=col, data=df, color="grey", alpha=0.7, ax=ax)
#         sns.boxenplot(x=y_label, y=col, data=df, ax=ax)
#     else:
#         sns.jointplot(x=col, y=y_label, hue=None, data=df, kind="reg")
#         sns.jointplot(x=col, y=y_label, hue=None, data=df, kind="hex")
#     plt.show()
#%%
# sns.pairplot(df, hue=y_label if CLASSIFICATION else None)
#%%
plot(df.drop(y_label, axis=1), df[y_label])
#%%
X, y = df.drop(y_label, axis=1), df[y_label]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=[None, y, y][CLASSIFICATION],
    random_state=RANDOM_STATE,
)
preprocessor_cat_ohe = make_pipeline(
    (SimpleImputer(strategy="most_frequent")),
    (OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)),
)
preprocessor_num_disc = make_pipeline(
    (KNNImputer(n_neighbors=5)),
)
preprocessor_num_cont = make_pipeline(
    (KNNImputer(n_neighbors=5)),
)
preprocessor_col = make_column_transformer(
    (preprocessor_cat_oe, col_cat_oe),
    (preprocessor_cat_ohe, col_cat_ohe),
    (preprocessor_num_disc, col_num_disc),
    (preprocessor_num_cont, col_num_cont),
    sparse_threshold=0
)
preprocessor = make_pipeline(
    (preprocessor_col),
    ([MinMaxScaler(), StandardScaler()][SCALER]),
    (VarianceThreshold(threshold=0)),
    (PCA())
)
X_train_processed, y_train_processed = preprocessor.fit_transform(X_train), y_train
X_test_processed, y_test_processed = preprocessor.transform(X_test), y_test
if CLASSIFICATION:
    class_weights = {i: w for i, w in enumerate(class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y_train_processed), y=y_train_processed))}
    fig, ax = plt.subplots(nrows=1, ncols=3)
    sns.despine(left=True)
    sns.countplot(y, ax=ax[0]).set_xlabel("y")
    sns.countplot(y_train, ax=ax[1]).set_xlabel("y_train")
    sns.countplot(y_train_processed, ax=ax[2]).set_xlabel("y_train_processed")
    plt.show()
    print("-------------------------------------------------------")
    print(f"class_weights: {class_weights}")
    print("-------------------------------------------------------")
    print(f"y:\n{y.value_counts(normalize=True)}")
    print("-------------------------------------------------------")
    print(f"y_train:\n{y_train.value_counts(normalize=True)}")
    print("-------------------------------------------------------")
    print(f"y_train_processed:\n{y_train_processed.value_counts(normalize=True)}")
    print("-------------------------------------------------------")
    print(f"y_test:\n{y_test.value_counts(normalize=True)}")
    print("-------------------------------------------------------")
    print(f"y_test_processed:\n{y_test_processed.value_counts(normalize=True)}")
print("-------------------------------------------------------")
print(f"col_cat_oe ({len(col_cat_oe)}): {col_cat_oe}")
print(f"col_cat_ohe ({len(col_cat_ohe)}): {col_cat_ohe}")
print(f"col_num_disc ({len(col_num_disc)}): {col_num_disc}")
print(f"col_num_cont ({len(col_num_cont)}): {col_num_cont}")
print("-------------------------------------------------------")
print(f"total cols for preprocessor: {len(col_cat_oe) + len(col_cat_ohe) + len(col_num_disc) + len(col_num_cont)}")
print("-------------------------------------------------------")
print(f"X: {X.shape}\tX_train: {X_train.shape}\tX_train_processed:{X_train_processed.shape}\tX_test: {X_test.shape}\t\tX_test_processed:{X_test_processed.shape}")
print(f"y: {y.shape}\ty_train: {y_train.shape}\t\ty_train_processed:{y_train_processed.shape}\ty_test: {y_test.shape}\t\ty_test_processed:{y_test_processed.shape}")
print("-------------------------------------------------------")
#%%
def build_auto_model():
    if CLASSIFICATION:
        SimpleClassifier(random_state=RANDOM_STATE).fit(clean(df), target_col=y_label)
    else:
        SimpleRegressor(random_state=RANDOM_STATE).fit(clean(df), target_col=y_label)

def build_ml_model():
    tests = [
        {
            "model": make_pipeline_imb(
                preprocessor_col,
                [MinMaxScaler(), StandardScaler()][SCALER],
                SMOTE(random_state=RANDOM_STATE),
                VarianceThreshold(threshold=0),
                PCA(),
                SelectPercentile(),
                RandomForestClassifier() if CLASSIFICATION else LinearRegression(),
            )
            if OVERSAMPLE
            else make_pipeline_imb(
                preprocessor_col,
                [MinMaxScaler(), StandardScaler()][SCALER],
                VarianceThreshold(threshold=0),
                PCA(),
                SelectPercentile(),
                RandomForestClassifier() if CLASSIFICATION else LinearRegression(),
            ),
            "params": {
                "columntransformer__pipeline-3__knnimputer__n_neighbors": [1, 3, 5, 7, 9],
                "columntransformer__pipeline-4__knnimputer__n_neighbors": [1, 3, 5, 7, 9],
                "selectpercentile__percentile": [i * 10 for i in range(1, 10)],
                "selectpercentile__score_func": [chi2, f_classif],
                "randomforestclassifier__n_estimators": [100, 150, 200, 500],
                "randomforestclassifier__criterion": ["gini", "entropy"],
                "randomforestclassifier__max_depth": [5, 10, 20, 50, 100, 200],
                "randomforestclassifier__min_samples_split": [2, 5, 10, 20, 50, 100, 200],
                "randomforestclassifier__min_samples_leaf": [5, 10, 20, 50, 100, 200],
                "randomforestclassifier__max_features": ["auto", "sqrt", "log2"],
            }
            if CLASSIFICATION
            else {
                "columntransformer__pipeline-3__knnimputer__n_neighbors": [1, 3, 5, 7, 9],
                "columntransformer__pipeline-4__knnimputer__n_neighbors": [1, 3, 5, 7, 9],
                "selectpercentile__percentile": [i * 10 for i in range(1, 10)],
                "selectpercentile__score_func": [chi2, f_classif],
            },
        },
    ]
    for test in tests:
        rscv = RandomizedSearchCV(
            estimator=test["model"],
            param_distributions=test["params"],
            n_jobs=-1,
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
            if CLASSIFICATION
            else 10,
            scoring="accuracy" if CLASSIFICATION else "r2",
            n_iter=10,
            return_train_score=True,
        )
        display(rscv)
        rscv.fit(X_train, y_train)
        print("===train============================")
        print(f"{rscv.best_score_ * 100:.2f}%\n{test['model'][-1]}\n{rscv.best_params_}")
        print("===params============================")
        display(pd.DataFrame(rscv.cv_results_).sort_values(by="rank_test_score"))
        print("===test============================")
        print(f"test score:{rscv.score(X_test, y_test) * 100:.2f}%")
        print("====end===========================\n")

    build_auto_model()
    print("-------------------------------------------------------")
    if CLASSIFICATION:
        print(
            classification_report(
                y_test,
                rscv.predict(X_test),
            )
        )
        sns.heatmap(
            tf.math.confusion_matrix(
                y_test,
                rscv.predict(X_test),
            ),
            cmap="Blues",
            fmt="d",
            annot=True,
            linewidths=1,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Truth")

    else:
        print(
            f"r2: {r2_score(y_test, rscv.predict(X_test)):.3f} neg_mean_squared_error: -{mean_squared_error(y_test, rscv.predict(X_test)):_.3f}"
        )
        plt.subplot(1, 3, 1)
        sns.regplot(y_train, y_train, color="darkorange", label="Truth")
        sns.regplot(
            y_test,
            rscv.predict(X_test),
            color="darkcyan",
            label="Predicted",
        )
        plt.title(
            "Truth vs Predicted",
            fontsize=10,
        )
        plt.xlabel("Truth values")
        plt.ylabel("Predicted values")
        plt.legend()
        plt.grid()

        plt.subplot(1, 3, 2)
        sns.scatterplot(
            data=pd.DataFrame(
                {
                    "Predicted values": rscv.predict(X_train),
                    "Residuals": rscv.predict(X_train) - y_train,
                }
            ),
            x="Predicted values",
            y="Residuals",
            color="darkorange",
            marker="o",
            s=35,
            alpha=0.5,
            label="Train data",
        )
        sns.scatterplot(
            data=pd.DataFrame(
                {
                    "Predicted values": rscv.predict(X_test),
                    "Residuals": rscv.predict(X_test) - y_test,
                }
            ),
            x="Predicted values",
            y="Residuals",
            color="darkcyan",
            marker="o",
            s=35,
            alpha=0.7,
            label="Test data",
        )
        plt.title(
            "Predicted vs Residuals",
            fontsize=10,
        )
        plt.hlines(y=0, xmin=0, xmax=df[y_label].max(), lw=2, color="red")
        plt.grid()

        plt.subplot(1, 3, 3)
        sns.distplot((y_train - rscv.predict(X_train)))
        plt.title("Error Terms")
        plt.xlabel("Errors")
        plt.grid()

    plt.show()

    display(
        pd.DataFrame(
            {
                "Truth": y_test[:10].values,
                "Predicted": rscv.predict(X_test[:10]).round(1),
            }
        )
    )

def tune_dl_model(hp):
    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            units=hp.Int("input_00", min_value=32, max_value=512, step=32),
            input_shape=X_train_processed.shape[1:],
        )
    )
    for i in range(1, hp.Int("num_layers", min_value=2, max_value=64)):
        model.add(
            keras.layers.Dense(
                units=hp.Int(f"hidden_{i:02}", min_value=32, max_value=512, step=32),
                activation="relu",
            )
        )
        model.add(keras.layers.Dropout(hp.Float("dropout", min_value=0, max_value=0.5, step=0.1)))
    model.add(
        keras.layers.Dense(
            units=[1, 1, df[y_label].nunique()][CLASSIFICATION],
            activation=["linear", "sigmoid", "softmax"][CLASSIFICATION],
        )
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss=["mean_squared_error", "binary_crossentropy", "sparse_categorical_crossentropy"][CLASSIFICATION],
        metrics=["mean_squared_error", "accuracy", "accuracy"][CLASSIFICATION],
    )
    return model

def build_dl_model(epochs):
    model = tuner.hypermodel.build(best_hps)
    model.fit(
        X_train_processed,
        y_train_processed,
        batch_size=256 if tf.config.list_physical_devices("GPU") else 64,
        epochs=epochs,
        validation_split=0.2,
        verbose=1,
        class_weight=class_weights if OVERSAMPLE else None
    )
    build_auto_model()
    print("-------------------------------------------------------")
    if CLASSIFICATION:
        print(
            classification_report(
                y_test_processed,
                [
                    model.predict(X_test_processed).round(),
                    np.argmax(model.predict(X_test_processed), axis=1),
                ][CLASSIFICATION - 1],
            )
        )
        sns.heatmap(
            tf.math.confusion_matrix(
                y_test_processed,
                [
                    model.predict(X_test_processed).round(),
                    np.argmax(model.predict(X_test_processed), axis=1),
                ][CLASSIFICATION - 1],
            ),
            cmap="Blues",
            fmt="d",
            annot=True,
            linewidths=1,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Truth")

    else:
        print(f"r2: {r2_score(y_test_processed, model.predict(X_test_processed).T[0]):.3f} neg_mean_squared_error: -{mean_squared_error(y_test_processed, model.predict(X_test_processed)):_.3f}")
        plt.subplot(1, 3, 1)
        sns.regplot(y_train_processed, y_train_processed, color="darkorange", label="Truth")
        sns.regplot(
            y_test_processed,
            model.predict(X_test_processed).T[0],
            color="darkcyan",
            label="Predicted",
        )
        plt.title(
            "Truth vs Predicted",
            fontsize=10,
        )
        plt.xlabel("Truth values")
        plt.ylabel("Predicted values")
        plt.legend()
        plt.grid()

        plt.subplot(1, 3, 2)
        sns.scatterplot(
            data=pd.DataFrame(
                {
                    "Predicted values": model.predict(X_train_processed).T[0],
                    "Residuals": model.predict(X_train_processed).T[0] - y_train_processed,
                }
            ),
            x="Predicted values",
            y="Residuals",
            color="darkorange",
            marker="o",
            s=35,
            alpha=0.5,
            label="Train data",
        )
        sns.scatterplot(
            data=pd.DataFrame(
                {
                    "Predicted values": model.predict(X_test_processed).T[0],
                    "Residuals": model.predict(X_test_processed).T[0] - y_test_processed,
                }
            ),
            x="Predicted values",
            y="Residuals",
            color="darkcyan",
            marker="o",
            s=35,
            alpha=0.7,
            label="Test data",
        )
        plt.title(
            "Predicted vs Residuals",
            fontsize=10,
        )
        plt.hlines(y=0, xmin=0, xmax=df[y_label].max(), lw=2, color="red")
        plt.grid()
        
        plt.subplot(1, 3, 3)
        sns.distplot((y_train_processed - model.predict(X_train_processed).T[0]))
        plt.title("Error Terms")
        plt.xlabel("Errors")
        plt.grid()

    plt.show()

    display(
        pd.DataFrame(
            {
                "Truth": y_test_processed[:10].values,
                "Predicted": [
                    model.predict(X_test_processed[:10]).T[0],
                    model.predict(X_test_processed[:10]).T[0].round(),
                    np.argmax(model.predict(X_test_processed[:10]), axis=1),
                ][CLASSIFICATION],
            }
        )
    )
    return model

if SEARCH == "hyperband":
    tuner = Hyperband(
        tune_dl_model,
        objective=["val_mean_squared_error", "val_accuracy", "val_accuracy"][CLASSIFICATION],
        max_epochs=MAX_TRIALS,
        factor=3,
        directory=".",
        project_name="keras_tuner",
        overwrite=True,
    )
elif SEARCH == "random":
    tuner = RandomSearch(
        tune_dl_model,
        objective=["val_mean_squared_error", "val_accuracy", "val_accuracy"][CLASSIFICATION],
        max_trials=MAX_TRIALS,
        executions_per_trial=3,
        directory=".",
        project_name="keras_tuner",
        overwrite=True,
    )
else:
    tuner = BayesianOptimization(
        tune_dl_model,
        objective=["val_mean_squared_error", "val_accuracy", "val_accuracy"][CLASSIFICATION],
        max_trials=MAX_TRIALS,
        executions_per_trial=3,
        directory=".",
        project_name="keras_tuner",
        overwrite=True,
    )
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(MAX_TRIALS/4))
tuner.search_space_summary()
#%%
# build_auto_model()
#%%
OVERSAMPLE = 0
build_ml_model()
#%%
# %pip install pycaret
#%%
# import chardet
# import pandas as pd
# import requests
# from pycaret.classification import *
# from pycaret.regression import *

# url = "https://raw.githubusercontent.com/lyoh001/AzureML/main/data.csv"
# df = pd.read_csv(url, delimiter=",", encoding=chardet.detect(requests.get(url).content)["encoding"], thousands=",")
# y_label = "target"

# model = setup(data=df, target=y_label)
# compare_models()
#%%
# %%time
# OVERSAMPLE = 0
# tuner.search(
#     X_train_processed,
#     y_train_processed,
#     batch_size=256 if tf.config.list_physical_devices("GPU") else 64,
#     callbacks=[early_stop],
#     epochs=MAX_TRIALS,
#     validation_split=0.2,
#     verbose=1,
# )
# tuner.results_summary()

# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# model = tuner.hypermodel.build(best_hps)
# history = model.fit(
#     X_train_processed,
#     y_train_processed,
#     batch_size=256 if tf.config.list_physical_devices("GPU") else 64,
#     epochs=EPOCHS,
#     validation_split=0.2,
#     verbose=1,
#     class_weight=class_weights if OVERSAMPLE else None
# )
# val_per_epoch = history.history[
#     ["val_mean_squared_error", "val_accuracy", "val_accuracy"][CLASSIFICATION]
# ]
# best_epoch = val_per_epoch.index([min(val_per_epoch), max(val_per_epoch), max(val_per_epoch)][CLASSIFICATION]) + 1

# plt.subplot(1, 2, 1)
# sns.lineplot(data=history.history[["mean_squared_error", "accuracy", "accuracy"][CLASSIFICATION]], color="deeppink", linewidth=2.5)
# sns.lineplot(data=history.history[["val_mean_squared_error", "val_accuracy", "val_accuracy"][CLASSIFICATION]], color="darkturquoise", linewidth=2.5)
# plt.title("Model Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(["Training Accuracy", "Val Accuracy"], loc="lower right")
# plt.grid()

# plt.subplot(1, 2, 2)
# sns.lineplot(data=history.history["loss"], color="deeppink", linewidth=2.5)
# sns.lineplot(data=history.history["val_loss"], color="darkturquoise", linewidth=2.5)
# plt.title("Model Loss")
# plt.ylabel("Loss")
# plt.xlabel("Epoch")
# plt.legend(["Training Loss", "Val Loss"], loc="upper right")
# plt.grid()
# plt.show()

# print(f"Best epoch: {best_epoch}")
# model = build_dl_model(best_epoch)
#%%
# model = build_dl_model(best_epoch)
#%%
# model.save(f"dl_model_{time_stamp}")
# shutil.make_archive(f"dl_model_{time_stamp}", "zip", f"./dl_model_{time_stamp}")
# dump(preprocessor, open(f"dl_preprocessor.pkl", "wb"))
# model.summary()
# plot_model(model, show_shapes=True)
#%%
# import logging
# import shutil
# import warnings
# from pickle import load

# import azure.functions as func
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow import keras

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# warnings.filterwarnings("ignore")


# def main(req: func.HttpRequest) -> func.HttpResponse:
#     logging.info("*******Starting main function*******")
#     logging.info(f"Request query: {req.get_json()}")
#     shutil.unpack_archive("dl_model.zip", "dl_model")
#     model = keras.models.load_model("dl_model")
#     preprocessor = load(open("dl_preprocessor/dl_preprocessor.pkl", "rb"))
#     payload = pd.DataFrame(
#         {k: [np.nan] if next(iter(v)) == "" else v for k, v in req.get_json().items()},
#         dtype="object",
#     )
#     logging.info("*******Finishing main function*******")
#     return func.HttpResponse(
#         status_code=200,
#         body=f"{model.predict(preprocessor.transform(payload))[0][0]:.2f}",
#     )
#     return func.HttpResponse(
#         status_code=200,
#         body=["", ""][
#             np.argmax(model.predict(preprocessor.transform(payload)))
#         ],
#     )
