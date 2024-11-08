import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import roc_auc_score
import dill
from datetime import datetime
from sklearn.feature_selection import SelectPercentile


def data_preparation(df1):

    category_columns = []
    col1 = df1.drop(columns=['id']).columns

    for i in col1:
        if len(df1[i].unique()) > 2:
            category_columns.append(i)

    data1 = df1[category_columns]
    ohe = OneHotEncoder(sparse=False, drop='first', dtype='int8')
    ohe.fit(data1)
    df1.drop(columns=category_columns, inplace=True)
    df1[ohe.get_feature_names_out()] = ohe.fit_transform(data1)

    del data1
    del ohe

    return df1


def sum_data(df2):

    df2 = df2.groupby(by=['id']).sum()
    df2 = pd.DataFrame(df2)

    return df2


def filters(dfx):

    SP = SelectPercentile(percentile=50)

    df3 = SP.fit_transform(dfx, y)

    return df3


df = pd.read_csv('data.csv')
df[df.drop(columns=['id']).columns] = df[df.drop(columns=['id']).columns].astype('int8')
X = df.drop(columns=['rn', 'flag'])
y = pd.read_csv('train_target.csv').astype('int8')['flag']

model = HistGradientBoostingClassifier(l2_regularization=200, loss='binary_crossentropy', max_depth=20,
                                       max_iter=500, max_leaf_nodes=41, min_samples_leaf=10, learning_rate=0.04)

pipe = Pipeline(steps=[
    ('preparation', FunctionTransformer(data_preparation)),
    ('sum', FunctionTransformer(sum_data)),
    ('filter', FunctionTransformer(filters)),
    ('classifier', model)
], verbose=True)

pipe.fit(X, y)
print('pickle')
with open('final.pkl', 'wb') as file:
    dill.dump({
        'model': pipe,
        'metadata': {
            'name': 'Session predict model',
            'author': 'Michael Astoshonok',
            'version': 1,
            'date': datetime.now(),
            'type': type(pipe.named_steps["classifier"]).__name__
        }
    }, file)



