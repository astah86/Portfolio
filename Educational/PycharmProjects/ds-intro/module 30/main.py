import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_selector
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC


def filter_data(df):
    dat = df.copy()
    columns_to_drop = [
        'id',
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long'
    ]

    return dat.drop(columns_to_drop, axis=1)


def new_columns(df):
    dat = df.copy()

    dat.loc[:, 'short_model'] = dat['model'].apply(lambda x: x.lower().split(' ')[0] if not pd.isna(x) else x)
    dat.loc[:, 'age_category'] = dat['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else
                                                                                     'average'))
    return dat


def calculate_outliers(df):
    q25 = df['year'].quantile(0.25)
    q75 = df['year'].quantile(0.75)
    dat = df.copy()
    iqr = q75 - q25
    boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
    dat.loc[df['year'] < boundaries[0], 'year'] = round(boundaries[0])
    dat.loc[df['year'] > boundaries[1], 'year'] = round(boundaries[1])

    return dat


def main():

    df = pd.read_csv('C:/Users/Msi-1/ds-intro/30_Deployment/homework.csv')

    X = df.drop(['price_category'], axis=1)
    y = df['price_category']

    numerical_transformer = Pipeline(steps=[
        ('inputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('inputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    data_preparation = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('outliers', FunctionTransformer(calculate_outliers)),
        ('new', FunctionTransformer(new_columns))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        ('categorical', categorical_transformer, make_column_selector(dtype_include=object))
    ])

    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC()
    )

    best_score = .0
    best_pipe = None
    for model in models:

        pipe = Pipeline(steps=[
            ('preparation', data_preparation),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy', error_score='raise')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    joblib.dump(best_pipe, 'car_pipe.pkl')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
