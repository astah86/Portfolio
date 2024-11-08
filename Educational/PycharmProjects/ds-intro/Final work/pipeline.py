import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_selector
from sklearn.metrics import roc_auc_score
import dill
from datetime import datetime


def filter_data(session):
    from sklearn.impute import SimpleImputer

    sess = session.copy()

    imp_const_unknown = SimpleImputer(strategy='constant', fill_value='unknown')
    imp_const_notset = SimpleImputer(strategy='constant', fill_value='(not set)')
    imp_const_freq = SimpleImputer(strategy='most_frequent')

    unk_cols = ['utm_source', 'utm_campaign']
    notset_cols = ['device_brand']
    freq_cols = ['utm_adcontent']

    sess[unk_cols] = imp_const_unknown.fit_transform(sess[unk_cols])
    sess[notset_cols] = imp_const_notset.fit_transform(sess[notset_cols])
    sess[freq_cols] = imp_const_freq.fit_transform(sess[freq_cols])

    sess = sess.drop(columns=['session_id', 'client_id', 'utm_keyword', 'device_os', 'device_model'])

    sess.loc[sess['utm_source'].value_counts()[sess['utm_source']].values < 50, 'utm_source'] = 'rare'
    sess.loc[sess['utm_campaign'].value_counts()[sess['utm_campaign']].values < 100, 'utm_campaign'] = 'rare'
    sess.loc[sess['utm_adcontent'].value_counts()[sess['utm_adcontent']].values < 50, 'utm_adcontent'] = 'rare'
    sess.loc[sess['device_brand'].value_counts()[sess['device_brand']].values < 5, 'device_brand'] = 'rare'
    sess.loc[sess['utm_medium'].value_counts()[sess['utm_medium']].values < 5, 'utm_medium'] = 'rare'

    sess.device_browser = sess.device_browser.str.split(expand=True).iloc[:, 0]
    sess = sess.replace({'device_browser': {'Mozilla': 'Firefox', 'MRCHROME': 'Chrome', '(not': '(not set)'}})

    sess.device_screen_resolution = sess.device_screen_resolution.str.split("x", expand=True).astype(
        'int32').iloc[:, 0] * sess.device_screen_resolution.str.split("x", expand=True).astype('int32').iloc[:, 1]

    return sess


def dates(session):
    sess = session.copy()
    sess.visit_date = sess.visit_date.astype('datetime64')
    sess.visit_time = sess.visit_time.astype('datetime64').dt.hour
    sess['day'] = sess.visit_date.dt.day
    sess['day_of_week'] = sess.visit_date.dt.day_of_week
    sess['month'] = sess.visit_date.dt.month
    sess = sess.drop(columns=['visit_date'])

    return sess


def new_columns(session):
    sess = session.copy()
    sess.visit_number = 1 / sess.visit_number ** 2

    sess.device_screen_resolution = sess.device_screen_resolution.apply(
        lambda x: x if (x < 9 * 10 ** 6) else 9 * 10 ** 6)

    return sess


def main():
    session = pd.read_csv('sess.csv', low_memory=False)

    country = session.geo_country[session.target == 1].unique()
    city = session.geo_city[session.target == 1].unique()

    session.geo_country = session.geo_country.apply(lambda x: x if x in country else 'other')
    session.geo_city = session.geo_city.apply(lambda x: x if x in city else 'other')

    X = session.drop(['target'], axis=1)
    y = session['target']

    data_preparation = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('dates', FunctionTransformer(dates)),
        ('new', FunctionTransformer(new_columns))
    ], verbose=True)

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ], verbose=True)

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(sparse=False, drop='first', dtype='int8'))
    ], verbose=True)

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        ('categorical', categorical_transformer, make_column_selector(dtype_include=object))
    ], verbose=True)

    model = HistGradientBoostingClassifier(l2_regularization=10, learning_rate=0.2, loss='binary_crossentropy',
                                           max_depth=10, max_iter=200, max_leaf_nodes=26, min_samples_leaf=10)

    pipe = Pipeline(steps=[
        ('preparation', data_preparation),
        ('preprocessor', preprocessor),
        ('classifier', model)
    ], verbose=True)

    pipe.fit(X, y)
    with open('final.pkl', 'wb') as file:
        dill.dump({
            'model': pipe,
            'metadata': {
                'name': 'Session predict model',
                'author': 'Michael Astoshonok',
                'version': 1,
                'date': datetime.now(),
                'type': type(pipe.named_steps["classifier"]).__name__,
                'roc-auc': roc_auc_score(y, pipe.predict_proba(X)[:, 1])
            }
        }, file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
