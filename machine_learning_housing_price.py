import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.utils import resample
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix


train_path = sys.argv[1]
test_path = sys.argv[2]

df = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# clean those super extreme price values
def extreme_value_cleansing(value,df):
    # calculate the IQR
    Q1 = df[value].quantile(0.25)
    Q3 = df[value].quantile(0.75)
    IQR = Q3 - Q1

    # set the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # replace outliers with median
    df[value] = df.apply(lambda row: row[value] if (row[value] >= lower_bound) and (row[value] <= upper_bound)
                       else df[value].median(), axis=1)
    return df

def fill_zero_with_median(df, col):
    median_value = df[col][df[col] != 0].median()  
    df[col] = df[col].replace(0, median_value)
    return df

# Step 1 data cleaning and preprocssing 
def data_preprocessing(df, label = None):

    # data cleaning: drop columns
    
    # drop the columns that are missing values 
    df = df.drop(columns=['time_to_cbd_driving_town_hall_st', 'time_to_cbd_public_transport_town_hall_st'])

    # check if there are duplicated rows in the dataset
    # there's ducpliated rows in the dataset: suburbpopulation and suburb_population
    # drop suburbpopulation since there's a mixed data type in the column
    df = df.drop(columns=['suburbpopulation','nearest_train_station','highlights_attractions','ideal_for'])

    # drop the columns that are not related to the target variable(from subjective knowledge)
    df = df.drop(columns=['id','postcode'])

    # engineereing preprocessing: 
    # convert date_sold to datetime and extract the year

    df['date_sold'] = pd.to_datetime(df['date_sold'])
    df['year_sold'] = df['date_sold'].dt.year
    # df['quarter_sold'] = df['date_sold'].dt.quarter
    df['price_year_ratio'] = df['suburb_median_house_price'] / (df['year_sold'] + 1)
    df['day_sold'] = df['date_sold'].dt.day
    df = df.drop(columns=['date_sold'])

    # transfer the object features into numerical features
    df['public_housing_pct'] = df['public_housing_pct'].str.rstrip('%').astype(float)

    # frequency encoding for the descriptive features
    for col in ['region','suburb']:
        freq_encoding = df[col].value_counts(normalize=True).to_dict()
        df[f'{col}_fre'] = df[col].map(freq_encoding)

    df['region_price_diff'] = df.groupby('region_fre')['suburb_median_house_price'].transform('mean') - df['suburb_median_house_price']
        
    df = df.drop(columns=['region','suburb'])

    # fill up the 0 value with median price
    for col in ['suburb_median_house_price','suburb_median_apartment_price','median_house_rent_per_week','median_apartment_rent_per_week','avg_years_held']:
        non_zero_mean = df.loc[df[col] != 0, col].mean()
        df.loc[df[col] == 0, col] = non_zero_mean

    le = LabelEncoder()
    le.fit(df['type'])

    df['type_encoded'] = le.transform(df['type'])
    
    # binning property size into several bins
    df['property_size_bin'] = pd.cut(
    df['property_size'],
    bins=[0, 100, 200, 300, 500, 700, 1000, 1500, 2000, 10000],
    labels=[1,2,3,4,5,6,7,8,9], 
    include_lowest=True
    )

    # binning apartment and house price into several bins
    df['suburb_house_price_bin'] = pd.cut(
    df['suburb_median_house_price'],
    bins=[300000, 800000, 1000000, 1500000,2000000, 4000000, 6000000, 8000000, 10000000, 12000000],
    labels=[1,2,3,4,5,6,7,8,9], 
    include_lowest=True
    )

    df['suburb_apartment_price_bin'] = pd.cut(
    df['suburb_median_apartment_price'],
    bins=[300000, 400000, 500000, 600000,700000, 800000, 900000, 1000000, 1500000, 2000000],
    labels=[1,2,3,4,5,6,7,8,9], 
    include_lowest=True
    )

    # combine num_bath and num_bed into one column
    df['num_bath_bed'] = df['num_bath'] + df['num_bed']
    df = df.drop(columns=['num_bath','num_bed'])
  
    # create new features from the existing ones, which is to extract the suburb population density from the suburb population and suburb area
    # df['suburb_population_density'] = df['suburb_population']/df['suburb_sqkm']
    df['house_rent_yield'] = df['median_house_rent_per_week'] * 52 / df['suburb_median_house_price']
    df['apartment_rent_yield'] = df['median_apartment_rent_per_week'] *52 / df['suburb_median_apartment_price']
    df['property/room_ratio'] =df['property_size']/ df['num_bath_bed']
    
    # replace those infinite values with nan and then fill up them with median
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['house_rent_yield'] = df['house_rent_yield'].fillna(df['house_rent_yield'].median())
    df['apartment_rent_yield'] = df['apartment_rent_yield'].fillna(df['apartment_rent_yield'].median())

    # apply to each row and expand into separate columns
    ethnic_df = df['ethnic_breakdown'].apply(parse_ethnic_string).apply(pd.Series)
    # concat the new columns with the original dataframe
    df_new = pd.concat([df.drop(columns=['ethnic_breakdown']), ethnic_df], axis=1)

    # delete the columns with too many missing values, filter those the percentage of missing values is greater than 20%
    missing_percentage = df_new.isnull().mean()
    df_new = df_new.drop(columns=missing_percentage[missing_percentage > 0.03].index.tolist())
   
    df = df_new.copy()
    df = zero_percentage(0.1, df)

    df = df.drop(columns=['property_size','suburb_median_house_price','suburb_median_apartment_price'])


    return df, le

# extract ethcnicity from into different columns
# use , to split the string into a list of items
def parse_ethnic_string(s):
    result = {}
    items = s.split(',')
    for item in items:
        if ' ' in item:
            group, perc = item.rsplit(' ', 1)
            result[group.strip()] = float(perc.strip('%'))
    return result

# delete the columns with too many zero values, filter those the percentage of zero values is greater than 20%
def zero_percentage(threshold, df):
    zero_ratio = (df == 0).sum() / len(df)
    columns_to_keep = zero_ratio[zero_ratio <= threshold].index
    df_filtered = df[columns_to_keep]
    return df_filtered


if __name__ == "__main__":
    # create x and y for training and testing
    def regressor():
        x_train, type_encoder = data_preprocessing(df)
        x_train = x_train.drop(columns=['price'])
        x_train = x_train.drop(columns=['type'])
        y_train = df['price']
        y_train = y_train[x_train.index]
        y_train_log = np.log1p(y_train)

        expected_columns = x_train.columns

        x_test, _ = data_preprocessing(df_test, label = type_encoder)
        x_test = x_test.reindex(columns = expected_columns, fill_value=0)
        y_test = df_test['price']
        y_test = y_test[x_test.index]

        param_dist = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'n_estimators': [100, 200, 300, 400],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_lambda': [0, 0.1, 1, 10],
    'reg_alpha': [0, 0.01, 0.1, 1]
}

        # create a XGBoost model
        model = XGBRegressor(objective='reg:squarederror', tree_method='hist', enable_categorical=True, random_state=42)

        # create a random search object
        random_search = RandomizedSearchCV(estimator=model, 
                                        param_distributions=param_dist, 
                                        n_iter=50,  # random search iterations
                                        scoring='neg_mean_squared_error',  # common scoring metric for regression
                                        cv=5,  # cross-validation folds
                                        verbose=1, 
                                        random_state=42,
                                        n_jobs=-1)

        # use random search to find the best parameters
        random_search.fit(x_train, y_train)

        # use the best parameters to train the model
        best_model = random_search.best_estimator_

        
        y_pred = best_model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)

        xgb = XGBRegressor(
        objective='reg:squarederror',
        enable_categorical=True,
        tree_method='hist',
        random_state=42
    )

        best_params = {
        'subsample': 0.9, 'reg_lambda': 10, 'reg_alpha': 0.1, 'n_estimators': 300, 'min_child_weight': 5, 'max_depth': 5, 'learning_rate': 0.05, 'gamma': 0.3, 'colsample_bytree': 0.6
    }
        # create a linear regression model
        model = XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        enable_categorical=True,
        random_state=42,
        **best_params  
    )

        # fit the model to the training data
        model.fit(x_train, y_train_log)
        y_train_pred = model.predict(x_train)
        y_train_pred = np.expm1(y_train_pred)
        y_pred = model.predict(x_test)
        y_pred = np.expm1(y_pred)


        reg_out = pd.DataFrame({'id': df_test['id'],'price': y_pred})
        reg_out.to_csv('regression.csv', index=False)
        

        print("train performance:")
        print("MAE:", mean_absolute_error(y_train, y_train_pred))


        print("test performance:")
        print("MAE:", mean_absolute_error(y_test, y_pred))
 
    
    def classification():
        x_train, type_encoder = data_preprocessing(df)
        x_train.to_csv('df_x_train.csv', index=True)
        

        y_train = x_train['type_encoded']
        x_train = x_train.drop(columns=['type', 'type_encoded'])
        
        y_train = y_train[x_train.index]

        expected_columns = x_train.columns


        x_test, _ = data_preprocessing(df_test, label=type_encoder)

        
        # code the 'type' column in the test set using the same encoder
        y_test = type_encoder.transform(x_test['type'].values)  
        
        x_test = x_test.drop(columns=['type', 'type_encoded'])
        x_test = x_test.reindex(columns=expected_columns, fill_value=0)

        y_test = y_test[x_test.index]

        # training and testing the model
        clf = xgb.XGBClassifier(enable_categorical=True, use_label_encoder=False, eval_metric='mlogloss')
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)
        y_pred_labels = type_encoder.inverse_transform(y_pred)

        reg_out = pd.DataFrame({'id': df_test['id'],'type': y_pred_labels})
        reg_out.to_csv('classificatioon.csv', index=False)

        print ("Classification Report:")
        print(classification_report(y_test, y_pred,zero_division=0))

# run the regressor and classifier
regressor()
classification()
