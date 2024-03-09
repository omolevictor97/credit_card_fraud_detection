import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler

class DataTransformation:
    def __init__(self, data):
        self.data_copy = data

    def transform(self):
        """
        The method returns many things which will include:
        (X) : The original predictors before preprocessing
        (y) : The original target variable before preprocessing
        (X_train) : Train dataset after preproessing
        (X_test) : Test dataset after preprocessing
        (y_train) : Target train dataset after preprocessing
        (y_test) : Target test dataset after preprocessing
        """ 
        self.X = self.data_copy.drop(["SK_ID_CURR", "TARGET"], axis=1)
        self.y = self.data_copy.TARGET

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42, shuffle=True)
        #Treat the issue of Imbalance dataset
        ros = RandomOverSampler(random_state=42)
        self.X_train, self.y_train = ros.fit_resample(self.X_train, self.y_train)
        cat_feats = self.X_train.select_dtypes(exclude="number").columns
        num_feats = self.X_train.select_dtypes(exclude="O").columns

        #Preprocessing starts here
        numerical_processor = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler(with_mean=False))
        ]
        )   

        categorical_processor = Pipeline(
            steps = [
                ('imputer', SimpleImputer(strategy='constant')),
                ('cat_encoder', (OneHotEncoder(handle_unknown='ignore'))),
                ('scaler', StandardScaler(with_mean=False))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num_pre', numerical_processor, num_feats),
                ('cat_pre', categorical_processor, cat_feats)
            ]
        )

        self.X_train = pd.DataFrame(preprocessor.fit_transform(self.X_train), columns=preprocessor.get_feature_names_out())
        self.X_test = pd.DataFrame(preprocessor.transform(self.X_test), columns=preprocessor.get_feature_names_out())

        return self.X, self.y, self.X_train, self.X_test, self.y_train, self.y_test
