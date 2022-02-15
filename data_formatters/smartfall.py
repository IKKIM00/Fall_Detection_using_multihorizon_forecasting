# -*- coding: utf-8 -*-
import data_formatters.base
import libs.utils as utils
import pandas as pd
import sklearn.preprocessing
from sklearn.model_selection import train_test_split

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes

class SmartFallFormatter(GenericDataFormatter):
    _column_definition = [
        ('per-id', DataTypes.CATEGORICAL, InputTypes.ID),
        ('circum', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('time', DataTypes.REAL_VALUED, InputTypes.TIME),
        ('known', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        (' ms_accelerometer_y', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        (' ms_accelerometer_z', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        (' ms_accelerometer_x', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('outcome', DataTypes.REAL_VALUED, InputTypes.TARGET)
    ]

    def __init__(self):
        """initializes formatter"""

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None

    def split_data(self, dataset_dir):
        print('Formatting train-valid-test splits.')

        train = pd.read_csv(dataset_dir + 'SmartFall Training.csv')
        test = pd.read_csv(dataset_dir + 'SmartFall Testing.csv')
        train['per-id'] = 0
        train['known'] = 0
        train['time'] = list(range(len(train)))
        train['circum'] = 0
        train.outcome = train.outcome.astype(float)
        test['per-id'] = 0
        test['known'] = 0
        test['time'] = list(range(len(test)))
        test['circum'] = 0
        test.outcome = test.outcome.astype(float)

        train, valid = train_test_split(train, test_size=0.25, shuffle=False, random_state=0)

        self.set_scalers(train)
        return (self.transform_inputs(data) for data in [train, valid, test])

    def set_scalers(self, df):
        """
        Calibrates scalers using the data supplied.
        (제공된 데이터를 사용해서 scaler 교정)

        Args:
          df: Data to use to calibrate scalers.
        """
        print('Setting scalers with training data...')

        column_definitions = self.get_column_definition()

        id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                       column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                           column_definitions)
        # extract identifiers in case required
        #         stock에는 identifier가 현재 따로 필요해보이지 않음(데이터 entity가 한개이기 때문)
        self.identifiers = list(df[id_column].unique())

        # Format real scalers
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        data = df[real_inputs].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)

        # target데이터를 prediction에 사용
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(df[[target_column]].values)

        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            srs = df[col].apply(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(srs.values)
            num_classes.append(srs.nunique())

        # set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

    def transform_inputs(self, df):
        """
        performs feature transformations.

        feature engineering, preprocessing and normalization을 포함

        Args:
            df - data frame to transform.

        Returns:
            Transformed data frame.
        """
        output = df.copy()

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set!')

        column_definitions = self.get_column_definition()
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME}
        )
        #         categorical_inputs = utils.extract_cols_from_data_type(
        #             DataTypes.CATEGORICAL, column_definitions,
        #             {InputTypes.ID, InputTypes.TIME})
        output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)
        #         for col in categorical_inputs:
        #             string_df = df[col].apply(str)
        #             output[col] = self._cat_scalers[col].transform(string_df)
        return output

    def format_predictions(self, predictions):
        output = predictions.copy()

        column_names = predictions.columns
        for col in column_names:
            if col not in {'forecast_time', 'identifier'}:
                output[col] = self._target_scaler.inverse_transform(predictions[col])

        return output

    def get_fixed_params(self):
        fixed_params = {
            'total_time_steps': 64,  # Total width of the Temporal Fusion Decoder
            'num_encoder_steps': 32,  # Length of LSTM decoder (ie. # historical inputs)
            'num_epochs': 100,  # Max number of epochs for training
            'early_stopping_patience': 5,  # Early stopping threshold for # iterations with no loss improvement
            'multiprocessing_workers': 5  # Number of multi-processing workers
        }

        return fixed_params

    def get_default_model_params(self):
        model_params = {'dropout_rate': 0.3,  # Dropout discard rate
                        'hidden_layer_size': 320,  # Internal state size of TFT
                        'learning_rate': 0.001,  # ADAM initial learning rate
                        'minibatch_size': 256,  # Minibatch size for training
                        'max_gradient_norm': 100.,  # Max norm for gradient clipping
                        'num_heads': 4,  # Number of heads for multi-head attention
                        'stack_size': 1  # Number of stacks (default 1 for interpretability)
                        }
        return model_params