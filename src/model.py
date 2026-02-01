"""Model module for Credit Risk Scoring System."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class CreditRiskPreprocessor:
    """
    Handles data preprocessing for Credit Risk Scoring System
    - Handles missing values
    - Encodes categorical variables
    - Scales numerical features
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = ['Income', 'LoanAmount', 'CreditHistory', 'WorkExperience', 'HomeOwnership']
        self.numerical_features = ['Income', 'LoanAmount', 'CreditHistory']
        self.categorical_features = ['WorkExperience', 'HomeOwnership']
        self.work_exp_mapping = {'0-2 years': 0, '2-5 years': 1, '5+ years': 2}
        self.home_ownership_mapping = {'Rent': 0, 'Mortgage': 1, 'Own': 2}

    def fit_transform(self, df):
        """Fit and transform the dataset"""
        df_processed = df.copy()

        # Handle missing values
        df_processed[self.numerical_features] = df_processed[self.numerical_features].fillna(
            df_processed[self.numerical_features].median()
        )

        # Encode categorical variables
        df_processed['WorkExperience'] = df_processed['WorkExperience'].map(self.work_exp_mapping)
        df_processed['HomeOwnership'] = df_processed['HomeOwnership'].map(self.home_ownership_mapping)

        # Scale numerical features
        df_processed[self.feature_names] = self.scaler.fit_transform(df_processed[self.feature_names])

        return df_processed

    def transform(self, df):
        """Transform new data using fitted parameters"""
        df_processed = df.copy()

        # Handle missing values
        df_processed[self.numerical_features] = df_processed[self.numerical_features].fillna(
            df_processed[self.numerical_features].median()
        )

        # Encode categorical variables
        df_processed['WorkExperience'] = df_processed['WorkExperience'].map(self.work_exp_mapping)
        df_processed['HomeOwnership'] = df_processed['HomeOwnership'].map(self.home_ownership_mapping)

        # Scale numerical features
        df_processed[self.feature_names] = self.scaler.transform(df_processed[self.feature_names])

        return df_processed

    def get_scaling_params(self):
        """Get mean and std for SQL implementation"""
        return {
            'means': self.scaler.mean_.tolist(),
            'stds': self.scaler.scale_.tolist(),
            'feature_names': self.feature_names
        }
