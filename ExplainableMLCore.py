import logging
import os
import sys
import warnings

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                              StackingRegressor)
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from config import CONFIG

os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"], mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
warnings.filterwarnings('ignore')

class ExplainableMLCore:
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.feature_importance = None

    def train(self, X_df, y_series):
        logging.info("=" * 80)
        logging.info("PHASE 4: MACHINE LEARNING (EXPLAINABLE)")
        logging.info("=" * 80)

        X = X_df.fillna(0)
        y = y_series.fillna(0)

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        estimators = [
            ('rf', RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)),
            ('xgb', xgb.XGBRegressor(n_estimators=150, max_depth=7, learning_rate=0.03)),
            ('gbm', GradientBoostingRegressor(n_estimators=150, max_depth=5))
        ]

        self.model = StackingRegressor(
            estimators=estimators,
            final_estimator=xgb.XGBRegressor(n_estimators=50, max_depth=4),
            cv=5
        )

        logging.info("  Training stacking ensemble (5-fold CV)...")
        self.model.fit(X_train, y_train)

        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        mae = mean_absolute_error(y_test, self.model.predict(X_test))

        logging.info(f"   Training R²: {train_score:.4f}")
        logging.info(f"   Testing R²:  {test_score:.4f}")
        logging.info(f"   MAE:         {mae:.4f}")

        self._extract_feature_importance(X_test, y_test, X_df.columns)

        return self.model

    def _extract_feature_importance(self, X_test, y_test, feature_names):
        logging.info("  Computing permutation feature importance...")

        result = permutation_importance(
            self.model, X_test, y_test, n_repeats=10, random_state=42
        )

        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': result.importances_mean
        }).sort_values('importance', ascending=False)

        logging.info("\n  Top 3 Most Important Features:")
        for idx, row in self.feature_importance.head(3).iterrows():
            logging.info(f"    {row['feature']:20s} | {row['importance']:.4f}")