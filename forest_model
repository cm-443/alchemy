from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import ta
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from joblib import dump
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class ForexClassBalancer:
    def __init__(self, lookback_window=1000):
        self.lookback_window = lookback_window
        self.scaler = StandardScaler()

    def calculate_dynamic_thresholds(self, price_changes):
        """Calculate dynamic thresholds based on rolling volatility"""
        rolling_std = pd.Series(price_changes).rolling(
            window=self.lookback_window,
            min_periods=100
        ).std()

        # Use 0.5 standard deviations as threshold
        up_threshold = rolling_std * 0.5
        down_threshold = -rolling_std * 0.5

        return up_threshold, down_threshold

    def categorize_movement(self, price_changes):
        """Categorize price movements using dynamic thresholds"""
        up_thresh, down_thresh = self.calculate_dynamic_thresholds(price_changes)

        categories = np.zeros(len(price_changes))
        categories[price_changes > up_thresh] = 2  # Up
        categories[price_changes < down_thresh] = 0  # Down
        categories[(price_changes >= down_thresh) & (price_changes <= up_thresh)] = 1  # Neutral

        return categories.astype(int)

    def prepare_aligned_data(self, forex_data, feature_columns):
        """Prepare and align feature and target data"""
        # Calculate price changes
        forex_data['price_change_pct'] = forex_data['close'].pct_change()

        # Generate categories with dynamic thresholds
        categories = self.categorize_movement(forex_data['price_change_pct'])

        # Prepare features DataFrame
        X = forex_data[feature_columns].copy()

        # Ensure X and y are the same length
        min_length = min(len(X), len(categories))
        X = X.iloc[:min_length]
        y = categories[:min_length]

        # Remove any NaN values
        mask = ~(X.isna().any(axis=1))
        X = X[mask]
        y = y[mask]

        return X, y


def load_master_list_from_file(filename='5s_list.py'):
    with open(filename, 'r') as file:
        content = file.read()
    master_list = ast.literal_eval(content.split(' = ')[1])
    return master_list


def load_and_prepare_data(master_list):
    forex_data = []
    for file in master_list:
        data = pd.read_csv(file)
        data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H:%M:%S')
        forex_data.append(data)

    forex_data = pd.concat(forex_data, ignore_index=True)
    forex_data.sort_values('timestamp', inplace=True)
    forex_data.reset_index(drop=True, inplace=True)
    return forex_data


def add_technical_indicators(df):
    # Technical indicators
    df['EMA_9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['EMA_21'] = ta.trend.ema_indicator(df['close'], window=21)
    df['RSI_5'] = ta.momentum.RSIIndicator(df['close'], window=5).rsi()

    macd_indicator = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd_indicator.macd()
    df['MACD_signal'] = macd_indicator.macd_signal()
    df['MACD_diff'] = macd_indicator.macd_diff()

    bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['BB_high'] = bb_indicator.bollinger_hband()
    df['BB_low'] = bb_indicator.bollinger_lband()
    df['BB_mid'] = bb_indicator.bollinger_mavg()

    stoch_indicator = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'],
                                                       window=14, smooth_window=3)
    df['Stoch_k'] = stoch_indicator.stoch()
    df['Stoch_d'] = stoch_indicator.stoch_signal()

    # Advanced feature engineering
    df['EMA_Crossover'] = (df['EMA_9'] > df['EMA_21']).astype(int)
    df['EMA_Crossover_Change'] = df['EMA_Crossover'].diff()

    df['MACD_Crossover'] = (df['MACD'] > df['MACD_signal']).astype(int)
    df['MACD_Crossover_Change'] = df['MACD_Crossover'].diff()

    df['BB_width'] = df['BB_high'] - df['BB_low']
    df['BB_Tight'] = (df['BB_width'] < df['BB_width'].rolling(window=20).mean()).astype(int)

    df['Breakout_Buy_Signal'] = ((df['BB_Tight'] == 1) &
                                 (df['MACD_Crossover_Change'] == 1)).astype(int)
    df['Breakout_Sell_Signal'] = ((df['BB_Tight'] == 1) &
                                  (df['MACD_Crossover_Change'] == -1)).astype(int)

    # Additional features
    df['RSI_Trend'] = (df['RSI_5'].diff() > 0).astype(int)
    df['Price_Above_BB_Mid'] = (df['close'] > df['BB_mid']).astype(int)
    df['Stoch_Crossover'] = (df['Stoch_k'] > df['Stoch_d']).astype(int)

    df.dropna(inplace=True)
    return df



def balance_classes(X, y):
    """Apply hybrid balancing approach with adjusted ratios"""
    print("\nOriginal class distribution:", Counter(y))

    # First remove Tomek links
    tl = TomekLinks(sampling_strategy='majority')
    X_clean, y_clean = tl.fit_resample(X, y)
    print("Class distribution after Tomek:", Counter(y_clean))

    # Calculate target numbers for SMOTE
    class_counts = Counter(y_clean)
    majority_class = max(class_counts.items(), key=lambda x: x[1])[0]
    majority_count = class_counts[majority_class]

    # Target about 50% of majority class size for minority classes
    target_count = int(majority_count * 0.5)

    # Create sampling strategy dict
    sampling_strategy = {}
    for class_label, count in class_counts.items():
        if class_label != majority_class and count < target_count:
            sampling_strategy[class_label] = target_count

    if sampling_strategy:
        smote_tomek = SMOTETomek(
            sampling_strategy=sampling_strategy,
            random_state=42
        )
        X_balanced, y_balanced = smote_tomek.fit_resample(X_clean, y_clean)
        print("Final class distribution:", Counter(y_balanced))
    else:
        X_balanced, y_balanced = X_clean, y_clean
        print("No SMOTE needed, using cleaned distribution:", Counter(y_balanced))

    return X_balanced, y_balanced


# Add these imports at the top of the file
import matplotlib

matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true, y_pred, fold=None):
    """Plot and save confusion matrix"""
    plt.close('all')  # Close any existing figures

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Down', 'Neutral', 'Up'])
    disp.plot(ax=ax, cmap='Blues')

    title = 'Confusion Matrix - Random Forest with Dynamic Thresholds'
    if fold is not None:
        title += f' (Fold {fold})'

    plt.title(title)
    plt.tight_layout()

    filename = f'Confusion_Matrix_Fold_{fold}.png' if fold else 'Confusion_Matrix_Final.png'
    plt.savefig(filename)
    plt.close(fig)


def plot_feature_importance(model, feature_columns, fold=None):
    """Plot and save feature importance"""
    plt.close('all')  # Close any existing figures

    importances = model.named_steps['rf'].feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)),
               [feature_columns[i] for i in indices],
               rotation=90)
    plt.title('Feature Importances - Random Forest')
    plt.tight_layout()

    filename = f'Feature_Importance_Fold_{fold}.png' if fold else 'Feature_Importance_Final.png'
    plt.savefig(filename)
    plt.close(fig)


def evaluate_with_time_series_cv(X, y, n_splits=5):
    """Evaluate model using time series cross-validation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Define more conservative parameter space
    search_spaces = {
        'rf__n_estimators': Integer(300, 800),
        'rf__max_depth': Integer(5, 20),
        'rf__min_samples_split': Integer(5, 15),
        'rf__min_samples_leaf': Integer(3, 8),
        'rf__max_features': Categorical(['sqrt', 'log2']),
        'rf__bootstrap': Categorical([True])
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        ))
    ])

    fold_scores = []
    train_scores = []
    best_models = []

    print("\nPerforming Time Series Cross-Validation:")
    print("=======================================")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"\nFold {fold}/{n_splits}")

        # Split data
        X_fold_train = X.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_train = y.iloc[train_idx]
        y_fold_val = y.iloc[val_idx]

        # Balance training data
        X_fold_train_balanced, y_fold_train_balanced = balance_classes(X_fold_train, y_fold_train)

        # Optimize model for this fold
        opt = BayesSearchCV(
            estimator=pipeline,
            search_spaces=search_spaces,
            n_iter=20,
            cv=3,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0
        )

        # Fit and evaluate
        opt.fit(X_fold_train_balanced, y_fold_train_balanced)

        # Get scores
        train_score = opt.score(X_fold_train_balanced, y_fold_train_balanced)
        val_score = opt.score(X_fold_val, y_fold_val)

        train_scores.append(train_score)
        fold_scores.append(val_score)
        best_models.append(opt.best_estimator_)

        # Predictions for this fold
        y_val_pred = opt.predict(X_fold_val)

        print(f"Training Score: {train_score:.4f}")
        print(f"Validation Score: {val_score:.4f}")
        print("\nValidation Set Classification Report:")
        print(classification_report(y_fold_val, y_val_pred,
                                    target_names=['Down', 'Neutral', 'Up']))

        try:
            # Plot confusion matrix and feature importance for this fold
            plot_confusion_matrix(y_fold_val, y_val_pred, fold)
            plot_feature_importance(opt.best_estimator_, X.columns, fold)
        except Exception as e:
            print(f"Warning: Could not create plots for fold {fold}: {str(e)}")

    # Calculate overall metrics
    print("\nOverall Results:")
    print(f"Average Training Score: {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}")
    print(f"Average Validation Score: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    # Check for overfitting
    overfit_gap = np.mean(train_scores) - np.mean(fold_scores)
    print(f"Overfitting Gap: {overfit_gap:.4f}")

    return best_models, fold_scores



def main():
    # Load and prepare data
    master_list = load_master_list_from_file()
    master_list = ['csv/' + filename for filename in master_list]
    forex_data = load_and_prepare_data(master_list)
    forex_data = add_technical_indicators(forex_data)

    feature_columns = ['EMA_9', 'EMA_21', 'RSI_5', 'MACD', 'MACD_signal', 'MACD_diff',
                       'BB_high', 'BB_low', 'BB_mid', 'Stoch_k', 'Stoch_d',
                       'BB_width', 'EMA_Crossover', 'MACD_Crossover',
                       'EMA_Crossover_Change', 'MACD_Crossover_Change',
                       'BB_Tight', 'Breakout_Buy_Signal', 'Breakout_Sell_Signal',
                       'RSI_Trend', 'Price_Above_BB_Mid', 'Stoch_Crossover']

    balancer = ForexClassBalancer(lookback_window=1000)
    try:
        # Prepare data
        X, y = balancer.prepare_aligned_data(forex_data, feature_columns)
        X = pd.DataFrame(X, columns=feature_columns)
        y = pd.Series(y)

        print("Data shapes after alignment:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {len(y)}")
        print("\nInitial class distribution:", Counter(y))

        # Evaluate model with time series cross-validation
        best_models, fold_scores = evaluate_with_time_series_cv(X, y, n_splits=5)

        # Save the best model (using the last fold's model)
        dump(best_models[-1], 'randomforest_best_model_with_scaler.joblib')

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



# C:\Users\alex_\PycharmProjects\new_pipeline\venv\Scripts\python.exe C:\Users\alex_\PycharmProjects\One_month_pipeline\forest_v2.1.py
# Data shapes after alignment:
# X shape: (316145, 22)
# y shape: 316145
#
# Initial class distribution: Counter({1: 167919, 2: 74267, 0: 73959})
#
# Performing Time Series Cross-Validation:
# =======================================
#
# Fold 1/5
#
# Original class distribution: Counter({1: 27550, 2: 12590, 0: 12555})
# Class distribution after Tomek: Counter({1: 23009, 2: 12590, 0: 12555})
# No SMOTE needed, using cleaned distribution: Counter({1: 23009, 2: 12590, 0: 12555})
# Training Score: 0.9216
# Validation Score: 0.7332
#
# Validation Set Classification Report:
#               precision    recall  f1-score   support
#
#         Down       0.59      0.82      0.69     11984
#      Neutral       0.86      0.62      0.72     28762
#           Up       0.70      0.90      0.79     11944
#
#     accuracy                           0.73     52690
#    macro avg       0.72      0.78      0.73     52690
# weighted avg       0.76      0.73      0.73     52690
#
#
# Fold 2/5
#
# Original class distribution: Counter({1: 56312, 0: 24539, 2: 24534})
# Class distribution after Tomek: Counter({1: 47427, 0: 24539, 2: 24534})
# No SMOTE needed, using cleaned distribution: Counter({1: 47427, 0: 24539, 2: 24534})
# Training Score: 0.9055
# Validation Score: 0.7577
#
# Validation Set Classification Report:
#               precision    recall  f1-score   support
#
#         Down       0.67      0.72      0.70     12023
#      Neutral       0.82      0.71      0.76     28485
#           Up       0.73      0.92      0.81     12182
#
#     accuracy                           0.76     52690
#    macro avg       0.74      0.78      0.76     52690
# weighted avg       0.77      0.76      0.76     52690
#
#
# Fold 3/5
#
# Original class distribution: Counter({1: 84797, 2: 36716, 0: 36562})
# Class distribution after Tomek: Counter({1: 71676, 2: 36716, 0: 36562})
# No SMOTE needed, using cleaned distribution: Counter({1: 71676, 2: 36716, 0: 36562})
# Training Score: 0.8953
# Validation Score: 0.7431
#
# Validation Set Classification Report:
#               precision    recall  f1-score   support
#
#         Down       0.62      0.85      0.71     12627
#      Neutral       0.87      0.58      0.69     27448
#           Up       0.71      0.97      0.82     12615
#
#     accuracy                           0.74     52690
#    macro avg       0.73      0.80      0.74     52690
# weighted avg       0.77      0.74      0.73     52690
#
#
# Fold 4/5
#
# Original class distribution: Counter({1: 112245, 2: 49331, 0: 49189})
# Class distribution after Tomek: Counter({1: 94734, 2: 49331, 0: 49189})
# No SMOTE needed, using cleaned distribution: Counter({1: 94734, 2: 49331, 0: 49189})
# Training Score: 0.8898
# Validation Score: 0.7290
#
# Validation Set Classification Report:
#               precision    recall  f1-score   support
#
#         Down       0.61      0.85      0.71     12135
#      Neutral       0.88      0.56      0.69     28365
#           Up       0.67      0.97      0.79     12190
#
#     accuracy                           0.72     52690
#    macro avg       0.72      0.79      0.73     52690
# weighted avg       0.77      0.72      0.72     52690
#
#
# Fold 5/5
#
# Original class distribution: Counter({1: 140610, 2: 61521, 0: 61324})
# Class distribution after Tomek: Counter({1: 118656, 2: 61521, 0: 61324})
# No SMOTE needed, using cleaned distribution: Counter({1: 118656, 2: 61521, 0: 61324})
# Training Score: 0.8829
# Validation Score: 0.7469
#
# Validation Set Classification Report:
#               precision    recall  f1-score   support
#
#         Down       0.65      0.83      0.73     12635
#      Neutral       0.85      0.60      0.71     27309
#           Up       0.70      0.95      0.81     12746
#
#     accuracy                           0.74     52690
#    macro avg       0.73      0.79      0.75     52690
# weighted avg       0.77      0.74      0.74     52690
#
#
# Overall Results:
# Average Training Score: 0.8990 ± 0.0135
# Average Validation Score: 0.7420 ± 0.0102
# Overfitting Gap: 0.1570
#
# Process finished with exit code 0
