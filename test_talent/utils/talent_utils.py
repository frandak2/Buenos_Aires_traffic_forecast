
from sklearn.pipeline import Pipeline
from joblib import dump
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import test_talent.utils.paths as path


def check_quality(df):
    """Check the quality from data, count the NA, DUPLICATED and unique.

    Args:
        df (DataFrame): DataFrame to check
    """
    # Verificamos la cantidad de valores NA en cada columna
    print("Valores NA por columna:")
    print(df.isna().sum())

    # Verificamos la cantidad de datos duplicados en el dataframe
    print("\nCantidad de datos duplicados:")
    print(df.duplicated().sum())

    # Verificamos la cantidad de valores únicos en cada columna
    print("\nValores únicos por columna:")
    print(df.nunique())


def replace_date(x):
    if x.find('/21')>0:
        x = x.replace("/21", "/2021")
    elif x.find('/22')>0:
        x = x.replace("/22", "/2022")
    else:
        x
    return x

def fixed_coord(x):
    if x.find(',')>0:
        x = x.replace(",", ".")
    else:
        x
    return x

def rm_outliers(df, col):
    p_05 = df[col].quantile(0.05) # 5th quantile
    p_95 = df[col].quantile(0.95) # 95th quantile
    df[col].clip(p_05, p_95, inplace=True)
    return df


def update_model(model: Pipeline) -> None:
    dump(model, path.models_dir('model.pkl'))


def save_simple_metrics_report(train_score: float, test_score: float, validation_score: float, mse: float, mae: float, mape: float, model: Pipeline) -> None:
    with open(path.reports_dir('report.txt'), 'w') as report_file:

        report_file.write('# Model Pipeline Description'+'\n')

        for key, value in model.named_steps.items():
            report_file.write(f'### {key}:{value.__repr__()}'+'\n')

        report_file.write(f'### Train Score: {train_score}'+'\n')
        report_file.write(f'### Test Score: {test_score}'+'\n')
        report_file.write(f'### Validation Score: {validation_score}'+'\n')
        report_file.write(f'### Mean Squared Error: {mse}'+'\n')
        report_file.write(f'### Mean Absolute Error: {mae}'+'\n')
        report_file.write(f'### Mean Absolute Percentage Error: {mape}'+'\n')

def get_model_performance_test_set(y_real: pd.Series, y_pred: pd.Series) ->None:
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    sns.regplot(x=y_pred, y=y_real, ax = ax)
    ax.set_xlabel('Predicted worldwide gross')
    ax.set_ylabel('Real worldwide gross')
    ax.set_title('Behavior of model prediction')
    fig.savefig(path.reports_figures_dir('prediction_behavior.png'))