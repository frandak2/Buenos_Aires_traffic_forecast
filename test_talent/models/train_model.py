# Importar las bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import logging
import logging.config
import warnings

import test_talent.utils.paths as path
from mlops_project.utils.logger import get_logging_config
from test_talent.utils.talent_utils import update_model, save_simple_metrics_report, get_model_performance_test_set

# Setting up logging configuration
logging.config.dictConfig(get_logging_config())

# Ignoring warnings
warnings.filterwarnings("ignore")

logging.info('Loading Data...')
# Cargar el dataset
df = pd.read_csv(path.data_processed_dir('data_clean.csv'))

# Crear un ColumnTransformer para hacer un OneHotEncoder en la columna "Nombre_Autopista"
numeric_features = ['Hora','dia','mes','anio']
numeric_transformer = StandardScaler()

categorical_features = ["Auto_Nombre"]
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder= 'passthrough'
)

X = df.drop(['Cant_Veh','Fecha'],axis=1)
y = df['Cant_Veh']
logging.info('Seraparating dataset into train and test')
# Train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)
# Crear una pipeline que incluya el ColumnTransformer y el XGBRegressor
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', XGBRegressor(random_state=42))])

# Definir los parametros para el GridSearchCV
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [3, 4, 5],
    'model__learning_rate': [0.1, 0.05, 0.01]
}

# Realizar el GridSearchCV
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X, y)

logging.info('Cross validating with best model...')
# realizar validacion cruzada
final_result = cross_validate(grid.best_estimator_, X_train, y_train, return_train_score=True, cv=5)
train_score = np.mean(final_result['train_score'])
test_score = np.mean(final_result['test_score'])
assert train_score > 0.7
assert test_score > 0.65

# Mostrar los resultados
print("Mejor puntaje: ", grid.best_score_)
print("Mejores parametros: ", grid.best_params_)

validation_score = grid.best_estimator_.score(X_test, y_test)

y_pred = grid.best_estimator_.predict(X_test)

logging.info('Generating model report...')
# Cálculo de las métricas
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

save_simple_metrics_report(train_score, test_score, validation_score,mse, mae, r2, grid.best_estimator_)
get_model_performance_test_set(y_test, y_pred)
update_model(grid.best_estimator_)

logging.info('Training Finished')