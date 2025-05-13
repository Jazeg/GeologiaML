#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para entrenar modelos predictivos para propiedades geotécnicas.
"""

import os
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """
    Carga los datos procesados.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame con los datos procesados
    """
    # Obtener directorio del proyecto
    project_dir = Path(__file__).resolve().parents[2]
    
    # Ruta del archivo procesado
    processed_path = project_dir / "data" / "processed" / "combined_data.csv"
    
    if processed_path.exists():
        return pd.read_csv(processed_path)
    else:
        print(f"No se encontró el archivo procesado: {processed_path}")
        return None

def prepare_vs_depth_data(df):
    """
    Prepara los datos para predecir Vs en función de la profundidad.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos procesados
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test, feature_names
    """
    # Filtrar solo datos de ensayos MW (que tienen Vs)
    df_mw = df[df['tipo_ensayo'] == 'MW'].copy()
    
    # Verificar que haya datos suficientes
    if len(df_mw) < 10:
        print("Datos insuficientes para modelado")
        return None, None, None, None, None
    
    # Features y target
    X = df_mw[['profundidad_media', 'Norte', 'Este']].copy()
    y = df_mw['Vs (m/s)'].copy()
    
    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

def prepare_nspt_depth_data(df):
    """
    Prepara los datos para predecir NSPT en función de la profundidad.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos procesados
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test, feature_names
    """
    # Filtrar solo datos de ensayos SPT (que tienen NSPT)
    df_spt = df[df['tipo_ensayo'] == 'SPT'].copy()
    
    # Verificar que haya datos suficientes
    if len(df_spt) < 10:
        print("Datos insuficientes para modelado")
        return None, None, None, None, None
    
    # Features y target
    X = df_spt[['profundidad_media', 'Norte', 'Este']].copy()
    y = df_spt['Nspt'].copy()
    
    # Eliminar filas con NaN en target
    valid_indices = ~y.isna()
    X = X[valid_indices]
    y = y[valid_indices]
    
    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

def train_models(X_train, y_train, model_type='vs'):
    """
    Entrena varios modelos y selecciona el mejor.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Features de entrenamiento
    y_train : pandas.Series
        Target de entrenamiento
    model_type : str
        Tipo de modelo ('vs' o 'nspt')
        
    Returns:
    --------
    tuple
        Mejor modelo, resultados de GridSearchCV
    """
    # Definir modelos a probar
    models = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'Linear': LinearRegression(),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'SVR': SVR()
    }
    
    # Parámetros para cada modelo
    params = {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'GradientBoosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'Linear': {},
        'Ridge': {
            'alpha': [0.1, 1.0, 10.0]
        },
        'Lasso': {
            'alpha': [0.1, 1.0, 10.0]
        },
        'SVR': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
    }
    
    # Resultados para cada modelo
    results = {}
    
    # Entrenar cada modelo
    for name, model in models.items():
        print(f"Entrenando {name}...")
        
        # Crear pipeline con escalado
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Ajustar parámetros de GridSearchCV según el modelo
        param_grid = {}
        for param, values in params[name].items():
            param_grid[f'model__{param}'] = values
        
        # Si no hay parámetros, usar un diccionario vacío
        if not param_grid:
            grid_search = pipeline.fit(X_train, y_train)
            best_model = pipeline
            best_params = {}
        else:
            # Realizar búsqueda de hiperparámetros
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Obtener mejor modelo
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        
        # Guardar resultados
        results[name] = {
            'model': best_model,
            'params': best_params,
            'train_score': best_model.score(X_train, y_train)
        }
        
        print(f"  Score: {results[name]['train_score']:.4f}")
    
    # Encontrar el mejor modelo basado en score de entrenamiento
    best_name = max(results, key=lambda k: results[k]['train_score'])
    best_model = results[best_name]['model']
    
    print(f"\nMejor modelo: {best_name} (score: {results[best_name]['train_score']:.4f})")
    
    return best_model, results

def evaluate_model(model, X_test, y_test, feature_names, model_type='vs'):
    """
    Evalúa el modelo en el conjunto de prueba.
    
    Parameters:
    -----------
    model : object
        Modelo entrenado
    X_test : pandas.DataFrame
        Features de prueba
    y_test : pandas.Series
        Target de prueba
    feature_names : list
        Nombres de las features
    model_type : str
        Tipo de modelo ('vs' o 'nspt')
        
    Returns:
    --------
    dict
        Métricas de evaluación
    """
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Crear directorio para figuras
    project_dir = Path(__file__).resolve().parents[2]
    figures_dir = project_dir / "models" / "evaluation"
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # Graficar predicciones vs. valores reales
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title(f'Predicción vs. Real ({model_type.upper()})')
    
    # Graficar residuos
    plt.subplot(1, 2, 2)
    residuos = y_test - y_pred
    plt.scatter(y_pred, residuos)
    plt.axhline(y=0, color='k', linestyle='--', lw=2)
    plt.xlabel('Predicciones')
    plt.ylabel('Residuos')
    plt.title('Residuos')
    
    plt.tight_layout()
    plt.savefig(figures_dir / f'evaluacion_{model_type}.png', dpi=300)
    plt.close()
    
    # Si el modelo tiene feature_importances_
    try:
        # Intentar obtener feature importances
        if hasattr(model, 'named_steps') and hasattr(model.named_steps['model'], 'feature_importances_'):
            importances = model.named_steps['model'].feature_importances_
            
            # Crear gráfico de importancia de features
            plt.figure(figsize=(10, 6))
            plt.bar(feature_names, importances)
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Feature')
            plt.ylabel('Importancia')
            plt.title('Importancia de Features')
            plt.tight_layout()
            plt.savefig(figures_dir / f'feature_importance_{model_type}.png', dpi=300)
            plt.close()
    except:
        print("No se pudo generar gráfico de importancia de features")
    
    # Devolver métricas
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def save_model(model, model_type, metrics, feature_names):
    """
    Guarda el modelo entrenado y sus métricas.
    
    Parameters:
    -----------
    model : object
        Modelo entrenado
    model_type : str
        Tipo de modelo ('vs' o 'nspt')
    metrics : dict
        Métricas de evaluación
    feature_names : list
        Nombres de las features
    """
    # Crear directorio para modelos
    project_dir = Path(__file__).resolve().parents[2]
    models_dir = project_dir / "models" / "saved"
    models_dir.mkdir(exist_ok=True, parents=True)
    
    # Guardar modelo
    model_path = models_dir / f"modelo_{model_type}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Guardar métricas
    metrics_path = models_dir / f"metricas_{model_type}.json"
    
    # Añadir nombres de features a las métricas
    metrics['features'] = feature_names
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    print(f"Modelo y métricas guardados en: {models_dir}")

def main():
    """Función principal"""
    # Cargar datos
    print("Cargando datos procesados...")
    df = load_data()
    
    if df is None:
        return
    
    # Entrenar modelo para Vs
    print("\n--- Entrenando modelo para Velocidad S (Vs) ---")
    X_train_vs, X_test_vs, y_train_vs, y_test_vs, features_vs = prepare_vs_depth_data(df)
    
    if X_train_vs is not None:
        # Entrenar modelo
        best_model_vs, _ = train_models(X_train_vs, y_train_vs, 'vs')
        
        # Evaluar modelo
        metrics_vs = evaluate_model(best_model_vs, X_test_vs, y_test_vs, features_vs, 'vs')
        print(f"Métricas de evaluación (Vs):")
        print(f"  RMSE: {metrics_vs['rmse']:.2f} m/s")
        print(f"  R²: {metrics_vs['r2']:.4f}")
        
        # Guardar modelo
        save_model(best_model_vs, 'vs', metrics_vs, features_vs)
    
    # Entrenar modelo para NSPT
    print("\n--- Entrenando modelo para NSPT ---")
    X_train_nspt, X_test_nspt, y_train_nspt, y_test_nspt, features_nspt = prepare_nspt_depth_data(df)
    
    if X_train_nspt is not None:
        # Entrenar modelo
        best_model_nspt, _ = train_models(X_train_nspt, y_train_nspt, 'nspt')
        
        # Evaluar modelo
        metrics_nspt = evaluate_model(best_model_nspt, X_test_nspt, y_test_nspt, features_nspt, 'nspt')
        print(f"Métricas de evaluación (NSPT):")
        print(f"  RMSE: {metrics_nspt['rmse']:.2f}")
        print(f"  R²: {metrics_nspt['r2']:.4f}")
        
        # Guardar modelo
        save_model(best_model_nspt, 'nspt', metrics_nspt, features_nspt)
    
    print("\nEntrenamiento completado con éxito!")

if __name__ == "__main__":
    main()