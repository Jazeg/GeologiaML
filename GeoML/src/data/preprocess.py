#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para el preprocesamiento de datos geotécnicos.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_combined_data():
    """
    Carga los datos combinados.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame con los datos combinados
    """
    # Obtener directorio del proyecto
    project_dir = Path(__file__).resolve().parents[2]
    
    # Ruta del archivo combinado
    combined_path = project_dir / "data" / "processed" / "combined_data.csv"
    
    if combined_path.exists():
        return pd.read_csv(combined_path)
    else:
        print(f"No se encontró el archivo: {combined_path}")
        return None

def preprocess_for_nspt_prediction(df):
    """
    Preprocesa los datos para la predicción de NSPT.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos combinados
        
    Returns:
    --------
    tuple
        X, y, preprocessor
    """
    # Verificar si hay datos
    if df is None or len(df) == 0:
        return None, None, None
    
    # Hacer una copia para no modificar el original
    df_copy = df.copy()
    
    # Verificar y crear columnas necesarias
    if 'profundidad_media' not in df_copy.columns and 'De' in df_copy.columns and 'Hasta' in df_copy.columns:
        df_copy['profundidad_media'] = (df_copy['De'] + df_copy['Hasta']) / 2
    
    # Identificar tipo de ensayo a partir del ítem si no existe
    if 'tipo_ensayo' not in df_copy.columns and '�tem' in df_copy.columns:
        df_copy['tipo_ensayo'] = df_copy['�tem'].str.split('-').str[0]
    
    # Columnas numéricas y categóricas
    numeric_features = [
        'profundidad_media', 'Vs (m/s)', 
        'Gravas', 'Arenas', 'Finos',
        'LL %', 'LP', 'IP %', 'W%'
    ]
    
    # Filtrar solo las columnas numéricas que existen en el DataFrame
    numeric_features = [col for col in numeric_features if col in df_copy.columns]
    
    # Verificar columnas categóricas
    categorical_features = ['SUCS', 'tipo_ensayo']
    categorical_features = [col for col in categorical_features if col in df_copy.columns]
    
    # Verificar si tenemos las columnas mínimas necesarias
    if 'profundidad_media' not in numeric_features or len(numeric_features) < 3:
        print("Datos insuficientes para preprocesamiento. Se requieren al menos profundidad_media y otras variables.")
        return None, None, None
    
    # Definir la variable objetivo
    target = 'Nspt'
    
    # Verificar si existe la variable objetivo
    if target not in df_copy.columns:
        print(f"La columna {target} no existe en el DataFrame")
        return None, None, None
    
    # Eliminar filas con valores nulos en la variable objetivo
    df_copy = df_copy.dropna(subset=[target])
    
    # Verificar si quedan suficientes datos
    if len(df_copy) < 10:
        print("Datos insuficientes después de eliminar valores nulos")
        return None, None, None
    
    # Definir transformaciones para columnas numéricas y categóricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combinar transformaciones
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features) if categorical_features else None
        ],
        remainder='drop'
    )
    
    # Eliminar transformadores None
    preprocessor.transformers = [t for t in preprocessor.transformers if t[1] is not None]
    
    # Preparar X e y
    X = df_copy[numeric_features + categorical_features]
    y = df_copy[target]
    
    return X, y, preprocessor

def preprocess_for_qadm_prediction(df):
    """
    Preprocesa los datos para la predicción de Qadm (Capacidad portante admisible).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos combinados
        
    Returns:
    --------
    tuple
        X, y, preprocessor
    """
    # Verificar si hay datos
    if df is None or len(df) == 0:
        return None, None, None
    
    # Hacer una copia para no modificar el original
    df_copy = df.copy()
    
    # Verificar y crear columnas necesarias
    if 'profundidad_media' not in df_copy.columns and 'De' in df_copy.columns and 'Hasta' in df_copy.columns:
        df_copy['profundidad_media'] = (df_copy['De'] + df_copy['Hasta']) / 2
    
    # Identificar tipo de ensayo a partir del ítem si no existe
    if 'tipo_ensayo' not in df_copy.columns and '�tem' in df_copy.columns:
        df_copy['tipo_ensayo'] = df_copy['�tem'].str.split('-').str[0]
    
    # Columnas numéricas y categóricas
    numeric_features = [
        'profundidad_media', 'Vs (m/s)', 'Nspt', '(N1)60',
        'Gravas', 'Arenas', 'Finos',
        'LL %', 'LP', 'IP %', 'W%'
    ]
    
    # Filtrar solo las columnas numéricas que existen en el DataFrame
    numeric_features = [col for col in numeric_features if col in df_copy.columns]
    
    # Verificar columnas categóricas
    categorical_features = ['SUCS', 'tipo_ensayo']
    categorical_features = [col for col in categorical_features if col in df_copy.columns]
    
    # Verificar si tenemos las columnas mínimas necesarias
    if 'profundidad_media' not in numeric_features or 'Nspt' not in numeric_features or len(numeric_features) < 3:
        print("Datos insuficientes para preprocesamiento. Se requieren al menos profundidad_media, Nspt y otras variables.")
        return None, None, None
    
    # Definir la variable objetivo
    target = 'Qadm. (kg/cm2)'
    
    # Verificar si existe la variable objetivo
    if target not in df_copy.columns:
        print(f"La columna {target} no existe en el DataFrame")
        return None, None, None
    
    # Eliminar filas con valores nulos en la variable objetivo
    df_copy = df_copy.dropna(subset=[target])
    
    # Verificar si quedan suficientes datos
    if len(df_copy) < 10:
        print("Datos insuficientes después de eliminar valores nulos")
        return None, None, None
    
    # Definir transformaciones para columnas numéricas y categóricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combinar transformaciones
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features) if categorical_features else None
        ],
        remainder='drop'
    )
    
    # Eliminar transformadores None
    preprocessor.transformers = [t for t in preprocessor.transformers if t[1] is not None]
    
    # Preparar X e y
    X = df_copy[numeric_features + categorical_features]
    y = df_copy[target]
    
    return X, y, preprocessor

def main():
    """Función principal."""
    # Cargar datos combinados
    print("Cargando datos combinados...")
    df = load_combined_data()
    
    if df is not None:
        print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Preprocesar para predicción de NSPT
        print("\nPreprocesando datos para predicción de NSPT...")
        X_nspt, y_nspt, preprocessor_nspt = preprocess_for_nspt_prediction(df)
        
        if X_nspt is not None:
            print(f"Datos listos para modelo NSPT: {X_nspt.shape[0]} filas, {X_nspt.shape[1]} columnas")
            print(f"Columnas utilizadas: {X_nspt.columns.tolist()}")
        
        # Preprocesar para predicción de Qadm
        print("\nPreprocesando datos para predicción de Qadm...")
        X_qadm, y_qadm, preprocessor_qadm = preprocess_for_qadm_prediction(df)
        
        if X_qadm is not None:
            print(f"Datos listos para modelo Qadm: {X_qadm.shape[0]} filas, {X_qadm.shape[1]} columnas")
            print(f"Columnas utilizadas: {X_qadm.columns.tolist()}")
    else:
        print("No se pudieron cargar los datos.")

if __name__ == "__main__":
    main()