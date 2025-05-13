#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para realizar predicciones con los modelos entrenados.
"""

import os
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_model_and_preprocessor(model_type):
    """
    Carga un modelo entrenado y su preprocesador.
    
    Parameters:
    -----------
    model_type : str
        Tipo de modelo ('nspt' o 'qadm')
        
    Returns:
    --------
    tuple
        Modelo entrenado, preprocesador, métricas, nombres de features
    """
    # Obtener directorio del proyecto
    project_dir = Path(__file__).resolve().parents[2]
    
    # Rutas de archivos
    model_path = project_dir / "models" / "saved" / f"modelo_{model_type}.pkl"
    preprocessor_path = project_dir / "models" / "saved" / f"preprocessor_{model_type}.pkl"
    metrics_path = project_dir / "models" / "saved" / f"metricas_{model_type}.json"
    
    # Verificar si existen los archivos
    if not model_path.exists():
        print(f"No se encontró el modelo para {model_type} en {model_path}")
        return None, None, None, None
    
    # Cargar modelo
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Cargar preprocesador (si existe)
    preprocessor = None
    if preprocessor_path.exists():
        try:
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
        except Exception as e:
            print(f"Error al cargar el preprocesador: {e}")
    
    # Cargar métricas
    metrics = None
    features = None
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                features = metrics.get('features', [])
        except Exception as e:
            print(f"Error al cargar las métricas: {e}")
    
    return model, preprocessor, metrics, features

def predict_nspt_profile(model, preprocessor, features, coordinates=None, max_depth=50):
    """
    Predice el perfil de NSPT con la profundidad para una ubicación específica.
    
    Parameters:
    -----------
    model : object
        Modelo entrenado para NSPT
    preprocessor : object
        Preprocesador para los datos
    features : list
        Lista de features del modelo
    coordinates : tuple, optional
        Coordenadas (Norte, Este) para la predicción.
        Si es None, se usan valores por defecto
    max_depth : int, optional
        Profundidad máxima para la predicción (m)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con las predicciones
    """
    # Si no se proporcionan coordenadas, usar valores por defecto
    if coordinates is None:
        # Usar coordenadas promedio desde los datos de entrenamiento
        north = 9457265  # Valor ejemplo, debe ajustarse
        east = 535010    # Valor ejemplo, debe ajustarse
    else:
        north, east = coordinates
    
    # Crear rango de profundidades
    depths = np.linspace(0, max_depth, 100)
    
    # Características básicas que deben existir en 'features'
    essential_features = ['profundidad_media']
    if not all(feat in features for feat in essential_features):
        print(f"Faltan características esenciales en el modelo. Se requieren: {essential_features}")
        return None
    
    # Crear DataFrame para predicción
    X_pred = pd.DataFrame()
    
    # Añadir características requeridas
    if 'profundidad_media' in features:
        X_pred['profundidad_media'] = depths
    
    # Añadir coordenadas si están en las características
    if 'Norte' in features:
        X_pred['Norte'] = north
    if 'Este' in features:
        X_pred['Este'] = east
    
    # Añadir otras características comunes como promedio o valor más frecuente
    # Esto debería adaptarse según los features del modelo
    common_features = {
        'Gravas': 20,       # Valor promedio ejemplo
        'Arenas': 50,       # Valor promedio ejemplo
        'Finos': 30,        # Valor promedio ejemplo
        'LL %': 30,         # Valor promedio ejemplo
        'LP': 20,           # Valor promedio ejemplo
        'IP %': 10,         # Valor promedio ejemplo
        'W%': 15,           # Valor promedio ejemplo
        'SUCS': 'SM',       # Valor más frecuente ejemplo
        'tipo_ensayo': 'SPT'  # Valor más frecuente ejemplo
    }
    
    for feat, value in common_features.items():
        if feat in features and feat not in X_pred.columns:
            X_pred[feat] = value
    
    # Asegurarse de que todas las características del modelo están presentes
    for feat in features:
        if feat not in X_pred.columns:
            print(f"Advertencia: Característica '{feat}' no proporcionada, usando valor por defecto")
            X_pred[feat] = 0  # Valor por defecto
    
    # Asegurarse de que el orden de las columnas sea correcto
    X_pred = X_pred[features]
    
    # Aplicar preprocesador si existe
    if preprocessor is not None:
        try:
            X_pred_transformed = preprocessor.transform(X_pred)
        except Exception as e:
            print(f"Error al aplicar preprocesador: {e}")
            return None
    else:
        X_pred_transformed = X_pred
    
    # Realizar predicción
    try:
        nspt_pred = model.predict(X_pred_transformed)
    except Exception as e:
        print(f"Error al realizar predicción: {e}")
        return None
    
    # Crear DataFrame con resultados
    results = pd.DataFrame()
    results['profundidad'] = depths
    results['NSPT_prediccion'] = nspt_pred
    
    return results

def predict_qadm_profile(model, preprocessor, features, coordinates=None, max_depth=50):
    """
    Predice el perfil de capacidad portante admisible (Qadm) con la profundidad.
    
    Parameters:
    -----------
    model : object
        Modelo entrenado para Qadm
    preprocessor : object
        Preprocesador para los datos
    features : list
        Lista de features del modelo
    coordinates : tuple, optional
        Coordenadas (Norte, Este) para la predicción.
        Si es None, se usan valores por defecto
    max_depth : int, optional
        Profundidad máxima para la predicción (m)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con las predicciones
    """
    # Si no se proporcionan coordenadas, usar valores por defecto
    if coordinates is None:
        # Usar coordenadas promedio desde los datos de entrenamiento
        north = 9457265  # Valor ejemplo, debe ajustarse
        east = 535010    # Valor ejemplo, debe ajustarse
    else:
        north, east = coordinates
    
    # Crear rango de profundidades
    depths = np.linspace(0, max_depth, 100)
    
    # Características básicas que deben existir en 'features'
    essential_features = ['profundidad_media']
    if not all(feat in features for feat in essential_features):
        print(f"Faltan características esenciales en el modelo. Se requieren: {essential_features}")
        return None
    
    # Crear DataFrame para predicción
    X_pred = pd.DataFrame()
    
    # Añadir características requeridas
    if 'profundidad_media' in features:
        X_pred['profundidad_media'] = depths
    
    # Añadir coordenadas si están en las características
    if 'Norte' in features:
        X_pred['Norte'] = north
    if 'Este' in features:
        X_pred['Este'] = east
    
    # Para Qadm necesitamos NSPT
    # Primero, cargar modelo de NSPT y predecir valores
    model_nspt, preprocessor_nspt, _, features_nspt = load_model_and_preprocessor('nspt')
    
    if model_nspt is not None and 'Nspt' in features:
        # Predecir NSPT para las profundidades
        nspt_predictions = predict_nspt_profile(model_nspt, preprocessor_nspt, features_nspt, coordinates, max_depth)
        
        if nspt_predictions is not None:
            # Usar las predicciones de NSPT como entrada para Qadm
            X_pred['Nspt'] = nspt_predictions['NSPT_prediccion'].values
    
    # Añadir otras características comunes como promedio o valor más frecuente
    common_features = {
        'Vs (m/s)': 500,    # Valor promedio ejemplo
        '(N1)60': 30,       # Valor promedio ejemplo
        'Gravas': 20,       # Valor promedio ejemplo
        'Arenas': 50,       # Valor promedio ejemplo
        'Finos': 30,        # Valor promedio ejemplo
        'LL %': 30,         # Valor promedio ejemplo
        'LP': 20,           # Valor promedio ejemplo
        'IP %': 10,         # Valor promedio ejemplo
        'W%': 15,           # Valor promedio ejemplo
        'SUCS': 'SM',       # Valor más frecuente ejemplo
        'tipo_ensayo': 'SPT'  # Valor más frecuente ejemplo
    }
    
    for feat, value in common_features.items():
        if feat in features and feat not in X_pred.columns:
            X_pred[feat] = value
    
    # Asegurarse de que todas las características del modelo están presentes
    for feat in features:
        if feat not in X_pred.columns:
            print(f"Advertencia: Característica '{feat}' no proporcionada, usando valor por defecto")
            X_pred[feat] = 0  # Valor por defecto
    
    # Asegurarse de que el orden de las columnas sea correcto
    X_pred = X_pred[features]
    
    # Aplicar preprocesador si existe
    if preprocessor is not None:
        try:
            X_pred_transformed = preprocessor.transform(X_pred)
        except Exception as e:
            print(f"Error al aplicar preprocesador: {e}")
            return None
    else:
        X_pred_transformed = X_pred
    
    # Realizar predicción
    try:
        qadm_pred = model.predict(X_pred_transformed)
    except Exception as e:
        print(f"Error al realizar predicción: {e}")
        return None
    
    # Crear DataFrame con resultados
    results = pd.DataFrame()
    results['profundidad'] = depths
    results['Qadm_prediccion'] = qadm_pred
    
    return results

def visualize_predictions(nspt_predictions, qadm_predictions, location_name='Ubicación'):
    """
    Visualiza las predicciones de NSPT y Qadm.
    
    Parameters:
    -----------
    nspt_predictions : pandas.DataFrame
        DataFrame con predicciones de NSPT
    qadm_predictions : pandas.DataFrame
        DataFrame con predicciones de Qadm
    location_name : str, optional
        Nombre de la ubicación para el título
    """
    # Crear directorio para resultados
    project_dir = Path(__file__).resolve().parents[2]
    figures_dir = project_dir / "models" / "predictions"
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # Combinar gráficos en una sola figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))
    
    # Gráfico de NSPT vs. Profundidad
    if nspt_predictions is not None:
        ax1.plot(nspt_predictions['NSPT_prediccion'], nspt_predictions['profundidad'], 'b-', linewidth=2)
        ax1.set_xlabel('NSPT (golpes)')
        ax1.set_ylabel('Profundidad (m)')
        ax1.set_title(f'Perfil NSPT vs. Profundidad - {location_name}')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.invert_yaxis()  # Invertir eje Y para que la profundidad aumente hacia abajo
    
    # Gráfico de Qadm vs. Profundidad
    if qadm_predictions is not None:
        ax2.plot(qadm_predictions['Qadm_prediccion'], qadm_predictions['profundidad'], 'r-', linewidth=2)
        ax2.set_xlabel('Qadm (kg/cm²)')
        ax2.set_ylabel('Profundidad (m)')
        ax2.set_title(f'Perfil Qadm vs. Profundidad - {location_name}')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.invert_yaxis()  # Invertir eje Y para que la profundidad aumente hacia abajo
    
    plt.tight_layout()
    
    # Guardar figura
    plt.savefig(figures_dir / f'prediccion_perfiles_{location_name.replace(" ", "_")}.png', dpi=300)
    plt.close()
    
    # También guardar como CSV
    if nspt_predictions is not None:
        nspt_predictions.to_csv(figures_dir / f'nspt_prediccion_{location_name.replace(" ", "_")}.csv', index=False)
    
    if qadm_predictions is not None:
        qadm_predictions.to_csv(figures_dir / f'qadm_prediccion_{location_name.replace(" ", "_")}.csv', index=False)
    
    print(f"Visualizaciones guardadas en: {figures_dir}")

def main():
    """Función principal."""
    # Cargar modelo para NSPT
    print("Cargando modelo para NSPT...")
    model_nspt, preprocessor_nspt, metrics_nspt, features_nspt = load_model_and_preprocessor('nspt')
    
    # Cargar modelo para Qadm
    print("Cargando modelo para Qadm...")
    model_qadm, preprocessor_qadm, metrics_qadm, features_qadm = load_model_and_preprocessor('qadm')
    
    # Verificar si se cargaron los modelos
    if model_nspt is None and model_qadm is None:
        print("No se pudieron cargar los modelos. Por favor, ejecuta primero el script de entrenamiento:")
        print("python -m src.models.train_model")
        return
    
    # Solicitar coordenadas al usuario
    print("\nIngrese coordenadas para la predicción (dejar en blanco para usar valores por defecto)")
    north_input = input("Norte (ej. 9457265): ").strip()
    east_input = input("Este (ej. 535010): ").strip()
    
    # Procesar entrada
    if north_input and east_input:
        try:
            north = float(north_input)
            east = float(east_input)
            coordinates = (north, east)
            location_name = f"Norte {north} Este {east}"
        except ValueError:
            print("Coordenadas inválidas, usando valores por defecto")
            coordinates = None
            location_name = "Ubicación Predeterminada"
    else:
        coordinates = None
        location_name = "Ubicación Predeterminada"
    
    # Solicitar profundidad máxima
    max_depth_input = input("Profundidad máxima en metros (dejar en blanco para 50m): ").strip()
    max_depth = float(max_depth_input) if max_depth_input else 50
    
    # Realizar predicciones
    nspt_predictions = None
    qadm_predictions = None
    
    if model_nspt is not None and features_nspt:
        print("\nGenerando perfil de NSPT...")
        nspt_predictions = predict_nspt_profile(model_nspt, preprocessor_nspt, features_nspt, coordinates, max_depth)
    
    if model_qadm is not None and features_qadm:
        print("Generando perfil de Qadm...")
        qadm_predictions = predict_qadm_profile(model_qadm, preprocessor_qadm, features_qadm, coordinates, max_depth)
    
    # Visualizar predicciones
    if nspt_predictions is not None or qadm_predictions is not None:
        print("Generando visualizaciones...")
        visualize_predictions(nspt_predictions, qadm_predictions, location_name)
        print("\nPredicciones completadas con éxito!")
    else:
        print("No se pudieron generar predicciones.")

if __name__ == "__main__":
    main()