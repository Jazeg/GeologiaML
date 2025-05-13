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

def load_model(model_type):
    """
    Carga un modelo entrenado.
    
    Parameters:
    -----------
    model_type : str
        Tipo de modelo ('vs' o 'nspt')
        
    Returns:
    --------
    tuple
        Modelo entrenado, métricas, nombres de features
    """
    # Obtener directorio del proyecto
    project_dir = Path(__file__).resolve().parents[2]
    
    # Rutas de archivos
    model_path = project_dir / "models" / "saved" / f"modelo_{model_type}.pkl"
    metrics_path = project_dir / "models" / "saved" / f"metricas_{model_type}.json"
    
    # Verificar si existen los archivos
    if not model_path.exists() or not metrics_path.exists():
        print(f"No se encontró el modelo para {model_type}")
        return None, None, None
    
    # Cargar modelo
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Cargar métricas
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Extraer nombres de features
    features = metrics.get('features', [])
    
    return model, metrics, features

def predict_vs_depth_profile(model_vs, features_vs, coordinates=None, max_depth=50):
    """
    Predice el perfil de Vs con la profundidad para una ubicación específica.
    
    Parameters:
    -----------
    model_vs : object
        Modelo entrenado para Vs
    features_vs : list
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
    
    # Crear DataFrame para predicción
    X_pred = pd.DataFrame()
    X_pred['profundidad_media'] = depths
    X_pred['Norte'] = north
    X_pred['Este'] = east
    
    # Asegurarse de que el orden de las columnas sea correcto
    X_pred = X_pred[features_vs]
    
    # Realizar predicción
    vs_pred = model_vs.predict(X_pred)
    
    # Crear DataFrame con resultados
    results = pd.DataFrame()
    results['profundidad'] = depths
    results['Vs_prediccion'] = vs_pred
    
    return results

def predict_nspt_depth_profile(model_nspt, features_nspt, coordinates=None, max_depth=50):
    """
    Predice el perfil de NSPT con la profundidad para una ubicación específica.
    
    Parameters:
    -----------
    model_nspt : object
        Modelo entrenado para NSPT
    features_nspt : list
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
    
    # Crear DataFrame para predicción
    X_pred = pd.DataFrame()
    X_pred['profundidad_media'] = depths
    X_pred['Norte'] = north
    X_pred['Este'] = east
    
    # Asegurarse de que el orden de las columnas sea correcto
    X_pred = X_pred[features_nspt]
    
    # Realizar predicción
    nspt_pred = model_nspt.predict(X_pred)
    
    # Crear DataFrame con resultados
    results = pd.DataFrame()
    results['profundidad'] = depths
    results['NSPT_prediccion'] = nspt_pred
    
    return results

def visualize_predictions(vs_predictions, nspt_predictions, location_name='Ubicación'):
    """
    Visualiza las predicciones de Vs y NSPT.
    
    Parameters:
    -----------
    vs_predictions : pandas.DataFrame
        DataFrame con predicciones de Vs
    nspt_predictions : pandas.DataFrame
        DataFrame con predicciones de NSPT
    location_name : str, optional
        Nombre de la ubicación para el título
    """
    # Crear directorio para resultados
    project_dir = Path(__file__).resolve().parents[2]
    figures_dir = project_dir / "models" / "predictions"
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # Combinar gráficos en una sola figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))
    
    # Gráfico de Vs vs. Profundidad
    if vs_predictions is not None:
        ax1.plot(vs_predictions['Vs_prediccion'], vs_predictions['profundidad'], 'b-', linewidth=2)
        ax1.set_xlabel('Velocidad S (m/s)')
        ax1.set_ylabel('Profundidad (m)')
        ax1.set_title(f'Perfil Vs vs. Profundidad - {location_name}')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.invert_yaxis()  # Invertir eje Y para que la profundidad aumente hacia abajo
    
    # Gráfico de NSPT vs. Profundidad
    if nspt_predictions is not None:
        ax2.plot(nspt_predictions['NSPT_prediccion'], nspt_predictions['profundidad'], 'r-', linewidth=2)
        ax2.set_xlabel('NSPT (golpes)')
        ax2.set_ylabel('Profundidad (m)')
        ax2.set_title(f'Perfil NSPT vs. Profundidad - {location_name}')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.invert_yaxis()  # Invertir eje Y para que la profundidad aumente hacia abajo
    
    plt.tight_layout()
    
    # Guardar figura
    plt.savefig(figures_dir / f'prediccion_perfiles_{location_name.replace(" ", "_")}.png', dpi=300)
    plt.close()
    
    # También guardar como CSV
    if vs_predictions is not None:
        vs_predictions.to_csv(figures_dir / f'vs_prediccion_{location_name.replace(" ", "_")}.csv', index=False)
    
    if nspt_predictions is not None:
        nspt_predictions.to_csv(figures_dir / f'nspt_prediccion_{location_name.replace(" ", "_")}.csv', index=False)
    
    print(f"Visualizaciones guardadas en: {figures_dir}")

def main():
    """Función principal"""
    # Cargar modelo para Vs
    print("Cargando modelo para Vs...")
    model_vs, metrics_vs, features_vs = load_model('vs')
    
    # Cargar modelo para NSPT
    print("Cargando modelo para NSPT...")
    model_nspt, metrics_nspt, features_nspt = load_model('nspt')
    
    # Verificar si se cargaron los modelos
    if model_vs is None and model_nspt is None:
        print("No se pudieron cargar los modelos")
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
    vs_predictions = None
    nspt_predictions = None
    
    if model_vs is not None:
        print("\nGenerando perfil de Vs...")
        vs_predictions = predict_vs_depth_profile(model_vs, features_vs, coordinates, max_depth)
    
    if model_nspt is not None:
        print("Generando perfil de NSPT...")
        nspt_predictions = predict_nspt_depth_profile(model_nspt, features_nspt, coordinates, max_depth)
    
    # Visualizar predicciones
    print("Generando visualizaciones...")
    visualize_predictions(vs_predictions, nspt_predictions, location_name)
    
    print("\nPredicciones completadas con éxito!")

if __name__ == "__main__":
    main()