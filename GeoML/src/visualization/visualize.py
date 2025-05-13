#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para visualizar datos y resultados geotécnicos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def load_data(file_path):
    """
    Carga datos desde un archivo CSV.
    
    Parameters:
    -----------
    file_path : str or Path
        Ruta al archivo CSV
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con los datos cargados
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error al cargar {file_path}: {e}")
        return None

def plot_correlation_matrix(df, columns=None, title="Matriz de Correlación"):
    """
    Grafica una matriz de correlación.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos
    columns : list, optional
        Lista de columnas para incluir en la matriz
    title : str, optional
        Título del gráfico
    """
    # Filtrar columnas si se especifican
    if columns:
        # Verificar cuáles de las columnas existen en el DataFrame
        existing_columns = [col for col in columns if col in df.columns]
        if not existing_columns:
            print(f"Ninguna de las columnas especificadas existe en el DataFrame")
            return
        df_subset = df[existing_columns]
    else:
        # Filtrar solo columnas numéricas
        df_subset = df.select_dtypes(include=['number'])
    
    # Calcular correlaciones
    corr = df_subset.corr()
    
    # Crear figura
    plt.figure(figsize=(12, 10))
    
    # Crear mapa de calor
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=.5,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": .5}
    )
    
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    # Crear directorio para guardar figura
    project_dir = Path(__file__).resolve().parents[2]
    figures_dir = project_dir / "results" / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # Guardar figura
    plt.savefig(figures_dir / "correlation_matrix.png", dpi=300)
    plt.close()
    
    print(f"Matriz de correlación guardada en {figures_dir / 'correlation_matrix.png'}")
    
    return corr

def plot_vs_depth_profile(df, location_name="Todos los Sitios"):
    """
    Grafica el perfil de Vs con respecto a la profundidad.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos
    location_name : str, optional
        Nombre del sitio para el título
    """
    # Verificar columnas necesarias
    if 'profundidad_media' not in df.columns or 'Vs (m/s)' not in df.columns:
        # Intentar crear profundidad_media si no existe
        if 'De' in df.columns and 'Hasta' in df.columns:
            df['profundidad_media'] = (df['De'] + df['Hasta']) / 2
        else:
            print("No se puede graficar el perfil de Vs sin columnas de profundidad y Vs")
            return
    
    # Filtrar datos válidos
    df_valid = df.dropna(subset=['profundidad_media', 'Vs (m/s)'])
    
    if len(df_valid) == 0:
        print("No hay datos válidos para graficar")
        return
    
    # Crear figura
    plt.figure(figsize=(10, 12))
    
    # Verificar si hay columna de tipo de ensayo o ítem
    if 'tipo_ensayo' in df_valid.columns:
        # Graficar por tipo de ensayo
        for ensayo, grupo in df_valid.groupby('tipo_ensayo'):
            plt.scatter(
                grupo['Vs (m/s)'],
                grupo['profundidad_media'],
                label=ensayo,
                alpha=0.7,
                s=50
            )
    elif '�tem' in df_valid.columns:
        # Graficar por ítem
        for item, grupo in df_valid.groupby('�tem'):
            plt.scatter(
                grupo['Vs (m/s)'],
                grupo['profundidad_media'],
                label=item,
                alpha=0.7,
                s=50
            )
    else:
        # Graficar todo junto
        plt.scatter(
            df_valid['Vs (m/s)'],
            df_valid['profundidad_media'],
            alpha=0.7,
            s=50
        )
    
    # Ajustar gráfico
    plt.xlabel('Velocidad S (m/s)', fontsize=12)
    plt.ylabel('Profundidad (m)', fontsize=12)
    plt.title(f'Perfil de Vs vs. Profundidad - {location_name}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().invert_yaxis()  # Invertir eje Y para que la profundidad aumente hacia abajo
    
    # Añadir leyenda si hay múltiples series
    if ('tipo_ensayo' in df_valid.columns or '�tem' in df_valid.columns) and len(df_valid.groupby('tipo_ensayo' if 'tipo_ensayo' in df_valid.columns else '�tem')) > 1:
        plt.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Crear directorio para guardar figura
    project_dir = Path(__file__).resolve().parents[2]
    figures_dir = project_dir / "results" / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # Guardar figura
    plt.savefig(figures_dir / f"perfil_vs_{location_name.replace(' ', '_')}.png", dpi=300)
    plt.close()
    
    print(f"Perfil de Vs guardado en {figures_dir / f'perfil_vs_{location_name.replace(' ', '_')}.png'}")

def plot_nspt_depth_profile(df, location_name="Todos los Sitios"):
    """
    Grafica el perfil de NSPT con respecto a la profundidad.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos
    location_name : str, optional
        Nombre del sitio para el título
    """
    # Verificar columnas necesarias
    if 'profundidad_media' not in df.columns or 'Nspt' not in df.columns:
        # Intentar crear profundidad_media si no existe
        if 'De' in df.columns and 'Hasta' in df.columns:
            df['profundidad_media'] = (df['De'] + df['Hasta']) / 2
        else:
            print("No se puede graficar el perfil de NSPT sin columnas de profundidad y NSPT")
            return
    
    # Filtrar datos válidos
    df_valid = df.dropna(subset=['profundidad_media', 'Nspt'])
    
    if len(df_valid) == 0:
        print("No hay datos válidos para graficar")
        return
    
    # Crear figura
    plt.figure(figsize=(10, 12))
    
    # Verificar si hay columna de tipo de ensayo o ítem
    if 'tipo_ensayo' in df_valid.columns:
        # Graficar por tipo de ensayo
        for ensayo, grupo in df_valid.groupby('tipo_ensayo'):
            plt.scatter(
                grupo['Nspt'],
                grupo['profundidad_media'],
                label=ensayo,
                alpha=0.7,
                s=50
            )
    elif '�tem' in df_valid.columns:
        # Graficar por ítem
        for item, grupo in df_valid.groupby('�tem'):
            plt.scatter(
                grupo['Nspt'],
                grupo['profundidad_media'],
                label=item,
                alpha=0.7,
                s=50
            )
    else:
        # Graficar todo junto
        plt.scatter(
            df_valid['Nspt'],
            df_valid['profundidad_media'],
            alpha=0.7,
            s=50
        )
    
    # Ajustar gráfico
    plt.xlabel('NSPT (golpes)', fontsize=12)
    plt.ylabel('Profundidad (m)', fontsize=12)
    plt.title(f'Perfil de NSPT vs. Profundidad - {location_name}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().invert_yaxis()  # Invertir eje Y para que la profundidad aumente hacia abajo
    
    # Añadir leyenda si hay múltiples series
    if ('tipo_ensayo' in df_valid.columns or '�tem' in df_valid.columns) and len(df_valid.groupby('tipo_ensayo' if 'tipo_ensayo' in df_valid.columns else '�tem')) > 1:
        plt.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Crear directorio para guardar figura
    project_dir = Path(__file__).resolve().parents[2]
    figures_dir = project_dir / "results" / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # Guardar figura
    plt.savefig(figures_dir / f"perfil_nspt_{location_name.replace(' ', '_')}.png", dpi=300)
    plt.close()
    
    print(f"Perfil de NSPT guardado en {figures_dir / f'perfil_nspt_{location_name.replace(' ', '_')}.png'}")

def plot_model_evaluation(y_true, y_pred, model_name, feature_names=None):
    """
    Grafica la evaluación de un modelo: predicciones vs. valores reales, residuos e importancia de features.
    
    Parameters:
    -----------
    y_true : array-like
        Valores reales
    y_pred : array-like
        Valores predichos
    model_name : str
        Nombre del modelo
    feature_names : list, optional
        Lista de nombres de features
    """
    # Crear directorio para figuras
    project_dir = Path(__file__).resolve().parents[2]
    figures_dir = project_dir / "results" / "figures" / "model_evaluation"
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # Calcular métricas
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Crear figura de evaluación
    plt.figure(figsize=(12, 6))
    
    # Predicciones vs. valores reales
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.7)
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    
    plt.xlabel('Valores Reales')
    plt.ylabel('Valores Predichos')
    plt.title(f'Predicción vs. Real - {model_name}')
    
    # Añadir texto con métricas
    text = f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.4f}"
    plt.annotate(
        text,
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Residuos
    plt.subplot(1, 2, 2)
    residuos = y_true - y_pred
    plt.scatter(y_pred, residuos, alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='--', lw=2)
    
    plt.xlabel('Valores Predichos')
    plt.ylabel('Residuos')
    plt.title('Residuos')
    
    plt.tight_layout()
    plt.savefig(figures_dir / f"evaluacion_{model_name}.png", dpi=300)
    plt.close()
    
    print(f"Evaluación del modelo guardada en {figures_dir / f'evaluacion_{model_name}.png'}")

def main():
    """Función principal."""
    # Obtener directorio del proyecto
    project_dir = Path(__file__).resolve().parents[2]
    
    # Cargar datos procesados
    processed_path = project_dir / "data" / "processed" / "combined_data.csv"
    
    if processed_path.exists():
        print("Cargando datos procesados...")
        df = load_data(processed_path)
        
        if df is not None:
            print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
            
            # Generar visualizaciones
            print("\nGenerando matriz de correlación...")
            correlations = plot_correlation_matrix(
                df, 
                columns=['profundidad_media', 'Vs (m/s)', 'Nspt', 'Gravas', 'Arenas', 'Finos'],
                title="Correlación entre Variables Geotécnicas"
            )
            
            print("\nGenerando perfil de Vs...")
            plot_vs_depth_profile(df)
            
            print("\nGenerando perfil de NSPT...")
            plot_nspt_depth_profile(df)
            
        else:
            print("No se pudieron cargar los datos procesados.")
    else:
        print(f"No se encontró el archivo: {processed_path}")
        print("Ejecuta primero el script src.data.make_dataset para procesar los datos.")

if __name__ == "__main__":
    main()