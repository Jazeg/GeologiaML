#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para combinar datos de múltiples archivos CSV en un solo dataset.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob

def load_and_combine_data(data_dir):
    """
    Carga todos los CSV de colegios y los combina en un solo DataFrame.
    
    Parameters:
    -----------
    data_dir : str
        Directorio que contiene los archivos CSV
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame combinado con todos los datos
    """
    # Obtener lista de archivos CSV
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print(f"No se encontraron archivos CSV en {data_dir}")
        return None
    
    # Lista para almacenar DataFrames
    dfs = []
    
    # Procesar cada archivo
    for file_path in csv_files:
        try:
            # Extraer nombre del colegio del archivo
            filename = os.path.basename(file_path)
            school_name = filename.replace(".csv", "").replace("_", " ")
            
            # Cargar CSV (probando diferentes encodings)
            for encoding in ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']:
                try:
                    df = pd.read_csv(file_path, delimiter=';', encoding=encoding)
                    if len(df.columns) > 1:  # Verificar que se haya leído correctamente
                        break
                except UnicodeDecodeError:
                    continue
            
            # Verificar si se pudo leer el archivo
            if len(df.columns) <= 1:
                print(f"No se pudo leer correctamente: {file_path}")
                continue
                
            # Eliminar filas totalmente vacías
            df = df.dropna(how='all')
            
            # Reemplazar '-' por NaN
            df = df.replace('-', np.nan)
            
            # Agregar columna con nombre del colegio
            df['colegio'] = school_name
            
            # Agregar a la lista
            dfs.append(df)
            print(f"Procesado: {filename} - {len(df)} filas")
            
        except Exception as e:
            print(f"Error procesando {file_path}: {e}")
    
    # Combinar todos los DataFrames
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Dataset combinado: {len(combined_df)} filas, {combined_df.shape[1]} columnas")
        return combined_df
    else:
        print("No se pudo procesar ningún archivo")
        return None

def clean_combined_data(df):
    """
    Limpia y prepara el dataset combinado.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame combinado a limpiar
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame limpio
    """
    if df is None:
        return None
    
    # Hacer una copia para no modificar el original
    df_clean = df.copy()
    
    # Convertir columnas numéricas
    numeric_cols = ['Norte', 'Este', 'De', 'Hasta', 'potencia', 
                    'Vs (m/s)', 'Vp (m/s)', 'Nspt', '(N1)60', 
                    'LL %', 'LP', 'IP %', 'W%', 'Gravas', 'Arenas', 'Finos']
    
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Calcular profundidad media si no existe
    if 'profundidad_media' not in df_clean.columns:
        if 'De' in df_clean.columns and 'Hasta' in df_clean.columns:
            df_clean['profundidad_media'] = (df_clean['De'] + df_clean['Hasta']) / 2
    
    # Identificar tipo de ensayo a partir del ítem
    if '�tem' in df_clean.columns:
        df_clean['tipo_ensayo'] = df_clean['�tem'].str.split('-').str[0]
    
    # Eliminar filas con datos críticos faltantes
    if 'profundidad_media' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['profundidad_media'])
    
    # Eliminar columnas totalmente vacías
    df_clean = df_clean.dropna(axis=1, how='all')
    
    return df_clean

def main():
    """Función principal"""
    # Obtener directorio del proyecto
    project_dir = Path(__file__).resolve().parents[2]
    
    # Directorios de datos
    raw_data_dir = project_dir / "data" / "raw"
    processed_data_dir = project_dir / "data" / "processed"
    
    # Crear directorio processed si no existe
    processed_data_dir.mkdir(exist_ok=True, parents=True)
    
    # Cargar y combinar datos
    print(f"Cargando datos desde: {raw_data_dir}")
    combined_df = load_and_combine_data(raw_data_dir)
    
    if combined_df is not None:
        # Limpiar datos
        print("Limpiando y preparando datos...")
        clean_df = clean_combined_data(combined_df)
        
        # Guardar datos procesados
        processed_path = processed_data_dir / "combined_data.csv"
        clean_df.to_csv(processed_path, index=False)
        print(f"Datos combinados guardados en: {processed_path}")

if __name__ == "__main__":
    main()