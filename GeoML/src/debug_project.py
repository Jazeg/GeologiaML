#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de diagnóstico para verificar la estructura del proyecto.
"""

import os
from pathlib import Path

def check_directory_structure():
    """Verifica la estructura de directorios del proyecto."""
    # Obtener directorio del proyecto
    project_dir = Path(__file__).resolve().parents[1]
    print(f"Directorio del proyecto: {project_dir}")
    
    # Verificar directorios principales
    directories = [
        "data/raw",
        "data/processed",
        "models/saved",
        "models/evaluation",
        "src/data",
        "src/features",
        "src/models",
        "src/visualization"
    ]
    
    for directory in directories:
        dir_path = project_dir / directory
        exists = dir_path.exists()
        print(f"Directorio '{directory}': {'✅ Existe' if exists else '❌ No existe'}")
        
        if exists:
            # Listar archivos en el directorio
            files = list(dir_path.glob("*"))
            print(f"  Archivos: {[f.name for f in files]}")

def check_data_files():
    """Verifica los archivos de datos."""
    # Obtener directorio del proyecto
    project_dir = Path(__file__).resolve().parents[1]
    
    # Verificar archivos CSV en data/raw
    raw_dir = project_dir / "data" / "raw"
    if raw_dir.exists():
        csv_files = list(raw_dir.glob("*.csv"))
        print(f"\nArchivos CSV encontrados en data/raw: {len(csv_files)}")
        
        for csv_file in csv_files:
            print(f"  - {csv_file.name}")
            # Intentar leer primeras líneas del archivo
            try:
                with open(csv_file, 'r', encoding='latin1') as f:
                    header = f.readline().strip()
                print(f"    Encabezado: {header[:80]}...")
            except Exception as e:
                print(f"    Error al leer archivo: {e}")

def main():
    """Función principal."""
    print("="*50)
    print("DIAGNÓSTICO DEL PROYECTO GEOML")
    print("="*50)
    
    # Verificar estructura de directorios
    check_directory_structure()
    
    # Verificar archivos de datos
    check_data_files()
    
    print("\n" + "="*50)
    print("DIAGNÓSTICO COMPLETADO")
    print("="*50)

if __name__ == "__main__":
    main()