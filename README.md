# API para Consultar un Modelo Estadístico predicción heart attack

Este proyecto implementa una API para consultar un modelo estadístico utilizando un pipeline previamente entrenado. 
La API está diseñada para ejecutarse en una instancia de EC2.

## Requisitos Previos

1. **Python 3.8 o superior**
2. **pip** instalado
3. **Acceso a una instancia de Amazon EC2 configurada**
4. **Git** instalado en la máquina local

## Contenido del Repositorio

- **`main.py`**: Código principal para la API. Implementa los endpoints y carga el modelo.
- **`pipeline.joblib`**: Archivo serializado del modelo estadístico que se utiliza para las predicciones.
- **`requirements.txt`**: Lista de dependencias necesarias para ejecutar el proyecto.
