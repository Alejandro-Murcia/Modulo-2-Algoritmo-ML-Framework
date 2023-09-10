# Modulo-2-Algoritmo-ML-Framework
Actividad para Módulo 2 - Alejandro Murcia - A00828513

# Reporte: Implementación y Análisis de Random Forest

## Introducción

En este reporte, se presenta la implementación del algoritmo de Random Forest utilizando el conjunto de datos Pima Indians Diabetes. El objetivo es predecir si una persona tiene diabetes basándose en características como el número de embarazos, BMI, nivel de insulina, edad, entre otros.

## Implementación

Se utilizó el framework `scikit-learn` para la implementación. Inicialmente, se entrenó un modelo de Random Forest con parámetros predefinidos. Posteriormente, se llevó a cabo una optimización de hiperparámetros mediante `GridSearchCV` para mejorar el rendimiento del modelo.

## Resultados

Los resultados obtenidos son los siguientes:

- **Modelo Inicial**:
  - Accuracy en entrenamiento: 78.26%
  - Accuracy en validación: 74.68%

- **Modelo Optimizado**:
  - Accuracy en entrenamiento: 96.74%
  - Accuracy en validación: 68.83%

## Análisis

El modelo inicial mostró un buen equilibrio entre bias y varianza, con accuracies de entrenamiento y validación cercanos entre sí. No se observaron signos evidentes de sobreajuste o subajuste.

Sin embargo, tras la optimización, el modelo mejorado mostró un alto rendimiento en el conjunto de entrenamiento pero una disminución en el conjunto de validación. Esta discrepancia es un indicativo claro de sobreajuste, lo que significa que el modelo se adaptó demasiado bien a los datos de entrenamiento, perdiendo capacidad de generalización en datos no vistos.

## Conclusión

Aunque la optimización de hiperparámetros puede mejorar significativamente el rendimiento en el conjunto de entrenamiento, es esencial evaluar el modelo en un conjunto de validación para asegurarse de que no esté sobreajustando. En la siguiente entrega vamos a intentar explorar técnicas de regularización o ajustar nuevamente los hiperparámetros para obtener un modelo que generalice mejor en datos no vistos.

