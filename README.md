# Predicción de Ingresos con Census Income Dataset (Adult)

Este proyecto tiene como objetivo *predecir si los ingresos anuales de una persona superan los $50,000 USD*, con base en datos del censo de los Estados Unidos.
Este conjunto de datos es conocido como *Census Income Dataset* o *Adult Dataset*.

Se trata de un problema de *clasificación supervisada binaria*, ampliamente utilizado en tareas de Machine Learning.

---

## Objetivo del Proyecto

Construir y evaluar un modelo de Machine Learning capaz de clasificar a una persona en una de las siguientes categorías:

* *0:* Ingresos anuales *≤ $50,000*
* *1:* Ingresos anuales *> $50,000*

utilizando variables demográficas, educativas y laborales.

---

## Dataset

* *Nombre:* Census Income Dataset (Adult)
* *Fuente:* UCI Machine Learning Repository
* *Tipo de problema:* Clasificación binaria
* *Cantidad de registros:* 48,842

### Variables incluidas

El dataset contiene atributos como:

* Edad
* Nivel educativo
* Estado civil
* Ocupación
* Relación laboral
* Horas trabajadas por semana
* Sexo
* País de origen

---

## Estructura del Repositorio

bash
.
├── data/

│   └── adult.csv

├── src/

│   ├── train.py

│   ├── preprocess.py

│   └── evaluate.py

├── outputs/

│   ├── classification_report.png

│   └── confusion_matrix.png

├── requirements.txt

└── README.md


---

## Metodología

1. *Carga del dataset*
2. *Preprocesamiento*

   * Limpieza de datos
   * Codificación de variables categóricas
   * Normalización / escalado de variables numéricas
3. *Entrenamiento del modelo*
4. *Evaluación del desempeño*

---

##  Resultados del Modelo

### 🔹 Reporte de Clasificación

| Clase            | Precision | Recall | F1-score   | Support |
| ---------------- | --------- | ------ | ---------- | ------- |
| ≤ 50K (0)        | 0.8856    | 0.9512 | 0.9172     | 37,155  |
| > 50K (1)        | 0.7969    | 0.6094 | 0.6907     | 11,687  |
| *Accuracy*     |           |        | *0.8694* |         |
| *Macro Avg*    | 0.8413    | 0.7803 | 0.8039     | 48,842  |
| *Weighted Avg* | 0.8644    | 0.8694 | 0.8630     | 48,842  |

---

### Matriz de Confusión

|                | Predicción ≤ 50K | Predicción > 50K |
| -------------- | ---------------- | ---------------- |
| *Real ≤ 50K* | 35,340           | 1,815            |
| *Real > 50K* | 4,565            | 7,122            |

*Interpretación:*

* El modelo presenta un alto desempeño al identificar personas con ingresos ≤ $50K.
* La clase de ingresos mayores a $50K es más difícil de predecir, debido al desbalance del dataset.
* Se obtiene un *accuracy global cercano al 87%*.

---

## Visualizaciones

### Reporte de Clasificación

![Reporte de Clasificación](models/<run_id>/evaluation/classification_report.html)

### Matriz de Confusión

![Matriz de Confusión](models/<run_id>/evaluation/confusion.png)

## Cómo Ejecutar el Proyecto

### 1️⃣ Clonar el repositorio

bash
git clone <url-del-repositorio>
cd <nombre-del-repo>


### 2️⃣ Crear entorno virtual (opcional)

bash
python -m venv venv
source venv/bin/activate


### 3️⃣ Instalar dependencias

bash
pip install -r requirements.txt


### 4️⃣ Entrenar el modelo

bash
python src/train.py


---

## Tecnologías Utilizadas

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib / Seaborn
* MLflow (opcional)

---

## Conclusiones

* El modelo logra un *buen rendimiento general* en la predicción de ingresos.
* Se observa una menor sensibilidad para la clase de ingresos altos.
* Posibles mejoras futuras:

  * Balanceo de clases (SMOTE)
  * Ajuste de hiperparámetros
  * Uso de modelos más complejos como Random Forest o XGBoost

---
