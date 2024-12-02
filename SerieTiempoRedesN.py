import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Ruta del archivo CSV
ruta_archivo = r'C:\Users\ben19\Downloads\codigoIA\time_series_covid_19_confirmed.csv'

# Leer el archivo CSV
try:
    datos_covid = pd.read_csv(ruta_archivo, delimiter=';', encoding='utf-8')
    print("Archivo cargado exitosamente.")
except FileNotFoundError:
    print("Archivo no encontrado en la ruta especificada. Verifica la ruta.")
    exit()
except pd.errors.ParserError:
    print("Error al analizar el archivo CSV. Verifica el delimitador y el formato.")
    exit()

# Seleccionar el país de forma interactiva
paises_disponibles = datos_covid['Country/Region'].unique()
print("Países disponibles:", ", ".join(paises_disponibles))
pais = input("Ingresa el nombre del país que deseas analizar: ").strip()

if pais not in paises_disponibles:
    print(f"El país '{pais}' no está en los datos. Verifica y vuelve a intentarlo.")
    exit()

# Filtrar por el país seleccionado
datos_pais = datos_covid[datos_covid['Country/Region'] == pais]

# Transformar las fechas en columnas a un índice de tiempo
try:
    serie_tiempo = datos_pais.iloc[:, 4:].T  # Las columnas de datos comienzan desde la 4ta
    serie_tiempo.index = pd.to_datetime(serie_tiempo.index, format='%m/%d/%y', errors='coerce')
    serie_tiempo.columns = ['Casos Confirmados']
    
    # Verificar y eliminar duplicados en las fechas
    if serie_tiempo.index.duplicated().sum() > 0:
        print(f"Se encontraron {serie_tiempo.index.duplicated().sum()} fechas duplicadas. Eliminándolas...")
        serie_tiempo = serie_tiempo[~serie_tiempo.index.duplicated(keep='first')]
    
    # Asegurar frecuencia diaria y rellenar valores faltantes
    serie_tiempo = serie_tiempo.asfreq('D').ffill()
except Exception as e:
    print(f"Error al procesar las fechas: {e}")
    exit()

# Normalizar los datos para que los valores estén entre 0 y 1
scaler = MinMaxScaler(feature_range=(0, 1))
serie_scaled = scaler.fit_transform(serie_tiempo)

# Función para crear secuencias de datos para entrenamiento
def crear_secuencias(data, pasos):
    X, y = [], []
    for i in range(len(data) - pasos - 1):
        X.append(data[i:(i + pasos), 0])
        y.append(data[i + pasos, 0])
    return np.array(X), np.array(y)

# Crear secuencias de datos (60 días de historial para predecir 1 día futuro)
pasos = 60
X, y = crear_secuencias(serie_scaled, pasos)

# Reshape para la entrada del LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Construcción del modelo LSTM
modelo = Sequential()
modelo.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))  # Aumentar unidades
modelo.add(LSTM(units=100, return_sequences=False))
modelo.add(Dense(units=1))

# Compilar el modelo
modelo.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
modelo.fit(X_train, y_train, epochs=50, batch_size=32)  # Aumentar épocas

# Hacer predicciones sobre el conjunto de prueba
predicciones = modelo.predict(X_test)

# Desnormalizar las predicciones
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
predicciones = scaler.inverse_transform(predicciones.reshape(-1, 1))

# Evaluación del modelo usando MSE y MAE
mse = mean_squared_error(y_test_original, predicciones)
mae = mean_absolute_error(y_test_original, predicciones)
print(f'MSE (Error Cuadrático Medio): {mse}')
print(f'MAE (Error Absoluto Medio): {mae}')

# Graficar los resultados
plt.figure(figsize=(12, 6))
plt.plot(serie_tiempo.index[-len(y_test):], y_test_original, label='Datos Reales', color='blue')
plt.plot(serie_tiempo.index[-len(y_test):], predicciones, label='Predicciones', color='red')
plt.title(f'Predicción de Casos Confirmados de COVID-19 en {pais}')
plt.xlabel('Fecha')
plt.ylabel('Casos Confirmados')
plt.text(0.01, 0.95, f'MSE: {mse:.2f}\nMAE: {mae:.2f}', transform=plt.gca().transAxes, fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.5))
plt.legend()
plt.grid(True)
plt.show()

# Calcular y graficar residuos
residuos = y_test_original - predicciones
plt.figure(figsize=(12, 6))
plt.plot(serie_tiempo.index[-len(y_test):], residuos, label='Residuos', color='purple')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('Análisis de Residuos')
plt.xlabel('Fecha')
plt.ylabel('Residuos')
plt.legend()
plt.grid(True)
plt.show()

# Predicción futura
pasos_futuros = 30
X_ultimo = serie_scaled[-pasos:].reshape(1, pasos, 1)
predicciones_futuras = []

for _ in range(pasos_futuros):
    prediccion = modelo.predict(X_ultimo)  # Salida tiene forma (1, 1)
    prediccion = np.expand_dims(prediccion, axis=-1)  # Expandir a (1, 1, 1)
    predicciones_futuras.append(prediccion[0, 0, 0])  # Guardar predicción
    X_ultimo = np.append(X_ultimo[:, 1:, :], prediccion, axis=1)

# Desnormalizar predicciones futuras
predicciones_futuras = scaler.inverse_transform(np.array(predicciones_futuras).reshape(-1, 1))

# Crear índice de fechas futuras
fechas_futuras = pd.date_range(start=serie_tiempo.index[-1], periods=pasos_futuros + 1, freq='D')[1:]

# Graficar predicciones futuras
plt.figure(figsize=(12, 6))
plt.plot(serie_tiempo.index, serie_tiempo['Casos Confirmados'], label='Datos Históricos', color='blue')
plt.plot(fechas_futuras, predicciones_futuras, label='Predicciones Futuras', color='orange')
plt.title(f'Predicción Futura de Casos Confirmados en {pais}')
plt.xlabel('Fecha')
plt.ylabel('Casos Confirmados')
plt.legend()
plt.grid(True)
plt.show()
