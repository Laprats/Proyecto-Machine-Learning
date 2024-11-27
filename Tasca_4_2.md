# Proyecto Recomendaciones de Productos Financieros Personalizados

## Fuentes de los Datos

### Identificación de los fuentes:

- Datos de campañas telefónicas almacenados en la base de datos interna del sistema bancario.

### Descripción de las Fuentes:

- Información recopilada directamente mediante interacciones telefónicas con los clientes durante campañas de marketing.

## Métodos de Recolección de Datos

### Procedimientos y Herramientas:

- Llamadas telefónicas realizadas por agentes a clientes potenciales.
- Registro automático de respuestas (positivo/negativo), duración de las llamadas, y otras interacciones relevantes mediante un sistema de gestión de llamadas (Call Management System).
- Integración automática con la base de datos interna del banco para almacenar la información recopilada

### Frecuencia de Recolección:

- Las campañas se realizan periódicamente, y los datos se recogen en tiempo real durante la interacción con los clientes.

### Scripts de Descarga:

`Phyton

import pandas as pd

data_path = "bank_dataset.csv"
bank_data = pd.read_csv(data_path)

print(bank_data.info())


## Formato y Estructura de los Datos

### Tipos de Datos:

- Numéricos: Edad, saldo de cuenta, duración de llamadas.
- Categóricos: Nivel educativo, estado civil, resultados de campañas anteriores.
- Temporales: Fecha, mes y día de la interacción.

### Formato de Almacenamiento:

- Datos tabulares en formato CSV con almacenamiento seguro en servidores del banco.

### Limitaciones de los Datos:

- Incompletitud:
    -Algunos clientes pueden no haber respondido a todas las campañas.
- Sesgos:
    -Los datos pueden estar sesgados hacia clientes que respondieron positivamente en campañas anteriores.
- Actualización Asincrónica:
    -Puede haber discrepancias en la actualización de los datos almacenados entre interacciones y procesamiento.

## Cosideración sobre Datos Sensibles

### Tipos de Datos Sensibles:

- Información personal (edad, estado civil, ocupación).
- Información financiera (saldo de la cuenta, historial de respuestas).

## Medidas de Protección:

- Anonimización de datos personales mediante Iniciales o Pseudónimos.
- Acceso restringido a datos sensibles únicamente al personal autorizado.
- Cumplimiento con la normativa de protección de datos (e.g., GDPR).


```python

```
