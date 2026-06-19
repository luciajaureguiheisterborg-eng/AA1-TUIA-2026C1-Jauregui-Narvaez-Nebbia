
# Instrucciones de uso

## Construir la imagen
```bash
docker build -t rain-predictor .
```

## Ejecutar el contenedor
Pasá los datos del día como JSON por stdin:

```bash
echo '{"Date":"2024-01-15","Location":"Sydney","MinTemp":15.0,"MaxTemp":25.0,"Rainfall":0.0,"Evaporation":5.0,"Sunshine":8.0,"WindGustDir":"NW","WindGustSpeed":35.0,"WindDir9am":"N","WindDir3pm":"NW","WindSpeed9am":10.0,"WindSpeed3pm":20.0,"Humidity9am":60.0,"Humidity3pm":45.0,"Pressure9am":1015.0,"Pressure3pm":1012.0,"Cloud9am":3.0,"Cloud3pm":4.0,"Temp9am":18.0,"Temp3pm":23.0,"RainToday":"No"}' | docker run -i rain-predictor
```

## Resultado
El contenedor devuelve un JSON con la predicción:
```json
{"prediccion": "No llueve", "probabilidad": 0.23}
```