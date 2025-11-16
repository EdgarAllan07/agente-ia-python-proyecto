import os
import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = None
scaler = None

# --- Cargar los modelos al iniciar el servidor ---
# try:
#     model = tf.keras.models.load_model('modelo_cardiovascular.h5')
#     scaler = joblib.load('scaler.joblib')
#     print("✅ Modelo y escalador cargados exitosamente.")
# except Exception as e:
#     print(f"❌ Error al cargar los archivos: {e}")
#     model = None
#     scaler = None

def load_resources():
    """Carga el modelo y el scaler solo una vez."""
    global model, scaler

    if model is None:
        print("🔄 Cargando modelo IA...")
        model = tf.keras.models.load_model('modelo_cardiovascular.h5')

    if scaler is None:
        print("🔄 Cargando scaler...")
        scaler = joblib.load('scaler.joblib')

load_resources()


# Definir el orden exacto de las columnas que el modelo Keras espera
EXPECTED_FEATURES = [
    'age', 'male', 'sysBP', 'diaBP', 
    'totChol', 'glucose', 'BMI', 'currentSmoker'
]

# --- NUEVAS REGLAS DE NEGOCIO ---

def calcular_hipertension(sys_bp, dia_bp):
    """Calcula el riesgo de Hipertensión (basado en guías de la AHA)"""
    if sys_bp >= 140 or dia_bp >= 90:
        return {"probabilidad": 92, "nivel": "Alto"}
    elif sys_bp >= 130 or dia_bp >= 80:
        return {"probabilidad": 75, "nivel": "Moderado"}
    elif sys_bp >= 120 and dia_bp < 80:
        return {"probabilidad": 45, "nivel": "Moderado"}
    else:
        return {"probabilidad": 10, "nivel": "Bajo"}

def calcular_diabetes(glucosa):
    """Calcula el riesgo de Diabetes (basado en guías de la ADA)"""
    if glucosa >= 126:
        return {"probabilidad": 95, "nivel": "Alto"}
    elif glucosa >= 100:
        return {"probabilidad": 45, "nivel": "Moderado"}
    else:
        return {"probabilidad": 10, "nivel": "Bajo"}

def calcular_dislipidemia(ldl, hdl, total):
    """
    Calcula el riesgo de Dislipidemia (Colesterol)
    Basado en guías de la AHA/CDC.
    Devuelve el peor riesgo encontrado.
    """
    riesgos = []
    # 1. LDL (Malo)
    if ldl >= 160:
        riesgos.append({"probabilidad": 85, "nivel": "Alto"})
    elif ldl >= 130:
        riesgos.append({"probabilidad": 60, "nivel": "Moderado"})
    
    # 2. HDL (Bueno)
    if hdl < 40:
        riesgos.append({"probabilidad": 90, "nivel": "Alto"})
        
    # 3. Total
    if total >= 240:
        riesgos.append({"probabilidad": 80, "nivel": "Alto"})
    elif total >= 200:
        riesgos.append({"probabilidad": 50, "nivel": "Moderado"})

    # Si no hay riesgos, es bajo
    if not riesgos:
        return {"probabilidad": 10, "nivel": "Bajo"}

    # Devolver el riesgo MÁS ALTO encontrado
    return max(riesgos, key=lambda x: x['probabilidad'])

def calcular_riesgo_estilo_vida(fumador, actividad, alcohol):
    """
    Calcula el riesgo basado en el estilo de vida.
    """
    puntos_riesgo = 0
    
    if fumador:
        puntos_riesgo += 1
    if actividad == 'Sedentario':
        puntos_riesgo += 1
    if alcohol in ['Moderado', 'Alto']:
        puntos_riesgo += 1
        
    # Mapear puntos a un resultado
    if puntos_riesgo == 3:
        return {"probabilidad": 90, "nivel": "Alto"}
    elif puntos_riesgo == 2:
        return {"probabilidad": 60, "nivel": "Moderado"}
    elif puntos_riesgo == 1:
        return {"probabilidad": 30, "nivel": "Bajo"}
    else:
        return {"probabilidad": 10, "nivel": "Bajo"}

def nivel_riesgo_coronario(prob_raw):
    """ Mapea la probabilidad (0.0 a 1.0) a un Nivel """
    if prob_raw >= 0.70:
        return "Alto"
    elif prob_raw >= 0.30:
        return "Moderado"
    else:
        return "Bajo"

# --- RUTA DE API PRINCIPAL (ACTUALIZADA) ---
@app.route('/api/evaluate', methods=['GET'])
def evaluate_risk_get():
    return jsonify({"message": "Hello, World!"})
    
@app.route('/api/evaluate', methods=['POST'])
def evaluate_risk():
    if not model or not scaler:
        return jsonify({"error": "El modelo de IA no está disponible."}), 500

    try:
        # 1. Obtener los datos JSON completos del backend de Node.js
        data = request.json
        
        # 2. --- INICIO DEL ETL (Preparar datos para Keras) ---
        altura_m = float(data['altura_cm']) / 100
        peso_kg = float(data['peso_kg'])
        bmi = peso_kg / (altura_m ** 2)

        input_data = [
            float(data['edad']),
            1.0 if data['sexo'] == 'Masculino' else 0.0,
            float(data['presion_sistolica']),
            float(data['presion_diastolica']),
            float(data['colesterol_total']),
            float(data['glucosa']),
            bmi,
            1.0 if data['fumador'] else 0.0
        ]
        
        # 3. Escalar los datos
        input_array = np.array([input_data])
        input_scaled = scaler.transform(input_array)
        
        # 4. --- CÁLCULO #1: Enfermedad Coronaria (IA) ---
        prob_coronaria_raw = model.predict(input_scaled)[0][0]
        prob_coronaria_pct = int(prob_coronaria_raw * 100)
        
        # 5. --- CÁLCULO #2: Hipertensión (Reglas) ---
        prob_hipertension_obj = calcular_hipertension(
            float(data['presion_sistolica']), 
            float(data['presion_diastolica'])
        )
        
        # 6. --- CÁLCULO #3: Diabetes (Reglas) ---
        prob_diabetes_obj = calcular_diabetes(
            float(data['glucosa'])
        )
        
        # 7. --- CÁLCULO #4: Dislipidemia (Reglas) ---
        prob_dislipidemia_obj = calcular_dislipidemia(
            float(data['colesterol_ldl']),
            float(data['colesterol_hdl']),
            float(data['colesterol_total'])
        )
        
        # 8. --- CÁLCULO #5: Riesgo Estilo de Vida (Reglas) ---
        prob_estilo_vida_obj = calcular_riesgo_estilo_vida(
            data['fumador'],
            data['actividad_fisica'],
            data['consumo_alcohol']
        )
        
        # Lista de todas las probabilidades calculadas
        probabilidades = [
            prob_coronaria_pct,
            prob_hipertension_obj['probabilidad'],
            prob_diabetes_obj['probabilidad'],
            prob_dislipidemia_obj['probabilidad'],
            prob_estilo_vida_obj['probabilidad']
        ]

        # 9. --- CÁLCULO #6: Riesgo General ---
        # El riesgo general es el MÁXIMO riesgo encontrado
        riesgo_general_pct = max(probabilidades)
        
        # 10. --- CONSTRUIR RESPUESTA JSON ---
        response_json = {
            "riesgo_general": riesgo_general_pct,
            "modelo_version": "1.2-hibrido-avanzado",
            "probabilidades_enfermedades": [
                {
                    "enfermedad": "Coronary Artery Disease",
                    "probabilidad": prob_coronaria_pct,
                    "nivel": nivel_riesgo_coronario(prob_coronaria_raw)
                },
                {
                    "enfermedad": "Hypertension",
                    "probabilidad": prob_hipertension_obj['probabilidad'],
                    "nivel": prob_hipertension_obj['nivel']
                },
                {
                    "enfermedad": "Type 2 Diabetes",
                    "probabilidad": prob_diabetes_obj['probabilidad'],
                    "nivel": prob_diabetes_obj['nivel']
                },
                {
                    "enfermedad": "Dislipidemia (Colesterol)",
                    "probabilidad": prob_dislipidemia_obj['probabilidad'],
                    "nivel": prob_dislipidemia_obj['nivel']
                },
                {
                    "enfermedad": "Riesgo por Estilo de Vida",
                    "probabilidad": prob_estilo_vida_obj['probabilidad'],
                    "nivel": prob_estilo_vida_obj['nivel']
                }
            ]
        }
        
        return jsonify(response_json)

    except KeyError as e:
        return jsonify({"error": f"Falta el campo en los datos: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error interno del agente: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
