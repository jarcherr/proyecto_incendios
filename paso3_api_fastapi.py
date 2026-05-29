"""
╔══════════════════════════════════════════════════════════════════════╗
║  EJERCICIO 7 - PASO 3: API REST CON FASTAPI                       ║
║  Proyecto: Predicción de Incendios Forestales                      ║
║  Tema: API REST + ML + RAG                                         ║
╚══════════════════════════════════════════════════════════════════════╝

DESCRIPCIÓN:
    API REST que integra:
    1. MODELO ML: Predicción de probabilidad de incendio basada en
       condiciones meteorológicas (Random Forest).
    2. SISTEMA RAG: Consultas en lenguaje natural sobre incendios
       forestales con contexto de fuentes verificadas.
    3. ESTADÍSTICAS: Análisis correlacional y métricas del modelo.

ENDPOINTS:
    POST /predict          → Predecir probabilidad de incendio
    POST /query            → Consultar base de conocimiento (RAG)
    GET  /stats            → Métricas del modelo y correlaciones
    GET  /alert-level      → Nivel de alerta según condiciones
    GET  /health           → Estado del servicio
    GET  /docs             → Documentación interactiva (Swagger)

EJECUCIÓN:
    python3 paso3_api_fastapi.py
    
    La API estará disponible en: http://localhost:8000
    Documentación Swagger: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import pickle
import json
import numpy as np
import os
import sys

# Agregar el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paso2_rag_knowledge_base import LightweightRAG

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE LA API
# ═══════════════════════════════════════════════════════════════
app = FastAPI(
    title="API de Predicción de Incendios Forestales",
    description="""
    ## Sistema Inteligente de Predicción y Consulta sobre Incendios Forestales
    
    Esta API combina **Machine Learning** y **RAG (Retrieval-Augmented Generation)** 
    para proporcionar:
    
    - **Predicciones** de probabilidad de incendio basadas en condiciones meteorológicas
    - **Consultas** en lenguaje natural sobre incendios forestales
    - **Estadísticas** correlacionales y métricas del modelo
    - **Niveles de alerta** según el sistema FWI
    
    ### Tecnologías
    - **Modelo ML:** Random Forest (scikit-learn)
    - **RAG:** TF-IDF + Cosine Similarity
    - **API:** FastAPI + Pydantic
    
    ### Fuentes de Datos
    - Dataset basado en Algerian Forest Fires (UCI ML Repository)
    - Base de conocimiento: NASA FIRMS, IPCC, CONAF, NIFC, FAO, EFFIS
    
    ---
    *Ejercicio 7 - Curso de Inteligencia Artificial*
    """,
    version="1.0.0",
    contact={"name": "Curso IA - Agentes Inteligentes"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════
# CARGAR MODELOS Y DATOS
# ═══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Modelo ML
with open(os.path.join(BASE_DIR, 'modelo_incendios.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'scaler_incendios.pkl'), 'rb') as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, 'metricas_modelo.json'), 'r') as f:
    metrics = json.load(f)

# Sistema RAG
rag = LightweightRAG()
rag.load(os.path.join(BASE_DIR, 'knowledge_base.pkl'))

# Feature columns
FEATURE_COLS = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']


# ═══════════════════════════════════════════════════════════════
# MODELOS DE DATOS (Pydantic)
# ═══════════════════════════════════════════════════════════════
class WeatherInput(BaseModel):
    """Condiciones meteorológicas para predicción."""
    temperature: float = Field(..., ge=0, le=50, description="Temperatura en °C (0-50)")
    humidity: float = Field(..., ge=0, le=100, description="Humedad relativa en % (0-100)")
    wind_speed: float = Field(..., ge=0, le=60, description="Velocidad del viento en km/h (0-60)")
    rain: float = Field(0.0, ge=0, le=50, description="Precipitación en mm (0-50)")
    ffmc: Optional[float] = Field(None, ge=0, le=100, description="Fine Fuel Moisture Code (0-100)")
    dmc: Optional[float] = Field(None, ge=0, le=500, description="Duff Moisture Code (0-500)")
    dc: Optional[float] = Field(None, ge=0, le=1000, description="Drought Code (0-1000)")
    isi: Optional[float] = Field(None, ge=0, le=50, description="Initial Spread Index (0-50)")
    bui: Optional[float] = Field(None, ge=0, le=300, description="Buildup Index (0-300)")
    fwi: Optional[float] = Field(None, ge=0, le=100, description="Fire Weather Index (0-100)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "temperature": 35.0,
                "humidity": 25.0,
                "wind_speed": 20.0,
                "rain": 0.0,
                "ffmc": 85.0,
                "dmc": 60.0,
                "dc": 200.0,
                "isi": 12.0,
                "bui": 50.0,
                "fwi": 25.0
            }
        }


class PredictionResponse(BaseModel):
    """Respuesta de predicción."""
    fire_probability: float
    fire_risk: str
    alert_level: str
    alert_color: str
    confidence: float
    factors: dict
    recommendation: str


class QueryInput(BaseModel):
    """Consulta en lenguaje natural."""
    question: str = Field(..., min_length=5, description="Pregunta sobre incendios forestales")
    top_k: int = Field(3, ge=1, le=5, description="Número de fuentes a consultar")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "¿Cuáles son los factores principales que causan incendios forestales?",
                "top_k": 3
            }
        }


class QueryResponse(BaseModel):
    """Respuesta de consulta RAG."""
    question: str
    context: str
    sources: list
    num_sources: int
    note: str


class AlertInput(BaseModel):
    """Entrada para nivel de alerta."""
    temperature: float = Field(..., description="Temperatura en °C")
    humidity: float = Field(..., description="Humedad relativa en %")
    wind_speed: float = Field(..., description="Velocidad del viento en km/h")
    days_without_rain: int = Field(0, description="Días sin lluvia")
    fwi: Optional[float] = Field(None, description="Fire Weather Index")


# ═══════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════════
def estimate_fwi_indices(temp, humidity, wind, rain):
    """
    Estima los índices FWI cuando no se proporcionan.
    Usa fórmulas simplificadas basadas en las correlaciones del dataset.
    """
    ffmc = max(0, min(96, 90 - humidity * 0.3 + temp * 0.5))
    dmc = max(1, min(300, temp * 2 - humidity * 0.5))
    dc = max(7, min(800, temp * 5 - rain * 10 + 100))
    isi = max(0, min(20, ffmc * 0.1 + wind * 0.3))
    bui = max(1, min(200, dmc * 0.5 + dc * 0.1))
    fwi = max(0, min(60, isi * bui * 0.1))
    return ffmc, dmc, dc, isi, bui, fwi


def get_alert_level(fwi_value):
    """Determina el nivel de alerta según el FWI."""
    if fwi_value < 10:
        return "VERDE", "green", "Riesgo bajo. Operaciones normales."
    elif fwi_value < 20:
        return "AMARILLO", "yellow", "Riesgo moderado. Aumentar vigilancia."
    elif fwi_value < 30:
        return "NARANJA", "orange", "Riesgo alto. Activar brigadas de prevención."
    else:
        return "ROJO", "red", "Riesgo EXTREMO. Prohibir actividades en zonas forestales."


def get_risk_factors(features):
    """Analiza los factores de riesgo individuales."""
    temp, rh, ws, rain = features[0], features[1], features[2], features[3]
    factors = {}
    
    if temp > 35:
        factors["temperatura"] = f"CRÍTICA ({temp}°C > 35°C)"
    elif temp > 30:
        factors["temperatura"] = f"ALTA ({temp}°C > 30°C)"
    else:
        factors["temperatura"] = f"Normal ({temp}°C)"
    
    if rh < 25:
        factors["humedad"] = f"CRÍTICA ({rh}% < 25%)"
    elif rh < 40:
        factors["humedad"] = f"BAJA ({rh}% < 40%)"
    else:
        factors["humedad"] = f"Normal ({rh}%)"
    
    if ws > 25:
        factors["viento"] = f"PELIGROSO ({ws} km/h > 25 km/h)"
    elif ws > 15:
        factors["viento"] = f"Moderado ({ws} km/h)"
    else:
        factors["viento"] = f"Normal ({ws} km/h)"
    
    if rain == 0:
        factors["lluvia"] = "Sin precipitación (riesgo elevado)"
    else:
        factors["lluvia"] = f"Precipitación: {rain} mm"
    
    return factors


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root():
    """Página principal con información de la API."""
    return """
    <html>
    <head>
        <title>API Predicción de Incendios Forestales</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; 
                   max-width: 800px; margin: 0 auto; padding: 40px; }
            h1 { color: #22d3ee; }
            h2 { color: #fbbf24; }
            a { color: #22d3ee; }
            .endpoint { background: #1e293b; padding: 15px; border-radius: 8px; margin: 10px 0;
                       border-left: 3px solid #22d3ee; }
            .method { color: #22c55e; font-weight: bold; }
            code { background: #334155; padding: 2px 6px; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1>🔥 API de Predicción de Incendios Forestales</h1>
        <p>Sistema inteligente que combina <strong>Machine Learning</strong> y 
        <strong>RAG</strong> para predicción y consulta sobre incendios.</p>
        
        <h2>Endpoints Disponibles</h2>
        
        <div class="endpoint">
            <span class="method">POST</span> <code>/predict</code> 
            — Predecir probabilidad de incendio
        </div>
        <div class="endpoint">
            <span class="method">POST</span> <code>/query</code> 
            — Consultar base de conocimiento (RAG)
        </div>
        <div class="endpoint">
            <span class="method">GET</span> <code>/stats</code> 
            — Métricas del modelo y correlaciones
        </div>
        <div class="endpoint">
            <span class="method">POST</span> <code>/alert-level</code> 
            — Calcular nivel de alerta
        </div>
        <div class="endpoint">
            <span class="method">GET</span> <code>/health</code> 
            — Estado del servicio
        </div>
        
        <h2>Documentación Interactiva</h2>
        <p>Visita <a href="/docs">/docs</a> para la documentación Swagger interactiva.</p>
        
        <h2>Ejemplo de Uso (cURL)</h2>
        <pre style="background: #1e293b; padding: 15px; border-radius: 8px; overflow-x: auto;">
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "temperature": 38,
    "humidity": 20,
    "wind_speed": 25,
    "rain": 0
  }'</pre>
        
        <p style="color: #94a3b8; margin-top: 30px;">
            Ejercicio 7 — Curso de Inteligencia Artificial — Agentes Inteligentes
        </p>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Verificar el estado del servicio."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "rag_loaded": len(rag.chunks) > 0,
        "model_accuracy": metrics.get("accuracy", 0),
        "knowledge_chunks": len(rag.chunks),
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_fire(weather: WeatherInput):
    """
    Predecir la probabilidad de incendio forestal.
    
    Recibe condiciones meteorológicas y devuelve:
    - Probabilidad de incendio (0-100%)
    - Nivel de riesgo (Bajo/Moderado/Alto/Extremo)
    - Nivel de alerta (Verde/Amarillo/Naranja/Rojo)
    - Factores de riesgo individuales
    - Recomendación
    
    Si no se proporcionan los índices FWI (FFMC, DMC, DC, ISI, BUI, FWI),
    se estimarán automáticamente a partir de las condiciones meteorológicas.
    """
    try:
        # Estimar índices FWI si no se proporcionan
        if weather.ffmc is None:
            ffmc, dmc, dc, isi, bui, fwi = estimate_fwi_indices(
                weather.temperature, weather.humidity, 
                weather.wind_speed, weather.rain
            )
        else:
            ffmc = weather.ffmc
            dmc = weather.dmc or 0
            dc = weather.dc or 0
            isi = weather.isi or 0
            bui = weather.bui or 0
            fwi = weather.fwi or 0
        
        # Preparar features
        features = np.array([[
            weather.temperature, weather.humidity, weather.wind_speed,
            weather.rain, ffmc, dmc, dc, isi, bui, fwi
        ]])
        
        # Escalar y predecir
        features_scaled = scaler.transform(features)
        probability = model.predict_proba(features_scaled)[0][1]
        prediction = model.predict(features_scaled)[0]
        
        # Determinar riesgo y alerta
        if probability < 0.25:
            risk = "BAJO"
        elif probability < 0.50:
            risk = "MODERADO"
        elif probability < 0.75:
            risk = "ALTO"
        else:
            risk = "EXTREMO"
        
        alert_level, alert_color, recommendation = get_alert_level(fwi)
        factors = get_risk_factors(features[0])
        
        # Confidence del modelo
        confidence = max(probability, 1 - probability)
        
        return PredictionResponse(
            fire_probability=round(float(probability) * 100, 2),
            fire_risk=risk,
            alert_level=alert_level,
            alert_color=alert_color,
            confidence=round(float(confidence) * 100, 2),
            factors=factors,
            recommendation=recommendation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_knowledge(query: QueryInput):
    """
    Consultar la base de conocimiento sobre incendios forestales.
    
    Usa RAG (Retrieval-Augmented Generation) para buscar información
    relevante en la base de conocimiento y devolver contexto con fuentes.
    
    En un sistema completo, este contexto se pasaría a un LLM para
    generar una respuesta en lenguaje natural.
    """
    try:
        results = rag.search(query.question, top_k=query.top_k)
        context = rag.generate_context(query.question, top_k=query.top_k)
        
        sources = [
            {
                "source": r['source'],
                "year": r['year'],
                "relevance": f"{r['similarity']:.2%}",
                "excerpt": r['chunk'][:200] + "..."
            }
            for r in results
        ]
        
        return QueryResponse(
            question=query.question,
            context=context,
            sources=sources,
            num_sources=len(sources),
            note="Este contexto se puede pasar a un LLM (GPT-4, Claude, etc.) "
                 "para generar una respuesta completa en lenguaje natural."
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en consulta: {str(e)}")


@app.get("/stats")
async def get_statistics():
    """
    Obtener estadísticas del modelo y análisis correlacional.
    
    Devuelve:
    - Métricas de rendimiento del modelo (accuracy, AUC-ROC)
    - Importancia de cada variable (Feature Importance)
    - Matriz de confusión
    - Reporte de clasificación detallado
    """
    return {
        "model_type": "Random Forest Classifier",
        "accuracy": metrics["accuracy"],
        "auc_roc": metrics["auc_roc"],
        "feature_importance": metrics["feature_importance"],
        "confusion_matrix": {
            "true_negatives": metrics["confusion_matrix"][0][0],
            "false_positives": metrics["confusion_matrix"][0][1],
            "false_negatives": metrics["confusion_matrix"][1][0],
            "true_positives": metrics["confusion_matrix"][1][1]
        },
        "classification_report": metrics["classification_report"],
        "interpretation": {
            "top_3_predictors": list(metrics["feature_importance"].keys())[:3],
            "note": "FWI (Fire Weather Index) es el predictor más fuerte, "
                    "seguido por DMC y BUI. Esto confirma que los índices "
                    "compuestos del sistema canadiense FWI son más informativos "
                    "que las variables meteorológicas individuales."
        }
    }


@app.post("/alert-level")
async def calculate_alert(alert: AlertInput):
    """
    Calcular el nivel de alerta según condiciones actuales.
    
    Evalúa las condiciones meteorológicas contra umbrales críticos
    y determina el nivel de alerta según el protocolo FWI.
    """
    # Calcular FWI si no se proporciona
    if alert.fwi is None:
        _, _, _, _, _, fwi = estimate_fwi_indices(
            alert.temperature, alert.humidity, alert.wind_speed, 0
        )
    else:
        fwi = alert.fwi
    
    level, color, recommendation = get_alert_level(fwi)
    
    # Evaluar condiciones críticas
    critical_conditions = []
    if alert.temperature > 35:
        critical_conditions.append(f"Temperatura crítica: {alert.temperature}°C > 35°C")
    if alert.humidity < 25:
        critical_conditions.append(f"Humedad crítica: {alert.humidity}% < 25%")
    if alert.wind_speed > 25:
        critical_conditions.append(f"Viento peligroso: {alert.wind_speed} km/h > 25 km/h")
    if alert.days_without_rain > 7:
        critical_conditions.append(f"Sequía prolongada: {alert.days_without_rain} días sin lluvia")
    
    critical_count = len(critical_conditions)
    
    if critical_count >= 3:
        override_note = ("ALERTA MÁXIMA: Se cumplen 3+ condiciones críticas simultáneamente. "
                        "La probabilidad de incendio supera el 80%.")
        level = "ROJO"
        color = "red"
    elif critical_count >= 2:
        override_note = "PRECAUCIÓN: Se cumplen 2 condiciones críticas."
    else:
        override_note = None
    
    return {
        "alert_level": level,
        "alert_color": color,
        "fwi_value": round(fwi, 1),
        "recommendation": recommendation,
        "critical_conditions": critical_conditions,
        "critical_count": critical_count,
        "override_note": override_note,
        "thresholds": {
            "verde": "FWI < 10",
            "amarillo": "FWI 10-20",
            "naranja": "FWI 20-30",
            "rojo": "FWI > 30"
        }
    }


# ═══════════════════════════════════════════════════════════════
# EJECUCIÓN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  PROYECTO: PREDICCIÓN DE INCENDIOS FORESTALES           ║")
    print("║  Paso 3: API REST con FastAPI                           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("Iniciando servidor...")
    print("  - Documentación: http://localhost:8000/docs")
    print("  - API Base:      http://localhost:8000")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
