# 🔥 Train-LLM: Sistema de Predicción de Incendios Forestales - Guatemala

Sistema de predicción de incendios forestales para Guatemala utilizando fine-tuning de LLM multimodal (Qwen3-VL) con Unsloth, análisis estadístico y predicción temporal.

## 📋 Estado del Proyecto

**Fase actual**: Fase 0 - Configuración Inicial ✅

- [x] Estructura de directorios creada
- [x] Archivos de configuración (.env, .gitignore)
- [ ] API Keys obtenidas (NASA FIRMS, Hugging Face)
- [ ] Datos descargados
- [ ] Análisis estadístico completado
- [ ] Modelo LLM fine-tuneado
- [ ] Predicción 2026 generada
- [ ] API desplegada
- [ ] Dashboard publicado

## 🎯 Objetivos

1. **Fine-tuning de LLM Multimodal**: Entrenar Qwen3-VL (4B/7B) para análisis de imágenes satelitales
2. **Análisis Estadístico**: Correlaciones, tests de significancia, feature importance
3. **Predicción Temporal 2026**: Forecasting con Prophet/ARIMA
4. **API REST**: FastAPI con endpoints para predicción tabular y vision
5. **Dashboard Interactivo**: Streamlit con mapas de calor y visualizaciones

## 🗂️ Estructura del Proyecto

```
proyecto_incendios/
├── Docs/
│   ├── INFORME.md          # Documento técnico completo
│   ├── PLAN.md             # Checklist de ejecución (9 fases)
│   └── GETTING_STARTED.md  # Guía de inicio rápido
├── notebooks/
│   ├── Incendios.ipynb     # Notebook 01: Data Download
│   └── (4 notebooks más por crear)
├── data/
│   ├── raw/                # Datos crudos NASA FIRMS
│   ├── processed/          # Datos limpios y procesados
│   └── outputs/            # Predicciones, calendarios
├── models/                 # Modelos entrenados
├── visualizations/         # Gráficas y mapas
├── scripts/                # Scripts de Python
├── api/                    # FastAPI + Dashboard
├── .env                    # Variables de entorno (NO SUBIR A GIT)
├── requirements.txt        # Dependencias Python
└── README.md              # Este archivo
```

## 🚀 Quick Start

### 1. Registros de APIs (15 minutos)

#### NASA FIRMS
```bash
# 1. Ir a: https://firms.modaps.eosdis.nasa.gov/api/area/
# 2. Completar formulario y obtener MAP_KEY
# 3. Revisar email
```

#### Hugging Face
```bash
# 1. Ir a: https://huggingface.co/join
# 2. Settings → Access Tokens → New token
# 3. Copiar token (hf_...)
```

#### Google Earth Engine
```bash
# 1. Ir a: https://signup.earthengine.google.com/
# 2. Registrarse con cuenta Google
# 3. Esperar aprobación (24-48h)
```

### 2. Configurar .env

```bash
# Editar archivo .env y reemplazar:
FIRMS_MAP_KEY=tu_map_key_aqui
HF_TOKEN=hf_tu_token_aqui
```

### 3. Instalar Dependencias

```bash
# Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar Unsloth (específico según tu CUDA)
pip install unsloth
```

### 4. Ejecutar Notebook 01

```bash
# Opción A: Jupyter local
jupyter notebook notebooks/Incendios.ipynb

# Opción B: Google Colab
# 1. Ir a: https://colab.research.google.com/
# 2. Upload → Incendios.ipynb
# 3. Ejecutar celdas
```

## 📚 Documentación

- **INFORME.md**: Blueprint técnico completo (arquitectura, modelo, datos)
- **PLAN.md**: Checklist ejecutable con 9 fases y 200+ pasos
- **GETTING_STARTED.md**: Guía de inicio paso a paso

## 🛠️ Stack Tecnológico

- **ML/DL**: PyTorch, Transformers, Unsloth, scikit-learn
- **LLM**: Qwen3-VL (7B) fine-tuned con LoRA
- **Geoespacial**: Google Earth Engine, Rasterio, GeoPandas, Folium
- **Series Temporales**: Prophet, ARIMA
- **API**: FastAPI, Uvicorn
- **Dashboard**: Streamlit, Plotly
- **Datos**: NASA FIRMS, Landsat 8/9, MODIS, VIIRS

## 📊 Datasets

- **NASA FIRMS**: Detecciones de puntos calientes 2014-2024
- **Landsat 8/9**: Imágenes satelitales alta resolución (30m)
- **MODIS/VIIRS**: Imágenes multiespectrales
- **WorldClim**: Datos meteorológicos históricos

## 🎓 Resultados Esperados

- **Random Forest**: Accuracy > 90% (mejora sobre 88% actual)
- **LLM Vision**: F1-Score > 0.85, BLEU > 0.6
- **Predicción 2026**: MAPE < 20%
- **API**: Latencia < 5s (Vision), < 2s (Tabular)

## 📅 Timeline

| Fase | Descripción | Duración |
|------|-------------|----------|
| 0 | Configuración inicial | 1 día |
| 1 | Adquisición de datos | 5 días |
| 2 | Procesamiento de datos | 5 días |
| 3 | Análisis estadístico | 2 días |
| 4 | Fine-tuning LLM | 2 días |
| 5 | Predicción 2026 | 2 días |
| 6 | Integración API | 2 días |
| 7 | Dashboard | 3 días |
| 8 | Testing | 2 días |
| 9 | Deployment | 1 día |
| **TOTAL** | | **~4 semanas** |

## 💰 Costos

- **Opción Gratuita**: Google Colab T4 + Kaggle (tiempo de entrenamiento mayor)
- **Opción Pago**: Colab Pro+ A100 ($50/mes por 1 mes)

## 🤝 Contribuciones

Este es un proyecto educativo. Si querés contribuir:
1. Fork del repo
2. Crear branch para tu feature
3. Commit con mensajes descriptivos
4. Pull request con descripción detallada

## 📝 Licencia

Este proyecto es de código abierto para fines educativos.

## 📧 Contacto

- **Autor**: Richard Ortiz
- **Fecha**: Febrero 2026
- **Proyecto**: Curso de Inteligencia Artificial 2026

---

**Próximos pasos**: Ver `Docs/GETTING_STARTED.md` para iniciar.
