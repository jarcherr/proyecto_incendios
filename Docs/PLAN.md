# PLAN DE EJECUCIÓN: Train-LLM - Guatemala Fire Detection
## Checklist Completo de Implementación

---

## 📋 ÍNDICE RÁPIDO

- [Fase 0: Configuración Inicial](#fase-0-configuración-inicial)
- [Fase 1: Adquisición de Datos](#fase-1-adquisición-de-datos)
- [Fase 2: Procesamiento de Datos](#fase-2-procesamiento-de-datos)
- [Fase 3: Análisis Estadístico](#fase-3-análisis-estadístico)
- [Fase 4: Fine-Tuning LLM](#fase-4-fine-tuning-llm)
- [Fase 5: Predicción Temporal 2026](#fase-5-predicción-temporal-2026)
- [Fase 6: Integración API](#fase-6-integración-api)
- [Fase 7: Visualización y Dashboard](#fase-7-visualización-y-dashboard)
- [Fase 8: Testing y Documentación](#fase-8-testing-y-documentación)
- [Fase 9: Deployment](#fase-9-deployment)

---

## FASE 0: CONFIGURACIÓN INICIAL
**Duración**: 1 día (pero se puede hacer en 30-60 minutos)  
**Objetivo**: Configurar entorno de desarrollo y obtener accesos

---

### 0.1 Registro de APIs (15 minutos) ⚠️ CRÍTICO

#### 0.1.1 NASA FIRMS (OBLIGATORIO - sin esto no hay datos)
- [ ] **Paso 1**: Abrir en navegador: https://firms.modaps.eosdis.nasa.gov/api/area/
- [ ] **Paso 2**: Click en botón **"Request Key"** (esquina superior derecha)
- [ ] **Paso 3**: Completar formulario:
  ```
  Email: tu@email.com
  First Name: Tu Nombre
  Last Name: Tu Apellido
  Purpose: Wildfire prediction research - Guatemala
  ```
- [ ] **Paso 4**: Click en **"Submit"**
- [ ] **Paso 5**: Revisar tu email (llega en 5-10 minutos)
- [ ] **Paso 6**: Copiar el **MAP_KEY** (formato: `a1b2c3d4-e5f6-7890-abcd-ef1234567890`)
- [ ] **Paso 7**: Guardar temporalmente en un archivo de texto (lo usaremos en 0.4)
- [ ] ✅ Verificación: Key debe tener ~36 caracteres con guiones

#### 0.1.2 Hugging Face (OBLIGATORIO - para fine-tuning)
- [ ] **Paso 1**: Abrir: https://huggingface.co/join
- [ ] **Paso 2**: Crear cuenta (email + contraseña)
- [ ] **Paso 3**: Verificar email (revisar bandeja de entrada)
- [ ] **Paso 4**: Login en: https://huggingface.co/login
- [ ] **Paso 5**: Ir a: https://huggingface.co/settings/tokens
- [ ] **Paso 6**: Click **"New token"**
  ```
  Name: train-llm-guatemala
  Role: Write
  ```
- [ ] **Paso 7**: Click **"Generate token"**
- [ ] **Paso 8**: Copiar token (formato: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx`)
- [ ] **Paso 9**: Guardar temporalmente (usaremos en 0.4)
- [ ] ✅ Verificación: Token empieza con `hf_` y tiene ~37 caracteres

#### 0.1.3 Google Earth Engine (OPCIONAL - no bloqueante)
- [ ] **Paso 1**: Abrir: https://signup.earthengine.google.com/
- [ ] **Paso 2**: Registrarse con cuenta Google (personal o institucional)
- [ ] **Paso 3**: Completar formulario de uso
- [ ] **Paso 4**: Esperar aprobación por email (24-48 horas)
- [ ] **Paso 5**: Mientras tanto, seguir con Fase 1 usando datos FIRMS
- [ ] ⚠️ NOTA: Podemos usar Landsat directo de USGS si Earth Engine tarda mucho

---

### 0.2 Configurar Google Colab (10 minutos)

#### 0.2.1 Verificar Cuenta Google
- [ ] Abrir: https://colab.research.google.com/
- [ ] Login con cuenta Google
- [ ] Aceptar términos y condiciones

#### 0.2.2 Verificar GPU Disponible
- [ ] Crear nuevo notebook (File → New notebook)
- [ ] Ejecutar en celda: `!nvidia-smi`
- [ ] Verificar output:
  - [ ] Si dice **"Tesla T4"**: ✅ Perfecto para Fases 1-3
  - [ ] Si dice **"No GPU found"**: Runtime → Change runtime type → GPU (T4)
  - [ ] Si necesitas **A100**: Contratar Colab Pro+ ($49.99/mes)

#### 0.2.3 Conectar Google Drive
- [ ] En celda de Colab, ejecutar:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
- [ ] Autorizar acceso (click en link, elegir cuenta, permitir)
- [ ] Verificar montaje: `!ls /content/drive/MyDrive`
- [ ] ✅ Deberías ver tus carpetas de Drive

#### 0.2.4 Decisión: ¿Colab Pro+ o Gratuito?
- [ ] **Opción A: Gratuito (Recomendado para empezar)**
  - GPU T4 (16GB VRAM)
  - Límite: 12 horas sesión
  - Fine-tuning lento (~2-3 días Fase 4)
  - Costo: $0
  
- [ ] **Opción B: Colab Pro+ (Para fine-tuning rápido)**
  - GPU A100 (40GB VRAM)
  - Límite: 24 horas sesión
  - Fine-tuning rápido (~4-6 horas Fase 4)
  - Costo: $49.99/mes (cancelar después de Fase 4)

- [ ] Mi decisión: ______________ (marcar arriba)

---

### 0.3 Crear Estructura de Directorios (2 minutos) ✅ COMPLETADO

#### 0.3.1 Ejecutar Comando de Creación
- [x] Abrir terminal en raíz del proyecto:
  ```bash
  cd /Users/richardortiz/workspace/Learning/curso_inteligenciaartificial2026/ejercicios_agentes/proyecto_incendios
  ```

- [x] Ejecutar comando único (crea todo de una vez):
  ```bash
  mkdir -p Docs notebooks data/{raw,processed,outputs} models visualizations scripts api data/raw/landsat_images data/processed/images_rgb_512 visualizations/maps
  ```

- [x] Crear archivos .gitkeep para preservar carpetas vacías:
  ```bash
  touch data/raw/.gitkeep data/processed/.gitkeep data/outputs/.gitkeep models/.gitkeep visualizations/.gitkeep visualizations/maps/.gitkeep
  ```

#### 0.3.2 Verificar Estructura Creada
- [x] Ejecutar: `tree -L 2 -d` (o `ls -R` si no tienes tree)
- [x] Deberías ver:
  ```
  proyecto_incendios/
  ├── Docs/              ✅ (ya tiene INFORME.md, PLAN.md, etc.)
  ├── notebooks/         ✅ (vacía por ahora)
  ├── data/
  │   ├── raw/          ✅
  │   ├── processed/    ✅
  │   └── outputs/      ✅
  ├── models/           ✅
  ├── visualizations/
  │   └── maps/         ✅
  ├── scripts/          ✅
  └── api/              ✅
  ```

#### 0.3.3 Estructura Completa Esperada
```
proyecto_incendios/
├── Docs/
│   ├── INFORME.md               ✅ Creado (33KB - Blueprint técnico)
│   ├── PLAN.md                  ✅ Creado (31KB - Este checklist)
│   ├── GETTING_STARTED.md       ✅ Creado (Guía rápida)
│   ├── PROGRESS.md              ✅ Creado (Tracker de progreso)
│   └── RESULTADOS.md            ⏳ Crear al final (Fase 8)
│
├── notebooks/
│   ├── Incendios.ipynb          ✅ Creado (Notebook 01: Data Download)
│   ├── 02_statistical_analysis.ipynb    ⏳ Fase 3
│   ├── 03_finetuning_unsloth.ipynb      ⏳ Fase 4
│   ├── 04_forecast_2026.ipynb           ⏳ Fase 5
│   └── 05_integration_api.ipynb         ⏳ Fase 6
│
├── data/
│   ├── raw/                               ⏳ Llenar en Fase 1
│   │   ├── nasa_firms_guatemala_2014_2024_combined.csv
│   │   └── landsat_images/
│   ├── processed/                         ⏳ Llenar en Fase 2
│   │   ├── fires_cleaned.csv
│   │   ├── fire_dataset.jsonl
│   │   └── images_rgb_512/
│   └── outputs/                           ⏳ Llenar en Fase 5
│       ├── calendario_incendios_2026.csv
│       └── predictions/
│
├── models/                                ⏳ Llenar en Fases 3-4
│   ├── random_forest_v2.pkl
│   └── qwen3-vl-guatemala-fires/
│
├── visualizations/                        ⏳ Llenar en Fases 1,3,5
│   ├── exploratory_analysis.png
│   ├── correlation_matrix_pearson.png
│   ├── feature_importance.png
│   ├── monthly_seasonality.png
│   ├── forecast_2026.png
│   └── maps/
│       └── heatmap_guatemala.html
│
├── scripts/                               ⏳ Crear según necesidad
│   ├── download_firms.py
│   ├── download_landsat.py
│   ├── process_images.py
│   ├── generate_dataset.py
│   └── utils.py
│
├── api/                                   ⏳ Fase 6
│   ├── paso1_datos_y_modelo.py    (existente - mejorar)
│   ├── paso2_llm_vision.py        (nuevo)
│   ├── paso3_api_fastapi.py       (existente - extender)
│   └── paso4_dashboard.py         (nuevo)
│
├── .env                           ✅ Creado (actualizar en 0.4)
├── .gitignore                     ✅ Creado
├── requirements.txt               ✅ Creado
└── README.md                      ✅ Creado
```

- [x] ✅ **VERIFICACIÓN**: Todos los directorios principales existen

---

### 0.4 Configurar archivo .env (5 minutos) ⚠️ CRÍTICO

#### 0.4.1 Verificar que .env existe
- [x] El archivo `.env` ya fue creado en raíz del proyecto
- [x] Ubicación: `/proyecto_incendios/.env`

#### 0.4.2 Editar .env con tus API Keys
- [ ] Abrir archivo `.env` con tu editor preferido:
  ```bash
  # Opción 1: VSCode
  code .env
  
  # Opción 2: Vim
  vim .env
  
  # Opción 3: Cualquier editor de texto
  open .env  # macOS
  ```

#### 0.4.3 Actualizar API Keys (REEMPLAZAR PLACEHOLDERS)
- [ ] Buscar esta línea:
  ```env
  FIRMS_MAP_KEY=TU_MAP_KEY_AQUI
  ```
- [ ] Reemplazar con tu MAP_KEY de NASA FIRMS (obtenida en 0.1.1):
  ```env
  FIRMS_MAP_KEY=a1b2c3d4-e5f6-7890-abcd-ef1234567890
  ```

- [ ] Buscar esta línea:
  ```env
  HF_TOKEN=hf_TU_TOKEN_AQUI
  ```
- [ ] Reemplazar con tu token de Hugging Face (obtenido en 0.1.2):
  ```env
  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  ```

- [ ] Guardar archivo (Ctrl+S / Cmd+S)

#### 0.4.4 Verificar .env Completo
- [ ] Tu archivo `.env` debería verse así (con TUS keys reales):
  ```env
  # NASA FIRMS API Key
  FIRMS_MAP_KEY=a1b2c3d4-e5f6-7890-abcd-ef1234567890  # ✅ TU KEY AQUÍ
  
  # Hugging Face Token
  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx           # ✅ TU TOKEN AQUÍ
  
  # Configuración del Proyecto
  PROJECT_NAME=train-llm-guatemala-fires
  COUNTRY_CODE=GTM
  START_DATE=2014-01-01
  END_DATE=2024-12-31
  
  # GPU/Device
  DEVICE=cuda
  
  # Guatemala Bounding Box (NO MODIFICAR)
  GUATEMALA_LAT_MIN=13.74
  GUATEMALA_LAT_MAX=17.82
  GUATEMALA_LON_MIN=-92.23
  GUATEMALA_LON_MAX=-88.23
  
  # Configuración de Modelos
  MIN_CONFIDENCE=50
  RANDOM_SEED=42
  ```

#### 0.4.5 Seguridad del .env
- [x] Verificar que `.gitignore` contiene `.env` (YA ESTÁ CONFIGURADO)
- [ ] ⚠️ **NUNCA** subir `.env` a GitHub/repositorios públicos
- [ ] ⚠️ **NUNCA** compartir tus API keys públicamente
- [ ] Si necesitas compartir el proyecto, crea `.env.example` con placeholders

#### 0.4.6 Probar que .env funciona
- [ ] Crear archivo temporal `test_env.py`:
  ```python
  from dotenv import load_dotenv
  import os
  
  load_dotenv()
  
  firms_key = os.getenv('FIRMS_MAP_KEY')
  hf_token = os.getenv('HF_TOKEN')
  
  print(f"FIRMS_MAP_KEY: {firms_key[:10]}... (OK)" if firms_key and firms_key != "TU_MAP_KEY_AQUI" else "❌ FIRMS_MAP_KEY no configurada")
  print(f"HF_TOKEN: {hf_token[:10]}... (OK)" if hf_token and hf_token != "hf_TU_TOKEN_AQUI" else "❌ HF_TOKEN no configurado")
  ```
- [ ] Ejecutar: `python test_env.py`
- [ ] Deberías ver:
  ```
  FIRMS_MAP_KEY: a1b2c3d4-e... (OK)
  HF_TOKEN: hf_xxxxxxx... (OK)
  ```
- [ ] Si ves "❌", revisa que reemplazaste los placeholders correctamente
- [ ] Borrar `test_env.py` después de verificar

- [ ] ✅ **VERIFICACIÓN FINAL**: Ambas keys están configuradas correctamente
```env
# NASA FIRMS
FIRMS_MAP_KEY=tu_key_aqui

# Hugging Face
HF_TOKEN=hf_tu_token_aqui

# Google Earth Engine (autenticación por OAuth)

# Configuración
PROJECT_NAME=train-llm-guatemala-fires
DEVICE=cuda
```

---

### 0.5 Instalar Dependencias (10-15 minutos)

#### 0.5.1 Verificar requirements.txt
- [x] El archivo `requirements.txt` ya fue creado
- [x] Contiene todas las dependencias necesarias (~50 paquetes)

#### 0.5.2 Decidir: ¿Instalación Local o solo Colab?

**Opción A: Solo Colab (Recomendado para empezar)**
- [ ] Saltar a 0.5.5 (instalación en Colab)
- Ventaja: No ensucias tu sistema local
- Ventaja: Colab ya tiene muchas libs preinstaladas
- Desventaja: No puedes trabajar offline

**Opción B: Local + Colab (Desarrollo serio)**
- [ ] Continuar con 0.5.3 (entorno virtual local)
- Ventaja: Trabajas offline en tu Mac
- Ventaja: Mejor para desarrollo de scripts/API
- Desventaja: Requiere ~2GB espacio + 15 min setup

- [ ] Mi decisión: ______________ (marcar arriba)

#### 0.5.3 Instalación Local (OPCIONAL - solo si elegiste Opción B)

**Crear Entorno Virtual**
- [ ] Abrir terminal en raíz del proyecto:
  ```bash
  cd proyecto_incendios
  ```

- [ ] Crear entorno virtual:
  ```bash
  # Python 3.10+ recomendado
  python3 -m venv venv
  ```

- [ ] Activar entorno virtual:
  ```bash
  # macOS/Linux:
  source venv/bin/activate
  
  # Windows:
  venv\Scripts\activate
  ```

- [ ] Verificar activación (debería aparecer `(venv)` en el prompt):
  ```bash
  which python  # Debe apuntar a ./venv/bin/python
  ```

**Instalar Dependencias Base**
- [ ] Actualizar pip:
  ```bash
  pip install --upgrade pip
  ```

- [ ] Instalar dependencias básicas (rápido, ~2 min):
  ```bash
  pip install requests pandas numpy matplotlib seaborn plotly folium tqdm python-dotenv
  ```

- [ ] Verificar instalación:
  ```bash
  python -c "import pandas; import requests; print('✅ OK')"
  ```

**Instalar PyTorch (según tu sistema)**
- [ ] Verificar si tienes GPU NVIDIA:
  ```bash
  # macOS: No tiene CUDA, usa CPU
  pip install torch torchvision torchaudio
  
  # Linux con CUDA 12.1:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  
  # Windows con CUDA 12.1:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

- [ ] Verificar PyTorch:
  ```bash
  python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
  ```

**Instalar resto de dependencias (lento, ~5-10 min)**
- [ ] Instalar todo (excepto Unsloth por ahora):
  ```bash
  pip install -r requirements.txt
  ```

- [ ] Si hay errores de compilación, instalar por grupos:
  ```bash
  # Data Science
  pip install scikit-learn scipy statsmodels prophet
  
  # Geospatial
  pip install geopandas rasterio folium pyproj
  
  # API
  pip install fastapi uvicorn streamlit gradio
  
  # ML
  pip install transformers datasets accelerate peft
  ```

**Instalar Unsloth (SOLO cuando vayas a hacer fine-tuning)**
- [ ] Ver CUDA version:
  ```bash
  python -c "import torch; print(torch.version.cuda)"
  ```

- [ ] Instalar según tu CUDA (ver INFORME.md para detalles):
  ```bash
  # Forma simple (auto-detecta):
  pip install unsloth
  
  # Forma avanzada (específica):
  pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
  ```

- [ ] ⚠️ NOTA: Unsloth es opcional para Fases 1-3. Solo necesario en Fase 4.

#### 0.5.4 Verificación de Instalación Local
- [ ] Crear archivo `test_installation.py`:
  ```python
  import sys
  
  packages = {
      'pandas': 'Data manipulation',
      'numpy': 'Numerical computing',
      'matplotlib': 'Plotting',
      'seaborn': 'Statistical viz',
      'requests': 'HTTP requests',
      'folium': 'Maps',
      'torch': 'PyTorch',
      'transformers': 'Hugging Face',
  }
  
  print("Verificando instalación...")
  for pkg, desc in packages.items():
      try:
          __import__(pkg)
          print(f"✅ {pkg:15s} - {desc}")
      except ImportError:
          print(f"❌ {pkg:15s} - FALTA")
  
  print(f"\n✅ Python {sys.version}")
  ```

- [ ] Ejecutar: `python test_installation.py`
- [ ] Deberías ver todos ✅
- [ ] Si hay ❌, reinstalar ese paquete: `pip install <paquete>`

#### 0.5.5 Instalación en Google Colab (SIEMPRE necesario)

**En tu Notebook de Colab (`Incendios.ipynb`)**
- [ ] Primera celda ya tiene instalación:
  ```python
  !pip install -q requests pandas numpy matplotlib seaborn plotly folium tqdm
  ```

- [ ] Ejecutar esa celda (tarda ~30 segundos)

- [ ] Cuando llegues a Fase 4 (fine-tuning), agregar celda:
  ```python
  !pip install -q unsloth xformers triton
  ```

- [ ] ⚠️ IMPORTANTE: En Colab NO necesitas entorno virtual
- [ ] ⚠️ IMPORTANTE: Las instalaciones en Colab se pierden al cerrar sesión

#### 0.5.6 Verificación Final
- [ ] Local (si instalaste):
  - [ ] `python test_installation.py` → todos ✅
  - [ ] Entorno virtual activado para trabajar
  
- [ ] Colab:
  - [ ] Celda de instalación ejecutada sin errores
  - [ ] `import pandas, numpy, matplotlib` funciona

- [ ] ✅ **CHECKPOINT**: Dependencias listas para Fase 1

---

### 0.6 Ejecutar Notebook 01: Data Download (15-30 minutos) 🚀

#### 0.6.1 Verificar Pre-requisitos
- [ ] API Keys configuradas en `.env` (0.4 completado)
- [ ] Google Colab configurado (0.2 completado)
- [ ] Notebook `Incendios.ipynb` existe en carpeta raíz

#### 0.6.2 Opción A: Ejecutar en Google Colab (RECOMENDADO)

**Subir Notebook a Colab**
- [ ] Abrir: https://colab.research.google.com/
- [ ] Click **"File"** → **"Upload notebook"**
- [ ] Seleccionar: `proyecto_incendios/Incendios.ipynb`
- [ ] Esperar a que cargue (~5 segundos)

**Configurar Runtime**
- [ ] Click **"Runtime"** → **"Change runtime type"**
- [ ] Seleccionar:
  - Hardware accelerator: **GPU**
  - GPU type: **T4** (gratis) o **A100** (si tienes Pro+)
  - Runtime shape: Standard
- [ ] Click **"Save"**

**Conectar Google Drive**
- [ ] Ejecutar celda que dice `drive.mount('/content/drive')`
- [ ] Click en link de autorización
- [ ] Seleccionar cuenta Google
- [ ] Click **"Allow"**
- [ ] Verificar mensaje: "Mounted at /content/drive"

**Configurar API Key en Notebook**
- [ ] Buscar celda con:
  ```python
  FIRMS_MAP_KEY = "TU_MAP_KEY_AQUI"
  ```
- [ ] Reemplazar con tu MAP_KEY real (de 0.1.1):
  ```python
  FIRMS_MAP_KEY = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"  # TU KEY
  ```
- [ ] ⚠️ CRÍTICO: Sin este paso, la descarga FALLARÁ

**Ejecutar Todo el Notebook**
- [ ] Click **"Runtime"** → **"Run all"**
- [ ] Esperar ejecución (~10-15 minutos total)
- [ ] Monitorear progreso en cada celda

**Verificar Descarga Exitosa**
- [ ] Ver output de celda "Descarga NASA FIRMS":
  ```
  ✓ Descargado: XX,XXX detecciones
  ✓ Período: 2014-01-01 a 2024-12-31
  ```
- [ ] Ver gráficas generadas (4 plots)
- [ ] Ver mapa de calor interactivo (Folium)
- [ ] Verificar mensaje final:
  ```
  ✅ FASE 1 COMPLETADA: Datos descargados, explorados y limpiados
  ```

**Verificar Archivos en Drive**
- [ ] Abrir Google Drive en navegador
- [ ] Navegar a: `MyDrive/train_llm_fires/`
- [ ] Verificar estructura:
  ```
  train_llm_fires/
  ├── data/
  │   ├── raw/
  │   │   └── nasa_firms_guatemala_2014_2024_combined.csv  (✅ existe)
  │   └── processed/
  │       └── fires_cleaned.csv  (✅ existe)
  └── visualizations/
      ├── exploratory_analysis.png  (✅ existe)
      └── heatmap_guatemala.html    (✅ existe)
  ```

**Descargar Datos a Local (OPCIONAL)**
- [ ] En Google Drive, click derecho en `train_llm_fires/`
- [ ] Seleccionar **"Download"**
- [ ] Esperar descarga (puede tardar según tamaño)
- [ ] Mover a `proyecto_incendios/data/` local

#### 0.6.3 Opción B: Ejecutar en Jupyter Local (ALTERNATIVA)

**SOLO si instalaste todo localmente en 0.5.3**

- [ ] Activar entorno virtual:
  ```bash
  source venv/bin/activate  # o venv\Scripts\activate en Windows
  ```

- [ ] Iniciar Jupyter:
  ```bash
  jupyter notebook Incendios.ipynb
  ```

- [ ] Se abrirá en navegador
- [ ] Ejecutar celdas una por una
- [ ] ⚠️ NOTA: Algunas celdas usan `drive.mount()` que solo funciona en Colab
- [ ] Modificar rutas de guardado para que sean locales:
  ```python
  # En Colab:
  '/content/drive/MyDrive/train_llm_fires/data/raw/...'
  
  # En local:
  './data/raw/...'
  ```

#### 0.6.4 Troubleshooting Común

**Error: "Invalid MAP_KEY"**
- [ ] Verificar que copiaste la key completa (sin espacios)
- [ ] Verificar que esperaste email de confirmación de NASA
- [ ] Probar key en navegador: `https://firms.modaps.eosdis.nasa.gov/api/area/csv/TU_KEY/VIIRS_SNPP_NRT/country/GTM/1/2024-01-01`

**Error: "No GPU found"**
- [ ] Runtime → Change runtime type → GPU (T4)
- [ ] Restart runtime
- [ ] Ejecutar `!nvidia-smi` de nuevo

**Error: "Drive not mounted"**
- [ ] Ejecutar celda `drive.mount()` manualmente
- [ ] Autorizar acceso
- [ ] Si persiste, reiniciar runtime

**Descarga muy lenta o timeout**
- [ ] Es normal, puede tardar 5-10 min
- [ ] Verificar conexión a internet
- [ ] Si timeout, ejecutar celda de descarga individualmente (no "Run all")

**No se generan gráficas**
- [ ] Verificar que hay datos descargados
- [ ] Ejecutar celdas de visualización manualmente
- [ ] Verificar imports: `matplotlib`, `seaborn`, `folium`

#### 0.6.5 Outputs Esperados al Finalizar

**Datos Descargados**
- [ ] VIIRS SNPP: ~XX,XXX detecciones
- [ ] MODIS: ~XX,XXX detecciones  
- [ ] VIIRS NOAA-20: ~XX,XXX detecciones
- [ ] **Total combinado**: ~50,000 - 200,000 detecciones (depende de actividad real)

**Archivos Generados**
- [ ] `nasa_firms_guatemala_2014_2024_combined.csv` (~50-100 MB)
- [ ] `fires_cleaned.csv` (~40-80 MB, después de filtros)
- [ ] `exploratory_analysis.png` (4 gráficas)
- [ ] `heatmap_guatemala.html` (mapa interactivo)

**Estadísticas Clave**
- [ ] Período: 2014-01-01 a 2024-12-31
- [ ] Coordenadas: Dentro de Guatemala (lat 13.74-17.82, lon -92.23 - -88.23)
- [ ] Confianza: ≥50% (filtrado aplicado)
- [ ] Features temporales: año, mes, día, hora creados
- [ ] Duplicados: Removidos

#### 0.6.6 Registro de Progreso

- [ ] Actualizar archivo `Docs/PROGRESS.md`:
  ```markdown
  ## ✅ FASE 0: COMPLETADA
  - [x] Todas las sub-tareas marcadas
  
  ## ✅ FASE 1: COMPLETADA (Parcial - Notebook 01)
  - [x] 1.1 Datos NASA FIRMS descargados
  - [x] Exploración inicial realizada
  - [x] Visualizaciones generadas
  - [x] Dataset limpio guardado
  ```

- [ ] Marcar en PLAN.md:
  ```
  ✅ FASE 0: COMPLETADA (100%)
  ✅ FASE 1: EN PROGRESO (30% - Notebook 01 ejecutado)
  ```

#### 0.6.7 Checkpoint Final Fase 0

**Antes de continuar a Fase 1 (resto de tareas), verificar:**
- [ ] ✅ API Keys configuradas y funcionando
- [ ] ✅ Google Colab funcional con GPU T4
- [ ] ✅ Google Drive conectado
- [ ] ✅ Notebook 01 ejecutado sin errores
- [ ] ✅ Datos NASA FIRMS descargados (~50k+ detecciones)
- [ ] ✅ Archivos en Drive: CSV + gráficas + mapa
- [ ] ✅ Estructura de proyecto completa

**Si todos los ✅ están marcados**:
- [ ] 🎉 **FASE 0 COMPLETADA AL 100%**
- [ ] 📊 Progreso global: **12% del proyecto** (Fase 0 + parte de Fase 1)
- [ ] ⏭️ **SIGUIENTE**: Continuar con FASE 1.2 (Descargar imágenes Landsat) o saltar a FASE 2 si quieres procesar primero los datos FIRMS

**Si hay ❌**:
- [ ] Revisar sección Troubleshooting (0.6.4)
- [ ] Pedir ayuda pegando el error específico
- [ ] No avanzar hasta resolver problemas

---

## ✅ RESUMEN FASE 0

**Completado**:
- ✅ Registro en NASA FIRMS, Hugging Face, Google Earth Engine
- ✅ Google Colab configurado con GPU
- ✅ Estructura de directorios creada (13 carpetas)
- ✅ Archivos de configuración (.env, .gitignore, requirements.txt)
- ✅ Documentación creada (INFORME.md, PLAN.md, README.md, PROGRESS.md)
- ✅ Notebook 01 creado y ejecutado
- ✅ Datos NASA FIRMS descargados y procesados
- ✅ Visualizaciones exploratorias generadas

**Archivos en el proyecto** (9 creados):
```
✅ Docs/INFORME.md (33KB)
✅ Docs/PLAN.md (este archivo, ~40KB actualizado)
✅ Docs/GETTING_STARTED.md
✅ Docs/PROGRESS.md
✅ Incendios.ipynb (Notebook 01 completo)
✅ .env (configurado con API keys)
✅ .gitignore
✅ requirements.txt
✅ README.md
```

**Tiempo invertido**: ~1-2 horas (vs. 1 día estimado original) ✅

**Próximo paso**: FASE 1.2 o FASE 2 (a tu elección)

---

## FASE 1: ADQUISICIÓN DE DATOS
**Duración**: 5 días  
**Objetivo**: Descargar datos históricos NASA FIRMS + imágenes satelitales

### 1.1 Descargar Datos NASA FIRMS (2014-2024)
**Script**: `scripts/download_firms.py`

- [ ] Crear script de descarga:
```python
import requests
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
MAP_KEY = os.getenv('FIRMS_MAP_KEY')

def download_firms_data(country='GTM', days=10, sensor='VIIRS_SNPP_NRT'):
    """
    Descarga datos FIRMS para Guatemala
    country: GTM (Guatemala)
    days: 10 = últimos 10 años (máximo permitido)
    sensor: VIIRS_SNPP_NRT, MODIS_NRT, etc.
    """
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{sensor}/country/{country}/{days}/2014-01-01"
    
    print(f"Descargando datos FIRMS...")
    response = requests.get(url, timeout=300)
    
    if response.status_code == 200:
        output_file = f'data/raw/nasa_firms_{country}_{sensor}_2014_2024.csv'
        with open(output_file, 'w') as f:
            f.write(response.text)
        
        df = pd.read_csv(output_file)
        print(f"✓ Descargado: {len(df):,} detecciones")
        print(f"✓ Guardado en: {output_file}")
        return df
    else:
        print(f"✗ Error {response.status_code}: {response.text}")
        return None

if __name__ == "__main__":
    # Descargar VIIRS (mejor resolución)
    df_viirs = download_firms_data(sensor='VIIRS_SNPP_NRT')
    
    # Descargar MODIS (mayor cobertura histórica)
    df_modis = download_firms_data(sensor='MODIS_NRT')
    
    # Combinar datasets
    df_combined = pd.concat([df_viirs, df_modis], ignore_index=True)
    df_combined.to_csv('data/raw/nasa_firms_guatemala_2014_2024_combined.csv', index=False)
    print(f"\n✓ Total combinado: {len(df_combined):,} detecciones")
```

- [ ] Ejecutar: `python scripts/download_firms.py`
- [ ] Verificar archivo generado: `data/raw/nasa_firms_guatemala_2014_2024_combined.csv`
- [ ] Inspeccionar datos:
  - [ ] Columnas: `latitude, longitude, brightness, scan, track, acq_date, acq_time, satellite, confidence, version, bright_t31, frp, daynight`
  - [ ] Rango de fechas: 2014-01-01 a 2024-12-31
  - [ ] Coordenadas dentro de Guatemala: lat [13.74, 17.82], lon [-92.23, -88.23]

### 1.2 Descargar Imágenes Landsat 8/9 (Google Earth Engine)
**Notebook**: `notebooks/01_data_download.ipynb`

- [ ] Abrir Google Colab
- [ ] Autenticarse en Earth Engine:
```python
import ee
ee.Authenticate()  # Seguir instrucciones en pantalla
ee.Initialize()
```

- [ ] Definir área de interés (Guatemala):
```python
guatemala = ee.Geometry.Rectangle([-92.23, 13.74, -88.23, 17.82])
```

- [ ] Filtrar colección Landsat 8:
```python
landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
    .filterBounds(guatemala) \
    .filterDate('2014-01-01', '2024-12-31') \
    .filter(ee.Filter.lt('CLOUD_COVER', 20))  # < 20% nubes

print(f"Imágenes disponibles: {landsat.size().getInfo()}")
```

- [ ] Descargar muestra de 500 imágenes:
```python
# Crear función de exportación
def export_image(image, index):
    task = ee.batch.Export.image.toDrive(
        image=image.select(['SR_B4', 'SR_B3', 'SR_B2']),  # RGB
        description=f'landsat_guatemala_{index}',
        folder='landsat_guatemala',
        scale=30,  # 30m resolución
        region=guatemala,
        maxPixels=1e13
    )
    task.start()
    return task

# Exportar primeras 500 imágenes
image_list = landsat.toList(500)
tasks = []
for i in range(500):
    image = ee.Image(image_list.get(i))
    task = export_image(image, i)
    tasks.append(task)
    print(f"Exportando imagen {i+1}/500...")
```

- [ ] Esperar descarga en Google Drive (puede tomar 1-2 días)
- [ ] Descargar desde Drive a `data/raw/landsat_images/`

**Alternativa más rápida**: Usar imágenes pre-procesadas de datasets públicos
- [ ] Buscar en Earth Engine Data Catalog: https://developers.google.com/earth-engine/datasets
- [ ] Usar composites mensuales (menos volumen de datos)

### 1.3 Descargar Datos Meteorológicos
**Fuente**: WorldClim / NOAA

- [ ] WorldClim (climatología histórica):
```python
import requests

# Descargar temperatura promedio mensual (1970-2000 baseline)
url_temp = "https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_30s_tavg.zip"
response = requests.get(url_temp)
with open('data/raw/worldclim_temp.zip', 'wb') as f:
    f.write(response.content)

# Similarmente para precipitación, humedad, viento
```

- [ ] Verificar archivos descargados en `data/raw/`

---

## FASE 2: PROCESAMIENTO DE DATOS
**Duración**: 5 días  
**Objetivo**: Limpiar, normalizar y generar dataset anotado

### 2.1 Limpieza de Datos NASA FIRMS
**Script**: `scripts/process_firms.py`

- [ ] Cargar datos:
```python
import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/nasa_firms_guatemala_2014_2024_combined.csv')
```

- [ ] Filtrar detecciones en Guatemala:
```python
# Bounding box Guatemala
df = df[
    (df['latitude'] >= 13.74) & (df['latitude'] <= 17.82) &
    (df['longitude'] >= -92.23) & (df['longitude'] <= -88.23)
]
```

- [ ] Remover duplicados:
```python
df = df.drop_duplicates(subset=['latitude', 'longitude', 'acq_date', 'acq_time'])
```

- [ ] Filtrar confianza mínima:
```python
df = df[df['confidence'] >= 50]  # Solo detecciones con confianza ≥50%
```

- [ ] Crear columna datetime:
```python
df['datetime'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'], format='%Y-%m-%d %H%M')
```

- [ ] Agregar departamento (región administrativa):
```python
import geopandas as gpd

# Descargar shapefile Guatemala
guatemala_shp = gpd.read_file('https://geodata.example.com/guatemala_departments.shp')

# Spatial join
gdf_fires = gpd.GeoDataFrame(
    df, 
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs='EPSG:4326'
)
df = gpd.sjoin(gdf_fires, guatemala_shp, how='left', predicate='within')
```

- [ ] Guardar datos limpios:
```python
df.to_csv('data/processed/fires_cleaned.csv', index=False)
print(f"✓ Datos procesados: {len(df):,} detecciones")
```

### 2.2 Procesamiento de Imágenes Landsat
**Script**: `scripts/process_images.py`

- [ ] Convertir GeoTIFF a RGB PNG (512x512):
```python
import rasterio
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

def geotiff_to_rgb(input_path, output_path, size=(512, 512)):
    """
    Convierte GeoTIFF multiespectral a RGB normalizado
    """
    with rasterio.open(input_path) as src:
        # Leer bandas RGB (4=Red, 3=Green, 2=Blue en Landsat 8)
        red = src.read(4)
        green = src.read(3)
        blue = src.read(2)
        
        # Stack y normalizar
        rgb = np.dstack((red, green, blue))
        rgb_norm = ((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255).astype(np.uint8)
        
        # Resize
        img = Image.fromarray(rgb_norm)
        img = img.resize(size, Image.LANCZOS)
        
        # Guardar
        img.save(output_path, quality=95)

# Procesar todas las imágenes
input_dir = 'data/raw/landsat_images/'
output_dir = 'data/processed/images_rgb_512/'

for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith('.tif'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace('.tif', '.png'))
        geotiff_to_rgb(input_path, output_path)

print(f"✓ Procesadas {len(os.listdir(output_dir))} imágenes")
```

- [ ] Ejecutar: `python scripts/process_images.py`
- [ ] Verificar imágenes en `data/processed/images_rgb_512/`

### 2.3 Generación de Dataset Anotado
**Script**: `scripts/generate_dataset.py`

- [ ] Crear pares imagen-texto para fine-tuning:
```python
import pandas as pd
import json
from pathlib import Path

df_fires = pd.read_csv('data/processed/fires_cleaned.csv')
image_dir = Path('data/processed/images_rgb_512/')

dataset = []

for idx, row in df_fires.iterrows():
    # Buscar imagen correspondiente (por fecha y coordenadas)
    image_date = row['acq_date']
    lat, lon = row['latitude'], row['longitude']
    
    # Encontrar imagen más cercana
    image_path = find_closest_image(image_date, lat, lon, image_dir)
    
    if image_path:
        # Generar descripción
        description = f"""
Análisis de imagen satelital de Guatemala:
- Ubicación: {lat:.2f}°N, {lon:.2f}°W ({row['department']})
- Fecha: {row['acq_date']} a las {row['acq_time']}
- Detección de incendio: {"SÍ" if row['confidence'] >= 80 else "PROBABLE"}
- Temperatura de superficie: {row['brightness']}K
- Fire Radiative Power: {row['frp']} MW
- Confianza: {row['confidence']}%
- Sensor: {row['satellite']}
- Cobertura terrestre: Bosque tropical
- Riesgo de propagación: {"ALTO" if row['frp'] > 100 else "MEDIO" if row['frp'] > 50 else "BAJO"}
        """.strip()
        
        # Formato Unsloth
        sample = {
            "image": str(image_path),
            "conversations": [
                {
                    "from": "human",
                    "value": "Analiza esta imagen satelital de Guatemala. ¿Hay evidencia de incendio forestal? Proporciona detalles."
                },
                {
                    "from": "gpt",
                    "value": description
                }
            ],
            "metadata": {
                "date": row['acq_date'],
                "lat": lat,
                "lon": lon,
                "department": row.get('department', 'Unknown'),
                "frp": row['frp'],
                "confidence": row['confidence']
            }
        }
        
        dataset.append(sample)

# Guardar en JSONL
with open('data/processed/fire_dataset.jsonl', 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')

print(f"✓ Dataset generado: {len(dataset)} muestras")
```

- [ ] Ejecutar: `python scripts/generate_dataset.py`
- [ ] Verificar: `data/processed/fire_dataset.jsonl`
- [ ] Objetivo: 500-1000 muestras mínimo

### 2.4 Split Train/Validation/Test
- [ ] Dividir dataset (70/15/15):
```python
from sklearn.model_selection import train_test_split
import json

with open('data/processed/fire_dataset.jsonl', 'r') as f:
    dataset = [json.loads(line) for line in f]

train, temp = train_test_split(dataset, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
    with open(f'data/processed/fire_dataset_{split_name}.jsonl', 'w') as f:
        for item in split_data:
            f.write(json.dumps(item) + '\n')
    print(f"{split_name}: {len(split_data)} muestras")
```

---

## FASE 3: ANÁLISIS ESTADÍSTICO
**Duración**: 2 días  
**Notebook**: `notebooks/02_statistical_analysis.ipynb`

### 3.1 Análisis de Correlación
- [ ] Cargar datos procesados
- [ ] Calcular correlación Pearson (ver INFORME.md sección "Análisis Estadístico")
- [ ] Calcular correlación Spearman
- [ ] Test de significancia (p-values)
- [ ] Generar gráfica: `visualizations/correlation_matrix_pearson.png`
- [ ] Generar gráfica: `visualizations/correlation_matrix_spearman.png`

### 3.2 Feature Importance (Random Forest)
- [ ] Entrenar Random Forest con datos actuales
- [ ] Extraer importancia de features
- [ ] Generar gráfica: `visualizations/feature_importance.png`
- [ ] Identificar top 5 features más predictivas

### 3.3 Análisis Temporal
- [ ] Agrupar incendios por mes (2014-2024)
- [ ] Identificar estacionalidad (pico marzo-mayo esperado)
- [ ] Generar gráfica: `visualizations/monthly_seasonality.png`
- [ ] Agrupar por año (tendencias)
- [ ] Generar gráfica: `visualizations/yearly_trends.png`

### 3.4 Análisis Espacial
- [ ] Agrupar incendios por departamento
- [ ] Calcular densidad (incendios/km²)
- [ ] Generar mapa de calor: `visualizations/maps/heatmap_guatemala.html` (Folium)
- [ ] Identificar departamentos de mayor riesgo

### 3.5 Tests Estadísticos
- [ ] Test de normalidad (Shapiro-Wilk)
- [ ] ANOVA (comparación grupos alto/bajo riesgo)
- [ ] Chi-cuadrado (asociación departamento vs. incendios)
- [ ] Guardar resultados: `Docs/statistical_tests_results.txt`

---

## FASE 4: FINE-TUNING LLM
**Duración**: 2 días (con Colab A100) o 5 días (con T4)  
**Notebook**: `notebooks/03_finetuning_unsloth.ipynb`

### 4.1 Configuración Inicial
- [ ] Abrir Google Colab
- [ ] Activar GPU: Runtime > Change runtime type > A100 (si tienes Pro+)
- [ ] Verificar GPU: `!nvidia-smi`
- [ ] Instalar Unsloth:
```python
!pip install unsloth
!pip install xformers triton
```

### 4.2 Cargar Modelo Base
- [ ] Cargar Qwen3-VL 7B cuantizado:
```python
from unsloth import FastVisionModel
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "unsloth/Qwen2-VL-7B-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
    dtype = torch.float16,
)
```

- [ ] Verificar VRAM usado: ~8-10GB (debería caber en T4/A100)

### 4.3 Configurar LoRA
- [ ] Aplicar adapters LoRA (ver INFORME.md):
```python
model = FastVisionModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0.05,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
```

### 4.4 Cargar Dataset
- [ ] Subir `fire_dataset_train.jsonl` a Colab
- [ ] Cargar con datasets library:
```python
from datasets import load_dataset

dataset = load_dataset('json', data_files={
    'train': 'fire_dataset_train.jsonl',
    'validation': 'fire_dataset_val.jsonl'
})
```

### 4.5 Entrenamiento
- [ ] Configurar trainer (ver INFORME.md sección "Fine-Tuning"):
```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset['train'],
    eval_dataset = dataset['validation'],
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = True,
        logging_steps = 10,
        optim = "adamw_8bit",
        output_dir = "outputs",
        evaluation_strategy = "steps",
        eval_steps = 50,
        save_strategy = "steps",
        save_steps = 100,
    ),
)
```

- [ ] Iniciar entrenamiento:
```python
trainer_stats = trainer.train()
```

- [ ] Monitorear pérdida (loss) - debería descender consistentemente
- [ ] Guardar checkpoints cada 100 steps

### 4.6 Evaluación
- [ ] Evaluar en test set:
```python
test_dataset = load_dataset('json', data_files='fire_dataset_test.jsonl')
results = trainer.evaluate(test_dataset)
print(results)
```

- [ ] Calcular métricas:
  - [ ] Accuracy
  - [ ] F1-Score
  - [ ] BLEU score (calidad texto generado)

### 4.7 Guardar Modelo
- [ ] Guardar localmente en Colab:
```python
model.save_pretrained("qwen3-vl-guatemala-fires")
tokenizer.save_pretrained("qwen3-vl-guatemala-fires")
```

- [ ] Descargar a Google Drive:
```python
!zip -r qwen3-vl-guatemala-fires.zip qwen3-vl-guatemala-fires
!cp qwen3-vl-guatemala-fires.zip /content/drive/MyDrive/
```

- [ ] Subir a Hugging Face Hub:
```python
model.push_to_hub("tu-usuario/qwen3-vl-guatemala-fires", token="hf_...")
tokenizer.push_to_hub("tu-usuario/qwen3-vl-guatemala-fires", token="hf_...")
```

---

## FASE 5: PREDICCIÓN TEMPORAL 2026
**Duración**: 2 días  
**Notebook**: `notebooks/04_forecast_2026.ipynb`

### 5.1 Preparar Datos para Serie Temporal
- [ ] Agrupar incendios por día (2014-2024):
```python
df_ts = df_fires.groupby('acq_date').size().reset_index(name='fire_count')
df_ts.columns = ['ds', 'y']  # Prophet requiere estas columnas
```

- [ ] Verificar datos faltantes (rellenar con 0)
- [ ] Crear features adicionales (temperatura, humedad, FWI)

### 5.2 Entrenar Modelo Prophet
- [ ] Instalar Prophet: `!pip install prophet`
- [ ] Configurar modelo (ver INFORME.md):
```python
from prophet import Prophet

model = Prophet(
    yearly_seasonality=True,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.05,
)
model.add_regressor('temp_max')
model.add_regressor('humidity')
model.fit(df_ts)
```

### 5.3 Generar Forecast 2026
- [ ] Crear dataframe futuro:
```python
future = model.make_future_dataframe(periods=365, freq='D')
future['temp_max'] = load_weather_forecast()  # Necesitás proyecciones climáticas
future['humidity'] = load_weather_forecast()
```

- [ ] Predecir:
```python
forecast = model.predict(future)
```

- [ ] Extraer predicciones 2026:
```python
forecast_2026 = forecast[forecast['ds'].dt.year == 2026]
```

### 5.4 Visualización
- [ ] Gráfica serie temporal completa:
```python
fig = model.plot(forecast)
plt.savefig('visualizations/forecast_2026.png', dpi=300)
```

- [ ] Gráfica componentes (tendencia, estacionalidad):
```python
fig = model.plot_components(forecast)
plt.savefig('visualizations/forecast_components.png', dpi=300)
```

### 5.5 Evaluación
- [ ] Usar 2024 como test set
- [ ] Calcular MAE, RMSE, MAPE
- [ ] Guardar métricas en `Docs/forecast_metrics.txt`

### 5.6 Exportar Calendario 2026
- [ ] Crear CSV con predicciones:
```python
calendar_2026 = forecast_2026[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
calendar_2026.columns = ['fecha', 'prediccion', 'limite_inferior', 'limite_superior']

# Agregar categoría de riesgo
def categorize_risk(value):
    if value < 10: return 'BAJO'
    elif value < 30: return 'MEDIO'
    elif value < 50: return 'ALTO'
    else: return 'MUY ALTO'

calendar_2026['riesgo'] = calendar_2026['prediccion'].apply(categorize_risk)
calendar_2026.to_csv('data/outputs/calendario_incendios_2026.csv', index=False)
```

- [ ] Verificar archivo: `data/outputs/calendario_incendios_2026.csv`

---

## FASE 6: INTEGRACIÓN API
**Duración**: 2 días  
**Script**: `api/paso3_api_fastapi.py` (extender existente)

### 6.1 Cargar Modelos
- [ ] Modificar `paso3_api_fastapi.py`:
```python
from fastapi import FastAPI, UploadFile, File
from unsloth import FastVisionModel
import torch
import pandas as pd

app = FastAPI()

# Cargar Random Forest (existente)
rf_model = joblib.load('models/random_forest_v2.pkl')

# Cargar LLM Vision (nuevo)
llm_model, llm_tokenizer = FastVisionModel.from_pretrained(
    "qwen3-vl-guatemala-fires",
    load_in_4bit=True
)

# Cargar calendario 2026
calendar_2026 = pd.read_csv('data/outputs/calendario_incendios_2026.csv')
```

### 6.2 Crear Nuevo Endpoint: /predict/vision
- [ ] Implementar análisis de imágenes con LLM:
```python
@app.post("/predict/vision")
async def predict_vision(image: UploadFile = File(...)):
    from PIL import Image
    import io
    
    # Cargar imagen
    image_data = Image.open(io.BytesIO(await image.read()))
    
    # Inference LLM
    prompt = "Analiza esta imagen satelital de Guatemala. ¿Hay evidencia de incendio forestal?"
    inputs = llm_tokenizer(prompt, images=image_data, return_tensors="pt")
    outputs = llm_model.generate(**inputs, max_new_tokens=200)
    response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"analysis": response}
```

- [ ] Probar endpoint: `curl -X POST -F "image=@test_image.jpg" http://localhost:8000/predict/vision`

### 6.3 Crear Nuevo Endpoint: /forecast/2026
- [ ] Devolver calendario completo:
```python
@app.get("/forecast/2026")
def get_forecast_2026():
    return calendar_2026.to_dict(orient='records')
```

- [ ] Probar: `curl http://localhost:8000/forecast/2026`

### 6.4 Crear Endpoint Ensemble: /predict/ensemble
- [ ] Combinar Random Forest + LLM Vision:
```python
@app.post("/predict/ensemble")
async def predict_ensemble(
    features: dict,  # Para Random Forest
    image: UploadFile = File(...)  # Para LLM
):
    # Predicción Random Forest
    rf_prob = rf_model.predict_proba([list(features.values())])[0][1]
    
    # Predicción LLM
    llm_analysis = await predict_vision(image)
    
    # Ensemble (weighted average)
    ensemble_score = 0.6 * rf_prob + 0.4 * (1 if "SÍ" in llm_analysis else 0)
    
    return {
        "random_forest_prob": rf_prob,
        "llm_analysis": llm_analysis,
        "ensemble_score": ensemble_score,
        "decision": "FIRE" if ensemble_score > 0.5 else "NO_FIRE"
    }
```

### 6.5 Documentación API (Swagger)
- [ ] Agregar descripciones a endpoints
- [ ] Verificar docs en: `http://localhost:8000/docs`

### 6.6 Testing Local
- [ ] Ejecutar servidor: `uvicorn api.paso3_api_fastapi:app --reload`
- [ ] Probar todos los endpoints
- [ ] Verificar logs y errores

---

## FASE 7: VISUALIZACIÓN Y DASHBOARD
**Duración**: 3 días  
**Script**: `api/paso4_dashboard.py`

### 7.1 Crear Dashboard con Streamlit
- [ ] Crear archivo `api/paso4_dashboard.py`:
```python
import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Guatemala Fire Detection", layout="wide")

st.title("🔥 Sistema de Predicción de Incendios Forestales - Guatemala")
st.markdown("Análisis histórico 2014-2024 y predicciones 2026")

# Sidebar
st.sidebar.header("Navegación")
page = st.sidebar.radio("Selecciona sección:", 
    ["Dashboard Principal", "Análisis Estadístico", "Predicción 2026", "Análisis de Imagen"])

# PÁGINA 1: Dashboard Principal
if page == "Dashboard Principal":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Incendios 2014-2024", "123,456")
    with col2:
        st.metric("Promedio Anual", "12,345")
    with col3:
        st.metric("Departamento Crítico", "Petén")
    
    # Mapa de calor
    st.subheader("Mapa de Incendios Históricos")
    # (Implementar Folium heatmap)

# PÁGINA 2: Análisis Estadístico
elif page == "Análisis Estadístico":
    st.subheader("Correlación de Variables")
    # Mostrar correlation_matrix_pearson.png
    
    st.subheader("Importancia de Features")
    # Mostrar feature_importance.png
    
    st.subheader("Estacionalidad")
    # Mostrar monthly_seasonality.png

# PÁGINA 3: Predicción 2026
elif page == "Predicción 2026":
    st.subheader("Calendario de Riesgo 2026")
    
    calendar = pd.read_csv('data/outputs/calendario_incendios_2026.csv')
    
    # Gráfica interactiva
    fig = px.line(calendar, x='fecha', y='prediccion',
                  title='Predicción de Incendios por Día - 2026')
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla
    st.dataframe(calendar)

# PÁGINA 4: Análisis de Imagen
elif page == "Análisis de Imagen":
    st.subheader("Analizar Imagen Satelital con LLM")
    
    uploaded_file = st.file_uploader("Sube una imagen satelital", type=['jpg', 'png'])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Imagen cargada")
        
        if st.button("Analizar"):
            # Llamar API /predict/vision
            response = requests.post("http://localhost:8000/predict/vision", 
                                     files={"image": uploaded_file})
            st.write(response.json()['analysis'])
```

- [ ] Ejecutar: `streamlit run api/paso4_dashboard.py`
- [ ] Verificar en navegador: `http://localhost:8501`

### 7.2 Crear Mapas Interactivos (Folium)
- [ ] Mapa de calor de incendios históricos:
```python
import folium
from folium.plugins import HeatMap

# Crear mapa base centrado en Guatemala
m = folium.Map(location=[15.5, -90.25], zoom_start=7)

# Agregar heatmap
heat_data = [[row['latitude'], row['longitude']] for _, row in df_fires.iterrows()]
HeatMap(heat_data, radius=10).add_to(m)

# Guardar
m.save('visualizations/maps/heatmap_guatemala.html')
```

- [ ] Integrar en Streamlit dashboard

### 7.3 Gráficas Interactivas (Plotly)
- [ ] Serie temporal interactiva (zoom, pan)
- [ ] Gráfica de barras por departamento
- [ ] Scatter plot FRP vs Confidence
- [ ] Guardar en dashboard

---

## FASE 8: TESTING Y DOCUMENTACIÓN
**Duración**: 2 días

### 8.1 Testing Unitario
- [ ] Crear `tests/test_models.py`:
```python
import pytest
from api.paso3_api_fastapi import predict_vision

def test_random_forest():
    # Test RF predictions
    pass

def test_llm_vision():
    # Test LLM inference
    pass
```

- [ ] Ejecutar: `pytest tests/`

### 8.2 Crear README.md
- [ ] Instrucciones de instalación
- [ ] Cómo ejecutar notebooks
- [ ] Cómo usar API
- [ ] Ejemplos de uso

### 8.3 Crear RESULTADOS.md
- [ ] Métricas de Random Forest v2
- [ ] Métricas de LLM Vision fine-tuned
- [ ] Métricas de predicción temporal
- [ ] Comparación con baselines
- [ ] Conclusiones

### 8.4 Documentar Código
- [ ] Agregar docstrings a funciones
- [ ] Comentarios en scripts complejos
- [ ] Type hints en Python

---

## FASE 9: DEPLOYMENT
**Duración**: 1 día

### 9.1 Deploy API (Render / Railway)
- [ ] Crear `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api.paso3_api_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] Deploy en Render: https://render.com/
- [ ] Verificar endpoint público

### 9.2 Deploy Dashboard (Streamlit Cloud)
- [ ] Subir código a GitHub repo
- [ ] Conectar Streamlit Cloud: https://streamlit.io/cloud
- [ ] Configurar secrets (API keys)
- [ ] Verificar URL pública

### 9.3 Publicar Modelo en Hugging Face
- [ ] Verificar modelo subido
- [ ] Crear Model Card (README.md en HF)
- [ ] Agregar ejemplo de uso
- [ ] Publicar dataset en HF Datasets

---

## 🎯 ENTREGABLES FINALES

### Código
- [ ] 5 Notebooks Colab ejecutables
- [ ] Scripts Python documentados
- [ ] API REST funcional
- [ ] Dashboard web interactivo

### Modelos
- [ ] Random Forest v2 (mejorado)
- [ ] Qwen3-VL fine-tuned (publicado en HF)

### Datos
- [ ] Dataset FIRMS (CSV)
- [ ] Dataset anotado (JSONL)
- [ ] Calendario 2026 (CSV)

### Visualizaciones
- [ ] 10+ gráficas de alta resolución
- [ ] Mapas interactivos (HTML)

### Documentación
- [ ] INFORME.md (completo)
- [ ] PLAN.md (este archivo)
- [ ] RESULTADOS.md
- [ ] README.md

### Deployment
- [ ] API en producción (URL pública)
- [ ] Dashboard en producción (URL pública)

---

## 📊 MÉTRICAS DE ÉXITO

- [ ] Dataset: Mínimo 500 imágenes anotadas
- [ ] Random Forest: Accuracy > 90% (mejora sobre 88% actual)
- [ ] LLM Vision: BLEU score > 0.6, F1 > 0.85
- [ ] Predicción 2026: MAPE < 20%
- [ ] API: Latencia < 2s (Random Forest), < 5s (LLM Vision)
- [ ] Documentación: 100% funciones con docstrings

---

## ⏰ CRONOGRAMA RESUMEN

| Semana | Fases | Horas Estimadas |
|--------|-------|-----------------|
| 1 | Fases 0-1 (Setup + Datos) | 30-40h |
| 2 | Fases 2-3 (Procesamiento + Estadística) | 35-45h |
| 3 | Fases 4-5 (Fine-tuning + Forecast) | 30-40h |
| 4 | Fases 6-9 (API + Dashboard + Deploy) | 30-40h |
| **TOTAL** | **9 Fases** | **125-165h (~4 semanas)** |

**Nota**: Asume dedicación 30-40h/semana (tiempo completo).

---

## 🚀 SIGUIENTE PASO INMEDIATO

**AHORA MISMO**:
1. [ ] Registrarse en NASA FIRMS (obtener API key)
2. [ ] Registrarse en Google Earth Engine
3. [ ] Crear cuenta Hugging Face
4. [ ] Abrir Google Colab y verificar GPU

**Luego ejecutar**:
```bash
mkdir -p Docs notebooks data/{raw,processed,outputs} models visualizations scripts api
python scripts/download_firms.py
```

---

**¿Listo para arrancar? Marcá el primer checkbox y arrancamos!** 🔥

