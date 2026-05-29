# GETTING STARTED - Train-LLM Guatemala Fires

## PASO 1: Registros de APIs (15 minutos)

### 1.1 NASA FIRMS
1. Ir a: https://firms.modaps.eosdis.nasa.gov/api/area/
2. Clic en "Request Key"
3. Completar formulario (email, nombre, propósito)
4. Revisar email y copiar MAP_KEY
5. Guardar en archivo .env

### 1.2 Google Earth Engine
1. Ir a: https://signup.earthengine.google.com/
2. Registrarse con cuenta Google
3. Esperar aprobación (24-48h)
4. Mientras tanto, seguir con otros pasos

### 1.3 Hugging Face
1. Ir a: https://huggingface.co/join
2. Crear cuenta
3. Settings → Access Tokens → New token
4. Copiar token (hf_...)
5. Guardar en .env

## PASO 2: Crear .env (5 minutos)

Crear archivo `.env` en raíz del proyecto:

```env
# NASA FIRMS
FIRMS_MAP_KEY=tu_key_aqui

# Hugging Face
HF_TOKEN=hf_tu_token_aqui

# Proyecto
PROJECT_NAME=train-llm-guatemala-fires
DEVICE=cuda
```

## PASO 3: Crear Estructura de Directorios (2 minutos)

Ejecutar en terminal:

```bash
cd proyecto_incendios
mkdir -p notebooks data/{raw,processed,outputs} models visualizations scripts api
mkdir -p data/raw/landsat_images data/processed/images_rgb_512 visualizations/maps
```

## PASO 4: Instalar Dependencias Base (10 minutos)

```bash
pip install requests pandas numpy tqdm python-dotenv
```

## SIGUIENTE: Ir a PLAN.md Fase 1
