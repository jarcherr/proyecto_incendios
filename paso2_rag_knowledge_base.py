"""
╔══════════════════════════════════════════════════════════════════════╗
║  EJERCICIO 7 - PASO 2: BASE DE CONOCIMIENTO RAG                   ║
║  Proyecto: Predicción de Incendios Forestales                      ║
║  Tema: Retrieval-Augmented Generation (RAG)                        ║
╚══════════════════════════════════════════════════════════════════════╝

DESCRIPCIÓN:
    Este script crea una base de conocimiento sobre incendios forestales
    usando la técnica RAG (Retrieval-Augmented Generation).
    
    RAG funciona así:
    1. INGESTA: Se recopilan documentos/reportes sobre incendios.
    2. CHUNKING: Se dividen en fragmentos pequeños.
    3. EMBEDDING: Se convierten a vectores numéricos (embeddings).
    4. ALMACENAMIENTO: Se guardan en una base de datos vectorial.
    5. CONSULTA: Ante una pregunta, se buscan los fragmentos más
       relevantes y se le pasan al LLM como contexto.

    En este caso, usamos una implementación ligera con TF-IDF
    (sin necesidad de APIs externas de embeddings) para que el
    ejercicio sea completamente autónomo.

EJECUCIÓN:
    python3 paso2_rag_knowledge_base.py
    
    Genera:
    - knowledge_base.pkl (base de conocimiento vectorizada)
    - knowledge_chunks.json (fragmentos de texto indexados)
"""

import json
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ═══════════════════════════════════════════════════════════════
# BASE DE CONOCIMIENTO: REPORTES SOBRE INCENDIOS FORESTALES
# ═══════════════════════════════════════════════════════════════
# Estos textos simulan información recopilada de fuentes públicas
# como NASA FIRMS, CONAF, NIFC, y reportes científicos.
# En un caso real, se usaría NotebookLM o un scraper para
# recopilar esta información de fuentes públicas.

KNOWLEDGE_DOCUMENTS = [
    {
        "source": "NASA FIRMS - Fire Information for Resource Management System",
        "year": 2023,
        "content": """Los incendios forestales a nivel mundial han aumentado un 30% en la última 
        década según datos satelitales de NASA FIRMS. El sistema MODIS y VIIRS detectan puntos 
        de calor (hotspots) con una resolución de 375 metros. Las regiones más afectadas incluyen 
        la Amazonía, el Mediterráneo, California, y el sudeste de Australia. La temporada de 
        incendios se ha extendido en promedio 18.7% más que hace 20 años debido al cambio climático."""
    },
    {
        "source": "IPCC - Panel Intergubernamental sobre Cambio Climático",
        "year": 2022,
        "content": """El cambio climático ha incrementado la frecuencia e intensidad de los incendios 
        forestales. Las temperaturas globales han aumentado 1.1°C desde la era preindustrial. 
        Las olas de calor, que son un factor clave en la ignición de incendios, son ahora 5 veces 
        más probables. La sequía prolongada reduce la humedad del combustible vegetal, creando 
        condiciones ideales para la propagación del fuego. Se proyecta que para 2050, el riesgo 
        de incendios extremos aumente entre un 30% y un 57%."""
    },
    {
        "source": "Sistema Canadiense FWI - Fire Weather Index",
        "year": 2021,
        "content": """El Fire Weather Index (FWI) es el sistema estándar internacional para evaluar 
        el peligro de incendios forestales. Utiliza seis componentes: FFMC (Fine Fuel Moisture Code) 
        mide la humedad de la hojarasca superficial; DMC (Duff Moisture Code) mide la humedad de 
        la capa orgánica media; DC (Drought Code) indica la sequía profunda del suelo; ISI (Initial 
        Spread Index) combina viento y FFMC para estimar la velocidad de propagación; BUI (Buildup 
        Index) combina DMC y DC para estimar el combustible disponible; FWI integra ISI y BUI en 
        un índice general. Un FWI superior a 30 indica peligro extremo."""
    },
    {
        "source": "CONAF Chile - Corporación Nacional Forestal",
        "year": 2023,
        "content": """Chile experimentó en 2023 una de las peores temporadas de incendios forestales 
        de su historia, con más de 430,000 hectáreas quemadas. La región del Biobío fue la más 
        afectada. Los factores principales fueron: temperaturas superiores a 40°C, humedad relativa 
        inferior al 20%, y vientos de más de 30 km/h. La combinación de sequía prolongada (más de 
        13 años de megasequía) y la presencia de plantaciones de eucalipto y pino, altamente 
        inflamables, agravaron la situación. El costo estimado superó los 3,000 millones de dólares."""
    },
    {
        "source": "NIFC - National Interagency Fire Center (EE.UU.)",
        "year": 2022,
        "content": """En Estados Unidos, el promedio anual de área quemada ha pasado de 3.3 millones 
        de acres en la década de 1990 a más de 7 millones de acres en la década de 2020. California 
        concentra el 40% de los incendios más destructivos. El incendio Camp Fire de 2018 en 
        Paradise, California, fue el más mortífero con 85 víctimas. Las causas principales son: 
        rayos (44%), actividad humana (47%), y causas desconocidas (9%). El costo de supresión 
        federal superó los 4 mil millones de dólares en 2021."""
    },
    {
        "source": "Investigación - Universidad de Barcelona",
        "year": 2021,
        "content": """Un estudio de la Universidad de Barcelona demostró que la temperatura es el 
        factor meteorológico más correlacionado con la ocurrencia de incendios (r=0.72), seguido 
        por la humedad relativa inversa (r=-0.65) y la velocidad del viento (r=0.45). Sin embargo, 
        el índice FWI, que integra múltiples variables, tiene la correlación más alta (r=0.82). 
        Los modelos de machine learning, especialmente Random Forest y Gradient Boosting, han 
        demostrado precisiones superiores al 85% en la predicción de ocurrencia de incendios 
        cuando se alimentan con datos meteorológicos y del FWI."""
    },
    {
        "source": "FAO - Organización de las Naciones Unidas para la Alimentación",
        "year": 2022,
        "content": """Según la FAO, los incendios forestales destruyen aproximadamente 340 millones 
        de hectáreas de bosque al año a nivel mundial. Los ecosistemas más vulnerables son los 
        bosques boreales de Siberia y Canadá, los bosques mediterráneos, y las sabanas tropicales. 
        La prevención es 10 veces más económica que la supresión. Las estrategias más efectivas 
        incluyen: monitoreo satelital en tiempo real, modelos predictivos basados en IA, gestión 
        del combustible forestal (quemas prescritas), y educación comunitaria."""
    },
    {
        "source": "European Forest Fire Information System (EFFIS)",
        "year": 2023,
        "content": """Europa registró en 2022 la peor temporada de incendios en décadas, con más 
        de 785,000 hectáreas quemadas. Grecia, España, Portugal y Francia fueron los países más 
        afectados. El sistema EFFIS utiliza datos de Copernicus para monitorear incendios activos 
        y evaluar el riesgo. Los modelos predictivos de EFFIS combinan datos meteorológicos 
        (temperatura, humedad, viento, precipitación) con índices de vegetación (NDVI) y datos 
        topográficos (pendiente, altitud, orientación) para generar mapas de riesgo diarios."""
    },
    {
        "source": "Estudio - Factores de Riesgo de Incendios",
        "year": 2020,
        "content": """Los principales factores que determinan el riesgo de incendio forestal son:
        1) Temperatura: por encima de 30°C el riesgo aumenta exponencialmente.
        2) Humedad Relativa: por debajo del 30% el material vegetal se seca rápidamente.
        3) Velocidad del Viento: vientos superiores a 20 km/h propagan el fuego rápidamente.
        4) Precipitación: períodos sin lluvia superiores a 7 días incrementan significativamente 
        el riesgo. 5) Tipo de vegetación: las coníferas y el eucalipto son altamente inflamables.
        6) Topografía: las pendientes pronunciadas aceleran la propagación del fuego.
        7) Actividad humana: el 90% de los incendios en Europa son de origen antrópico."""
    },
    {
        "source": "Reporte - Tecnologías de Predicción de Incendios",
        "year": 2023,
        "content": """Las tecnologías modernas para la predicción de incendios forestales incluyen:
        Satélites (MODIS, VIIRS, Sentinel-2) para detección de puntos de calor en tiempo real.
        Drones equipados con cámaras térmicas para vigilancia de zonas de alto riesgo.
        Sensores IoT en el terreno que miden temperatura, humedad y gases.
        Modelos de Machine Learning (Random Forest, XGBoost, Redes Neuronales) que procesan 
        datos meteorológicos históricos para predecir la probabilidad de incendio.
        Sistemas RAG (Retrieval-Augmented Generation) que combinan bases de datos de incendios 
        históricos con LLMs para generar análisis contextuales y recomendaciones."""
    },
    {
        "source": "Análisis Estadístico - Patrones Temporales de Incendios",
        "year": 2022,
        "content": """El análisis de 10 años de datos de incendios forestales revela patrones 
        temporales claros: el 78% de los incendios ocurren entre junio y septiembre en el 
        hemisferio norte. Los meses más peligrosos son julio y agosto, cuando la temperatura 
        media supera los 35°C y la humedad cae por debajo del 25%. Existe una correlación 
        significativa entre el fenómeno de El Niño y el aumento de incendios en Sudamérica 
        y el sudeste asiático. Los incendios nocturnos, aunque menos frecuentes, son más 
        difíciles de controlar debido a la inversión térmica."""
    },
    {
        "source": "Guía de Prevención - Protocolo de Alerta Temprana",
        "year": 2023,
        "content": """Un sistema de alerta temprana efectivo para incendios forestales debe incluir:
        Nivel Verde (FWI < 10): Riesgo bajo, operaciones normales.
        Nivel Amarillo (FWI 10-20): Riesgo moderado, aumentar vigilancia.
        Nivel Naranja (FWI 20-30): Riesgo alto, activar brigadas de prevención.
        Nivel Rojo (FWI > 30): Riesgo extremo, prohibir actividades en zonas forestales.
        Las condiciones críticas se definen como: temperatura > 35°C, humedad < 25%, 
        viento > 25 km/h, y sin lluvia en los últimos 10 días. Cuando se cumplen 3 de 
        estas 4 condiciones simultáneamente, la probabilidad de incendio supera el 80%."""
    }
]


# ═══════════════════════════════════════════════════════════════
# CLASE: SISTEMA RAG LIGERO
# ═══════════════════════════════════════════════════════════════
class LightweightRAG:
    """
    Sistema RAG (Retrieval-Augmented Generation) ligero.
    
    Usa TF-IDF + Cosine Similarity en lugar de embeddings de LLM.
    Esto permite funcionar sin APIs externas.
    
    En producción, se reemplazaría TF-IDF por:
    - OpenAI Embeddings (text-embedding-3-small)
    - Sentence Transformers (all-MiniLM-L6-v2)
    - Cohere Embed
    
    Y el almacenamiento por:
    - ChromaDB, Pinecone, Weaviate, o FAISS
    """
    
    def __init__(self):
        self.chunks = []
        self.metadata = []
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=None,  # Mantener stop words en español
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.vectors = None
    
    def ingest_documents(self, documents):
        """
        Ingesta de documentos: chunking + vectorización.
        
        En un sistema real, aquí se haría:
        1. Cargar PDFs, HTMLs, etc.
        2. Dividir en chunks de ~500 tokens con overlap
        3. Generar embeddings con un modelo
        4. Almacenar en vector store
        """
        print("Ingiriendo documentos en la base de conocimiento RAG...")
        
        for doc in documents:
            # Chunking simple: dividir por párrafos/oraciones
            content = doc['content'].strip()
            sentences = [s.strip() for s in content.replace('\n', ' ').split('.') if len(s.strip()) > 20]
            
            # Crear chunks de 2-3 oraciones
            for i in range(0, len(sentences), 2):
                chunk = '. '.join(sentences[i:i+3]) + '.'
                self.chunks.append(chunk)
                self.metadata.append({
                    'source': doc['source'],
                    'year': doc['year'],
                    'chunk_id': len(self.chunks) - 1
                })
        
        # Vectorizar con TF-IDF
        self.vectors = self.vectorizer.fit_transform(self.chunks)
        
        print(f"  - {len(documents)} documentos procesados")
        print(f"  - {len(self.chunks)} chunks generados")
        print(f"  - Vocabulario: {len(self.vectorizer.vocabulary_)} términos")
    
    def search(self, query, top_k=3):
        """
        Búsqueda semántica: encuentra los chunks más relevantes.
        
        Proceso:
        1. Vectorizar la consulta con el mismo TF-IDF
        2. Calcular similitud coseno con todos los chunks
        3. Devolver los top_k más similares
        """
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # Top K resultados
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Umbral mínimo de relevancia
                results.append({
                    'chunk': self.chunks[idx],
                    'source': self.metadata[idx]['source'],
                    'year': self.metadata[idx]['year'],
                    'similarity': round(float(similarities[idx]), 4)
                })
        
        return results
    
    def generate_context(self, query, top_k=3):
        """
        Genera el contexto para el LLM basado en los chunks recuperados.
        
        Este contexto se insertaría en el prompt del LLM:
        
        PROMPT = f'''
        Basándote en la siguiente información:
        {context}
        
        Responde la siguiente pregunta:
        {query}
        '''
        """
        results = self.search(query, top_k)
        
        if not results:
            return "No se encontró información relevante en la base de conocimiento."
        
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[Fuente {i}: {r['source']} ({r['year']})] "
                f"(Relevancia: {r['similarity']:.2%})\n{r['chunk']}"
            )
        
        return "\n\n".join(context_parts)
    
    def save(self, filepath):
        """Guarda la base de conocimiento."""
        data = {
            'chunks': self.chunks,
            'metadata': self.metadata,
            'vectorizer': self.vectorizer,
            'vectors': self.vectors
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Base de conocimiento guardada en: {filepath}")
    
    def load(self, filepath):
        """Carga la base de conocimiento."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.chunks = data['chunks']
        self.metadata = data['metadata']
        self.vectorizer = data['vectorizer']
        self.vectors = data['vectors']
        print(f"Base de conocimiento cargada: {len(self.chunks)} chunks")


# ═══════════════════════════════════════════════════════════════
# EJECUCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  PROYECTO: PREDICCIÓN DE INCENDIOS FORESTALES           ║")
    print("║  Paso 2: Base de Conocimiento RAG                       ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    
    # Crear e ingerir documentos
    rag = LightweightRAG()
    rag.ingest_documents(KNOWLEDGE_DOCUMENTS)
    
    # Guardar
    rag.save(os.path.join(output_dir, 'knowledge_base.pkl'))
    
    # Guardar chunks como JSON para referencia
    chunks_data = [
        {'chunk_id': i, 'text': chunk, **meta}
        for i, (chunk, meta) in enumerate(zip(rag.chunks, rag.metadata))
    ]
    with open(os.path.join(output_dir, 'knowledge_chunks.json'), 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    # Demo: Consultas de ejemplo
    print("\n" + "=" * 60)
    print("DEMO: Consultas a la Base de Conocimiento")
    print("=" * 60)
    
    queries = [
        "¿Cuáles son los factores principales que causan incendios forestales?",
        "¿Qué es el índice FWI y cómo se usa?",
        "¿Qué tecnologías se usan para predecir incendios?",
        "¿Cuál es el nivel de alerta cuando el FWI supera 30?",
    ]
    
    for query in queries:
        print(f"\n{'─'*60}")
        print(f"CONSULTA: {query}")
        print(f"{'─'*60}")
        
        results = rag.search(query, top_k=2)
        for i, r in enumerate(results, 1):
            print(f"\n  Resultado {i} (Similitud: {r['similarity']:.2%}):")
            print(f"  Fuente: {r['source']} ({r['year']})")
            print(f"  Texto: {r['chunk'][:200]}...")
        
        context = rag.generate_context(query, top_k=2)
        print(f"\n  CONTEXTO GENERADO PARA LLM:")
        print(f"  {context[:300]}...")
    
    print("\n" + "=" * 60)
    print("PASO 2 COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    print(f"\nArchivos generados:")
    print(f"  - knowledge_base.pkl")
    print(f"  - knowledge_chunks.json")
    print(f"\nSiguiente paso: Ejecutar paso3_api_fastapi.py")
