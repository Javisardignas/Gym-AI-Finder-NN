# 🏋️ GYM AI FINDER - Neural Network Exercise Search System

Sistema inteligente de búsqueda de ejercicios de gimnasio usando redes neuronales y embeddings semánticos. Combina un backend Python con IA y una interfaz Flutter moderna.

## 📋 Descripción

**GYM AI FINDER** es un sistema de búsqueda inteligente que utiliza modelos de lenguaje (Sentence Transformers) para encontrar ejercicios de gimnasio basándose en descripciones en lenguaje natural. El sistema aprende de un dataset de 675 ejercicios diferentes y puede encontrar coincidencias semánticas precisas.

### 🎯 Características Principales

- 🧠 **Red Neuronal Fine-tuned**: Modelo `sentence-transformers/all-MiniLM-L6-v2` entrenado específicamente para ejercicios de gimnasio
- 🔍 **Búsqueda Semántica**: Encuentra ejercicios por descripción, músculos objetivo, o movimiento
- 📊 **Sistema Multi-modelo**: Registro y versionado de modelos con métricas de rendimiento
- 🎨 **Interfaz Flutter**: UI moderna con tema oscuro/claro, testing interactivo y resultados visuales
- 🚀 **API REST**: Servidor Flask para comunicación Python ↔ Flutter
- 💾 **Base de Datos Vectorial**: Indexación optimizada de embeddings para búsqueda rápida

## 🏗️ Arquitectura

```
┌─────────────────┐
│  Flutter UI     │  ← Interfaz gráfica (Dart)
└────────┬────────┘
         │ HTTP
         ↓
┌─────────────────┐
│  Flask API      │  ← Servidor REST (Python)
│  (puerto 5000)  │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  GymBrain       │  ← Motor de IA (PyTorch + Transformers)
│  Neural Network │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Vector DB      │  ← Base de datos de embeddings (PKL)
│  675 ejercicios │
└─────────────────┘
```

## 📁 Estructura del Proyecto

```
Entrega/
├── nngym_v2.py              # 🧠 Motor principal de IA
├── deploy_en.bat            # 🚀 Script de despliegue automatizado
├── servidor_simple.py       # 🌐 Servidor Flask para API
├── model_registry.py        # 📚 Registro de modelos
├── training_session.py      # 📊 Gestión de sesiones de entrenamiento
│
├── gym_exercise_dataset.csv # 📋 Dataset (675 ejercicios)
├── gym_brain_finetuned.pt   # 💾 Modelo entrenado (PyTorch)
├── gym_database.pkl         # 🗄️ Base de datos vectorial
├── config.json              # ⚙️ Estado del modelo
│
├── lib/
│   ├── main.dart            # 🎨 App Flutter principal
│   └── testing_page.dart    # 🧪 Página de testing
│
└── assets/
    ├── gym_exercise_dataset.csv      # Copia para Flutter
    └── validation_set.json            # Set de validación
```

## 🚀 Instalación

### Requisitos Previos

- **Python 3.8+**
- **Flutter 3.0+**
- **Windows 10/11** (para usar `deploy_en.bat`)

### Instalación Rápida

```bash
# 1. Ejecutar script de despliegue (instala todo automáticamente)
deploy_en.bat

# El script instalará:
# - Entorno virtual Python en C:\gym_env
# - Dependencias: torch, transformers, flask, pandas, sklearn
# - Flutter SDK (si no está instalado)
# - Compilará la app Flutter
```

### Instalación Manual

```bash
# Crear entorno virtual
python -m venv C:\gym_env

# Activar entorno
C:\gym_env\Scripts\activate

# Instalar dependencias
pip install torch torchvision
pip install transformers sentence-transformers
pip install flask pandas scikit-learn

# Instalar dependencias Flutter
flutter pub get
```

## 📖 Uso

### Opción 1: Entrenar Red Neuronal (Primera vez)

```bash
python nngym_v2.py 1
```

**Qué hace:**

- Descarga el modelo base `all-MiniLM-L6-v2` desde Hugging Face
- Entrena el modelo con el dataset (6 épocas)
- Genera embeddings para todos los ejercicios
- Crea `gym_brain_finetuned.pt` y `gym_database.pkl`
- Realiza test de validación con 5 muestras
- Entra en modo de búsqueda interactiva

**Tiempo estimado:** 5-15 minutos (dependiendo de CPU/GPU)

### Opción 2: Cargar Modelo Pre-entrenado

```bash
python nngym_v2.py 3
```

**Qué hace:**

- Carga el modelo ya entrenado
- Carga la base de datos vectorial
- Entra directamente en búsqueda interactiva

**Tiempo estimado:** 5-10 segundos

### Opción 3: Iniciar Servidor API

```bash
python nngym_v2.py api
```

**Qué hace:**

- Inicia servidor Flask en `http://localhost:5000`
- Expone endpoint `/api/search` para Flutter
- Mantiene el modelo en memoria para respuestas rápidas

### Opción 4: Ejecutar App Flutter

```bash
# Terminal 1: Iniciar servidor API
python nngym_v2.py api

# Terminal 2: Ejecutar Flutter
flutter run -d windows
```

O usar el script todo-en-uno:

```bash
deploy_en.bat
# Seleccionar opción: [5] Launch FULL SYSTEM
```

## 🔍 Ejemplos de Búsqueda

El sistema entiende lenguaje natural:

### Búsqueda por Descripción de Movimiento

```
Query: "Move arm up and bring weight down to chest"
Results:
  1. Dumbbell Bench Press (95.2%)
  2. Barbell Bench Press (93.8%)
  3. Cable Fly (89.4%)
```

### Búsqueda por Músculo Objetivo

```
Query: "exercise for quadriceps and glutes"
Results:
  1. Barbell Squat (94.7%)
  2. Leg Press (92.3%)
  3. Bulgarian Split Squat (90.1%)
```

### Búsqueda por Equipamiento

```
Query: "cable machine for back muscles"
Results:
  1. Cable Row (93.5%)
  2. Lat Pulldown (91.8%)
  3. Cable Face Pull (88.9%)
```

## 🧪 Testing Page (Flutter)

La aplicación incluye una página de testing interactiva:

- ✅ Test de conectividad con el servidor
- 📋 Set de validación de 20 ejercicios predefinidos
- 🎯 Muestra precisión de búsqueda en tiempo real
- 📊 Visualización de scores de similitud

Acceso: Botón "🧪 Testing" en la esquina superior derecha

## ⚙️ Configuración

### `config.json`

```json
{
  "model_ready": true,
  "last_updated": "2026-01-29T17:56:55.449077",
  "model_type": "trained"
}
```

- `model_ready`: Indica si el modelo está listo para usar
- `model_type`: `"trained"` o `"preloaded"`

### Parámetros del Modelo (en `nngym_v2.py`)

```python
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TRAINING_EPOCHS = 6
SIMILARITY_THRESHOLD = 0.85
LEARNING_RATE = 1e-5
```

## 📊 Dataset

El dataset incluye **675 ejercicios** con la siguiente estructura:

| Campo             | Descripción                                             |
| ----------------- | ------------------------------------------------------- |
| Exercise Name     | Nombre del ejercicio                                    |
| Equipment         | Equipamiento necesario (Cable, Barbell, Dumbbell, etc.) |
| Preparation       | Instrucciones de preparación                            |
| Execution         | Instrucciones de ejecución                              |
| Target_Muscles    | Músculos principales trabajados                         |
| Synergist_Muscles | Músculos sinérgicos                                     |
| Difficulty        | Nivel de dificultad (1-5)                               |

**Fuente:** Base de datos profesional de ejercicios de fitness

## 🛠️ Tecnologías Utilizadas

### Backend (Python)

- **PyTorch**: Framework de deep learning
- **Transformers** (Hugging Face): Modelos de lenguaje pre-entrenados
- **Flask**: API REST
- **Pandas**: Manipulación de datos
- **scikit-learn**: División train/test

### Frontend (Flutter)

- **Dart 3.0+**
- **Material Design 3**
- **HTTP Client**: Comunicación con API
- **Provider**: State management

## 📈 Métricas de Rendimiento

### Precisión del Modelo

- **Validación (top-1):** ~85-95%
- **Validación (top-3):** ~95-99%

### Velocidad

- **Búsqueda:** < 100ms (con modelo cargado en memoria)
- **Entrenamiento:** 5-15 minutos (CPU) / 2-5 minutos (GPU)

### Uso de Memoria

- **Modelo en RAM:** ~90 MB
- **Base de datos vectorial:** ~5 MB

## 🐛 Troubleshooting

### Error: "Database not found"

```bash
# Solución: Entrenar el modelo primero
python nngym_v2.py 1
```

### Error: "Connection timeout to Hugging Face"

```bash
# Solución: Verificar conexión a internet y reintentar
# El script tiene 3 reintentos automáticos con timeout de 5 minutos
```

### Error: "Port 5000 already in use"

```bash
# Solución: Cambiar puerto en servidor_simple.py o matar proceso
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Flutter no se conecta al servidor

```bash
# Verificar que el servidor esté corriendo
curl http://localhost:5000/api/search

# Verificar firewall de Windows
# Permitir Python en firewall si es necesario
```

## 🔧 Scripts de Utilidad

### `deploy_en.bat`

Script master de despliegue con menú interactivo:

- [1] Check System Status
- [2] Install Python Dependencies
- [3] Install Flutter Dependencies
- [4] Build Flutter Application
- [5] Launch FULL SYSTEM (API + Flutter)
- [6] Run Tests
- [7] Generate Production Build

### Otros Scripts

- `servidor_simple.py`: Servidor Flask standalone
- `generar_database.py`: Regenerar base de datos vectorial
- `model_registry.py`: Gestión de múltiples versiones de modelos

## 📝 Notas Importantes

1. **Primera ejecución:** Siempre ejecutar `python nngym_v2.py 1` para entrenar el modelo
2. **Modelo en memoria:** El servidor API mantiene el modelo cargado para respuestas rápidas
3. **Assets Flutter:** Los archivos en `assets/` son necesarios para la compilación
4. **Base de datos:** `gym_database.pkl` se regenera automáticamente si se borra

## 🚀 Próximas Mejoras

- [ ] Soporte para imágenes de ejercicios
- [ ] Filtros por dificultad y equipamiento
- [ ] Sistema de favoritos y rutinas personalizadas
- [ ] Modo offline con base de datos local
- [ ] Soporte para múltiples idiomas
- [ ] Integración con APIs de fitness tracking

## 👨‍💻 Autor

Sistema de IA para búsqueda de ejercicios de gimnasio  
**Versión:** 2.0  
**Fecha:** Enero 2026

## 📄 Licencia

Proyecto educativo - Uso libre para aprendizaje

---

**🏋️ "LIGHT WEIGHT, BABY!" 💪**
