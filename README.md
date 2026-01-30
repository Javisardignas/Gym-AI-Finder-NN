Developed by Mario Alonso Lopez, Emilian Alexandru Bujanca, and Javier Sariñas Carreto.

# 🏋️ GYM AI FINDER - Neural Network Exercise Search System

## ⚡ Quick Start

**IMPORTANT:** To compile and run this project, execute `deploy_en.bat`. The neural network model is `nngym_v2.py`.

---

Intelligent gym exercise search system using neural networks and semantic embeddings. Combines a Python backend with AI and a modern Flutter interface.

## 📋 Description

**GYM AI FINDER** is an intelligent search system that uses language models (Sentence Transformers) to find gym exercises based on natural language descriptions. The system learns from a dataset of 675 different exercises and can find precise semantic matches.

### 🎯 Main Features

- 🧠 **Fine-tuned Neural Network**: `sentence-transformers/all-MiniLM-L6-v2` model specifically trained for gym exercises
- 🔍 **Semantic Search**: Find exercises by description, target muscles, or movement
- 📊 **Multi-model System**: Model registry and versioning with performance metrics
- 🎨 **Flutter Interface**: Modern UI with dark/light theme, interactive testing, and visual results
- 🚀 **REST API**: Flask server for Python ↔ Flutter communication
- 💾 **Vector Database**: Optimized embedding indexing for fast search

## 🏗️ Architecture

```
┌─────────────────┐
│  Flutter UI     │  ← Graphical interface (Dart)
└────────┬────────┘
         │ HTTP
         ↓
┌─────────────────┐
│  Flask API      │  ← REST Server (Python)
│  (port 5000)    │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  GymBrain       │  ← AI Engine (PyTorch + Transformers)
│  Neural Network │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Vector DB      │  ← Embeddings database (PKL)
│  675 exercises  │
└─────────────────┘
```

## 📁 Project Structure

```
Entrega/
├── nngym_v2.py              # 🧠 Main AI engine
├── deploy_en.bat            # 🚀 Automated deployment script
├── servidor_simple.py       # 🌐 Flask server for API
├── model_registry.py        # 📚 Model registry
├── training_session.py      # 📊 Training session management
│
├── gym_exercise_dataset.csv # 📋 Dataset (675 exercises)
├── gym_brain_finetuned.pt   # 💾 Trained model (PyTorch)
├── gym_database.pkl         # 🗄️ Vector database
├── config.json              # ⚙️ Model state
│
├── lib/
│   ├── main.dart            # 🎨 Main Flutter app
│   └── testing_page.dart    # 🧪 Testing page
│
└── assets/
    ├── gym_exercise_dataset.csv      # Copy for Flutter
    └── validation_set.json            # Validation set
```

## 🚀 Installation

### Prerequisites

- **Python 3.8+**
- **Flutter 3.0+**
- **Windows 10/11** (to use `deploy_en.bat`)

### Quick Installation

```bash
# 1. Run deployment script (installs everything automatically)
deploy_en.bat

# The script will install:
# - Python virtual environment at C:\gym_env
# - Dependencies: torch, transformers, flask, pandas, sklearn
# - Flutter SDK (if not installed)
# - Compile the Flutter app
```

### Manual Installation

```bash
# Create virtual environment
python -m venv C:\gym_env

# Activate environment
C:\gym_env\Scripts\activate

# Install dependencies
pip install torch torchvision
pip install transformers sentence-transformers
pip install flask pandas scikit-learn

# Install Flutter dependencies
flutter pub get
```

## 📖 Usage

### Option 1: Train Neural Network (First time)

```bash
python nngym_v2.py 1
```

**What it does:**

- Downloads the base model `all-MiniLM-L6-v2` from Hugging Face
- Trains the model with the dataset (6 epochs)
- Generates embeddings for all exercises
- Creates `gym_brain_finetuned.pt` and `gym_database.pkl`
- Performs validation test with 5 samples
- Enters interactive search mode

**Estimated time:** 5-15 minutes (depending on CPU/GPU)

### Option 2: Load Pre-trained Model

```bash
python nngym_v2.py 3
```

**What it does:**

- Loads the already trained model
- Loads the vector database
- Enters interactive search mode directly

**Estimated time:** 5-10 seconds

### Option 3: Start API Server

```bash
python nngym_v2.py api
```

**What it does:**

- Starts Flask server at `http://localhost:5000`
- Exposes `/api/search` endpoint for Flutter
- Keeps the model in memory for fast responses

### Option 4: Run Flutter App

```bash
# Terminal 1: Start API server
python nngym_v2.py api

# Terminal 2: Run Flutter
flutter run -d windows
```

Or use the all-in-one script:

```bash
deploy_en.bat
# Select option: [5] Launch FULL SYSTEM
```

## 🔍 Search Examples

The system understands natural language:

### Search by Movement Description

```
Query: "Move arm up and bring weight down to chest"
Results:
  1. Dumbbell Bench Press (95.2%)
  2. Barbell Bench Press (93.8%)
  3. Cable Fly (89.4%)
```

### Search by Target Muscle

```
Query: "exercise for quadriceps and glutes"
Results:
  1. Barbell Squat (94.7%)
  2. Leg Press (92.3%)
  3. Bulgarian Split Squat (90.1%)
```

### Search by Equipment

```
Query: "cable machine for back muscles"
Results:
  1. Cable Row (93.5%)
  2. Lat Pulldown (91.8%)
  3. Cable Face Pull (88.9%)
```

## 🧪 Testing Page (Flutter)

The application includes an interactive testing page:

- ✅ Server connectivity test
- 📋 Validation set of 20 predefined exercises
- 🎯 Shows search accuracy in real-time
- 📊 Similarity score visualization

Access: "🧪 Testing" button in the upper right corner

## ⚙️ Configuration

### `config.json`

```json
{
  "model_ready": true,
  "last_updated": "2026-01-29T17:56:55.449077",
  "model_type": "trained"
}
```

- `model_ready`: Indicates if the model is ready to use
- `model_type`: `"trained"` or `"preloaded"`

### Model Parameters (in `nngym_v2.py`)

```python
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TRAINING_EPOCHS = 6
SIMILARITY_THRESHOLD = 0.85
LEARNING_RATE = 1e-5
```

## 📊 Dataset

The dataset includes **675 exercises** with the following structure:

| Field             | Description                                         |
| ----------------- | --------------------------------------------------- |
| Exercise Name     | Exercise name                                       |
| Equipment         | Required equipment (Cable, Barbell, Dumbbell, etc.) |
| Preparation       | Preparation instructions                            |
| Execution         | Execution instructions                              |
| Target_Muscles    | Main muscles worked                                 |
| Synergist_Muscles | Synergist muscles                                   |
| Difficulty        | Difficulty level (1-5)                              |

**Source:** Professional fitness exercise database

## 🛠️ Technologies Used

### Backend (Python)

- **PyTorch**: Deep learning framework
- **Transformers** (Hugging Face): Pre-trained language models
- **Flask**: REST API
- **Pandas**: Data manipulation
- **scikit-learn**: Train/test split

### Frontend (Flutter)

- **Dart 3.0+**
- **Material Design 3**
- **HTTP Client**: API communication
- **Provider**: State management

## 📈 Performance Metrics

### Model Accuracy

- **Validation (top-1):** ~85-95%
- **Validation (top-3):** ~95-99%

### Speed

- **Search:** < 100ms (with model loaded in memory)
- **Training:** 5-15 minutes (CPU) / 2-5 minutes (GPU)

### Memory Usage

- **Model in RAM:** ~90 MB
- **Vector database:** ~5 MB

## 🐛 Troubleshooting

### Error: "Database not found"

```bash
# Solution: Train the model first
python nngym_v2.py 1
```

### Error: "Connection timeout to Hugging Face"

```bash
# Solution: Check internet connection and retry
# The script has 3 automatic retries with 5-minute timeout
```

### Error: "Port 5000 already in use"

```bash
# Solution: Change port in servidor_simple.py or kill process
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Flutter doesn't connect to server

```bash
# Verify that the server is running
curl http://localhost:5000/api/search

# Check Windows firewall
# Allow Python in firewall if necessary
```

## 🔧 Utility Scripts

### `deploy_en.bat`

Master deployment script with interactive menu:

- [1] Check System Status
- [2] Install Python Dependencies
- [3] Install Flutter Dependencies
- [4] Build Flutter Application
- [5] Launch FULL SYSTEM (API + Flutter)
- [6] Run Tests
- [7] Generate Production Build

### Other Scripts

- `servidor_simple.py`: Standalone Flask server
- `generar_database.py`: Regenerate vector database
- `model_registry.py`: Manage multiple model versions

## 📝 Important Notes

1. **First run:** Always execute `python nngym_v2.py 1` to train the model
2. **Model in memory:** The API server keeps the model loaded for fast responses
3. **Flutter Assets:** Files in `assets/` are necessary for compilation
4. **Database:** `gym_database.pkl` is automatically regenerated if deleted

## 🚀 Future Improvements

- [ ] Support for exercise images
- [ ] Filters by difficulty and equipment
- [ ] Favorites system and personalized routines
- [ ] Offline mode with local database
- [ ] Multi-language support
- [ ] Integration with fitness tracking APIs

## 👨‍💻 Author

AI system for gym exercise search  
**Version:** 2.0  
**Date:** January 2026

## 📄 License

Educational project - Free use for learning

---

**🏋️ "LIGHT WEIGHT, BABY!" 💪**
