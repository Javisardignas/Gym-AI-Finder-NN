"""
GYM AI - Neural Network Exercise Search System
Reorganized for optimal structure and readability
"""

# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd
import os
import torch
import json
import pickle
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from transformers import AutoTokenizer, AutoModel
from transformers import logging as hf_logging
from sklearn.model_selection import train_test_split
import logging
import warnings

# Suppress transformers library logging and warnings
hf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*position_ids.*')

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
FILE_NAME = SCRIPT_DIR / "gym_exercise_dataset.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TRAINING_LOG = SCRIPT_DIR / "training_log.json"
MODEL_PATH = SCRIPT_DIR / "gym_brain_finetuned.pt"
DATABASE_PATH = SCRIPT_DIR / "gym_database.pkl"
VALIDATION_SET_PATH = SCRIPT_DIR / "assets" / "validation_set.json"
CONFIG_PATH = SCRIPT_DIR / "config.json"

# ============================================================================
# MODEL STATE MANAGEMENT
# ============================================================================
def init_model_state():
    """Initialize model state configuration file"""
    config = {
        "model_ready": False,
        "last_updated": None,
        "model_type": None
    }
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return config

def load_model_state():
    """Load model state from configuration file"""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except:
        pass
    return init_model_state()

def save_model_state(ready, model_type):
    """Save model state - marks model as ready for production use"""
    config = {
        "model_ready": ready,
        "last_updated": datetime.now().isoformat(),
        "model_type": model_type
    }
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✅ Model state updated: model_ready={ready}")

def is_model_ready():
    """Check if model is ready for production use"""
    state = load_model_state()
    return state.get("model_ready", False)

# ============================================================================
# NEURAL NETWORK CLASS
# ============================================================================
class GymBrain:
    """
    Neural Network for Exercise Search and Recommendation
    Uses sentence-transformers for semantic similarity matching
    """
    
    def __init__(self, load_pretrained_weights=False):
        """
        Initialize the neural network model
        
        Args:
            load_pretrained_weights: Whether to load fine-tuned weights from disk
        """
        self.has_test_access = False
        print(f"🧠 Loading Model: {MODEL_NAME}...")
        
        # Download model with retry logic
        self._download_model_with_retry()
        
        # Setup device (GPU/CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"📱 Device: {self.device}")
        
        # Load fine-tuned weights if requested and available
        if load_pretrained_weights and os.path.exists(MODEL_PATH):
            print(f"✅ Loading pre-trained weights from {MODEL_PATH}")
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        
        # Initialize model properties
        self.similarity_threshold = 0.85
        self.training_history = self._load_training_log()
        self.database = None
        self.database_names = None
        self.database_descriptions = None
    
    def _download_model_with_retry(self, max_retries=3):
        """Download model from Hugging Face with retry logic"""
        for retry_count in range(max_retries):
            try:
                os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5 minutes timeout
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                self.model = AutoModel.from_pretrained(MODEL_NAME)
                return
            except Exception as e:
                if retry_count >= max_retries - 1:
                    print(f"\n❌ ERROR: Could not download model after {max_retries} attempts")
                    print(f"   Error: {str(e)}")
                    print(f"\n   Possible solutions:")
                    print(f"   1. Check your internet connection")
                    print(f"   2. Check if Hugging Face Hub is accessible")
                    print(f"   3. Try again in a few moments")
                    sys.exit(1)
                else:
                    print(f"\n⚠️  Connection timeout (attempt {retry_count+1}/{max_retries})")
                    print(f"   Retrying model download...")
                    import time
                    time.sleep(5)
    
    def get_embeddings(self, text_list, enable_learning=False):
        """
        Generate sentence embeddings from text
        
        Args:
            text_list: List of texts to embed
            enable_learning: If True, compute gradients for backpropagation
            
        Returns:
            Normalized sentence embeddings tensor
        """
        import torch.nn.functional as F
        
        encoded_input = self.tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        if enable_learning:
            model_output = self.model(**encoded_input)
        else:
            with torch.no_grad():
                model_output = self.model(**encoded_input)

        # Mean pooling with attention mask
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return F.normalize(sentence_embeddings, p=2, dim=1)
    
    def _load_training_log(self):
        """Load training history from JSON file"""
        if os.path.exists(TRAINING_LOG):
            try:
                with open(TRAINING_LOG, 'r') as f:
                    data = json.load(f)
                # Handle both formats: dict with 'trainings' key or direct list
                if isinstance(data, dict) and 'trainings' in data:
                    history = data['trainings']
                elif isinstance(data, list):
                    history = data
                else:
                    history = []
                print(f"📊 History loaded: {len(history)} previous sessions")
                return history
            except:
                return []
        return []
    
    def _save_training_log(self, epoch_data):
        """Save training epoch data to history log"""
        # Ensure training_history is a list
        if not isinstance(self.training_history, list):
            self.training_history = []
        self.training_history.append(epoch_data)
        with open(TRAINING_LOG, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def train(self, train_df, epochs=3):
        """
        Fine-tune the model using contrastive learning
        Trains the model to match exercise descriptions with their names
        
        Args:
            train_df: DataFrame with 'Full_Desc' and 'Exercise Name' columns
            epochs: Number of training epochs
        """
        from torch.optim import AdamW
        from torch.nn import CrossEntropyLoss
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        print(f"\n💪 STARTING TRAINING ({epochs} Epochs)...")
        self.model.train()
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        loss_fn = CrossEntropyLoss()
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-6)
        
        # Prepare data
        descriptions = train_df['Full_Desc'].tolist()
        names = train_df['Exercise Name'].tolist()
        batch_size = 16
        accum_steps = 4
        
        train_desc, val_desc, train_names, val_names = train_test_split(
            descriptions, names, test_size=0.1, random_state=42
        )

        # Save validation set
        self._save_validation_set(val_desc, val_names)

        # Training loop with early stopping
        patience = 2
        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(epochs):
            print(f"📈 Epoch {epoch+1}/{epochs} | LR: {scheduler.get_last_lr()[0]:.2e}")
            
            # Train one epoch
            avg_loss, steps = self._train_epoch(train_desc, train_names, batch_size, 
                                                accum_steps, optimizer, loss_fn)
            
            # Validate
            val_acc, val_loss = self._validate(val_desc, val_names, batch_size, loss_fn)
            
            print(f"   ├─ Train Loss: {avg_loss:.4f}")
            print(f"   └─ Val Accuracy: {val_acc:.2f}%")

            # Log metrics
            epoch_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch + 1,
                "train_loss": round(avg_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 2),
                "learning_rate": scheduler.get_last_lr()[0]
            }
            self._save_training_log(epoch_data)

            # Save best model and early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), MODEL_PATH)
                print(f"   ✅ Best model saved!")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"⚠️  Early stopping (patience={patience})")
                    break

            scheduler.step()

        torch.save(self.model.state_dict(), MODEL_PATH)
        self.model.eval()
        print("✅ Training Complete!\n")

    def train_real(self, train_df, epochs=3):
        """
        Backward-compatible alias for training.
        Some scripts still call `train_real`.
        """
        return self.train(train_df, epochs=epochs)
    
    def _train_epoch(self, train_desc, train_names, batch_size, accum_steps, optimizer, loss_fn):
        """Train for one epoch"""
        total_loss = 0
        steps = 0
        
        # Shuffle data
        perm = np.random.permutation(len(train_desc))
        epoch_desc = [train_desc[i] for i in perm]
        epoch_names = [train_names[i] for i in perm]

        for i in range(0, len(epoch_desc), batch_size):
            batch_d = epoch_desc[i:i+batch_size]
            batch_n = epoch_names[i:i+batch_size]
            if len(batch_d) < 2:
                continue

            # Compute embeddings and similarity scores
            emb_desc = self.get_embeddings(batch_d, enable_learning=True)
            emb_names = self.get_embeddings(batch_n, enable_learning=True)
            scores = torch.mm(emb_desc, emb_names.transpose(0, 1)) * 20.0
            
            # Contrastive loss: diagonal should be highest
            labels = torch.arange(len(scores)).to(self.device)
            loss = loss_fn(scores, labels) / accum_steps

            loss.backward()
            total_loss += loss.item() * accum_steps
            
            if (i // batch_size + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                steps += 1

        if (len(epoch_desc) // batch_size) % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
            steps += 1
        
        avg_loss = total_loss / steps if steps > 0 else 0.0
        return avg_loss, steps
    
    def _validate(self, val_desc, val_names, batch_size, loss_fn):
        """Validate model on validation set"""
        self.model.eval()
        val_total = 0
        val_correct = 0
        val_total_loss = 0.0
        val_steps = 0
        
        for i in range(0, len(val_desc), batch_size):
            batch_d = val_desc[i:i+batch_size]
            batch_n = val_names[i:i+batch_size]
            if len(batch_d) < 2:
                continue

            emb_desc = self.get_embeddings(batch_d, enable_learning=False)
            emb_names = self.get_embeddings(batch_n, enable_learning=False)
            scores = torch.mm(emb_desc, emb_names.transpose(0, 1)) * 20.0
            labels = torch.arange(len(scores)).to(self.device)

            preds = torch.argmax(scores, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += len(labels)

            vloss = loss_fn(scores, labels)
            val_total_loss += vloss.item()
            val_steps += 1

        val_acc = (val_correct / val_total * 100) if val_total > 0 else 0.0
        val_loss = (val_total_loss / val_steps) if val_steps > 0 else float('inf')
        
        self.model.train()
        return val_acc, val_loss
    
    def _save_validation_set(self, val_desc, val_names):
        """Save validation set for testing purposes"""
        validation_data = {'descriptions': val_desc, 'names': val_names}
        VALIDATION_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(VALIDATION_SET_PATH, 'w', encoding='utf-8') as f:
            json.dump(validation_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Validation set saved: {len(val_names)} samples")
    
    def build_database(self, train_df):
        """
        Build searchable database from training data
        Creates embeddings for all exercises for fast similarity search
        
        Args:
            train_df: DataFrame with exercise data
        """
        print("\n📚 Building Exercise Database...")
        train_desc = train_df['Full_Desc'].tolist()
        train_names = train_df['Exercise Name'].values
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(train_desc), 64):
            batch = train_desc[i:i+64]
            all_embeddings.append(self.get_embeddings(batch, enable_learning=False))
        
        self.database = torch.cat(all_embeddings, dim=0)
        self.database_names = train_names
        self.database_descriptions = train_desc
        self._save_database()
    
    def _save_database(self):
        """Save database to disk"""
        data = {
            'tensor': self.database.cpu(),
            'names': self.database_names,
            'descriptions': self.database_descriptions,
            'timestamp': datetime.now().isoformat()
        }
        with open(DATABASE_PATH, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Database saved: {len(self.database_names)} indexed exercises")
    
    def load_database(self, force=False):
        """Load pre-built database from disk.

        Args:
            force: If True, build database from CSV when missing.
        """
        if os.path.exists(DATABASE_PATH):
            try:
                with open(DATABASE_PATH, 'rb') as f:
                    data = pickle.load(f)
                self.database = data['tensor'].to(self.device)
                self.database_names = data['names']
                self.database_descriptions = data.get('descriptions', None)
                print(f"✅ Database loaded: {len(self.database_names)} exercises")
                return True
            except:
                return False

        if force:
            try:
                if not FILE_NAME.exists():
                    print(f"❌ Dataset not found: {FILE_NAME}")
                    return False

                print("⚠️  Database missing. Building from CSV...")
                df = pd.read_csv(FILE_NAME)
                df['Full_Desc'] = (df['Preparation'].fillna('') + " " + df['Execution'].fillna('')).str.strip()
                df = df[df['Full_Desc'] != '']

                self.build_database(df)
                return True
            except Exception as e:
                print(f"❌ Failed to build database: {str(e)}")
                return False

        return False
    
    def search(self, query, top_k=5):
        """
        Search for exercises similar to the query
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            
        Returns:
            List of dicts with 'exercise' and 'similarity' keys
        """
        if self.database is None:
            return []
        
        query_vec = self.get_embeddings([query], enable_learning=False)
        scores = torch.mm(query_vec, self.database.transpose(0, 1)).squeeze()
        
        top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
        
        results = []
        for score, idx in zip(top_scores, top_indices):
            results.append({
                'exercise': self.database_names[idx.item()],
                'similarity': round(score.item(), 4)
            })
        return results

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================
def load_and_prepare_data():
    """
    Load CSV data and prepare for training
    
    Returns:
        Tuple of (train_df, test_df)
    """
    if not FILE_NAME.exists():
        print(f"\n❌ ERROR: {FILE_NAME.name} not found")
        print(f"   Expected location: {FILE_NAME.absolute()}")
        sys.exit(1)
    
    print("\n📖 Loading data...")
    df = pd.read_csv(FILE_NAME)
    print(f"   Loaded {len(df)} exercises")
    
    # Combine preparation and execution into full description
    df['Full_Desc'] = (df['Preparation'].fillna('') + " " + df['Execution'].fillna('')).str.strip()
    df = df[df['Full_Desc'] != '']
    print(f"   Processed {len(df)} valid exercises")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"   Training set: {len(train_df)}, Test set: {len(test_df)}")
    
    return train_df, test_df

def load_test_data():
    """Load test data for displaying examples"""
    try:
        if not FILE_NAME.exists():
            return None

        df = pd.read_csv(FILE_NAME)
        if 'Preparation' not in df.columns or 'Execution' not in df.columns:
            return None

        df['Full_Desc'] = (df['Preparation'].fillna('') + " " + df['Execution'].fillna('')).str.strip()
        df = df[df['Full_Desc'] != '']

        if len(df) == 0:
            return None

        _, test_df = train_test_split(df, test_size=0.2, random_state=42)
        return test_df
    except Exception as e:
        print(f"\n⚠️  Could not load test data: {str(e)}")
        return None

def load_test_df_from_csv():
    """
    Load test dataframe from CSV for external use
    Returns the test split of the dataset
    """
    return load_test_data()

def validate_test_results(brain, test_df, num_samples=5):
    """
    Test model on validation samples and print results
    
    Args:
        brain: GymBrain instance
        test_df: Test dataframe
        num_samples: Number of samples to test
    """
    print("\n🔍 VALIDATION TEST - Testing on samples:")
    print("=" * 50)
    test_samples = test_df.head(num_samples)
    
    correct = 0
    for _, row in test_samples.iterrows():
        query = row['Full_Desc']
        correct_answer = row['Exercise Name']

        results = brain.search(query, top_k=3)
        
        print(f"\n🎯 Expected: {correct_answer}")
        print(f"📝 Query: {query[:80]}...")
        print(f"🏆 Best Results:")
        for i, result in enumerate(results, 1):
            similarity = result['similarity'] * 100
            print(f"   {i}. {result['exercise']} ({similarity:.1f}%)")
            if i == 1 and result['exercise'].lower() == correct_answer.lower():
                correct += 1
        print("-" * 50)
    
    accuracy = (correct / len(test_samples)) * 100
    print(f"\n📊 Validation Accuracy: {accuracy:.1f}% ({correct}/{len(test_samples)})")

# ============================================================================
# UI/MENU FUNCTIONS (All user interface logic at the end)
# ============================================================================
def show_test_examples(test_df, num_examples=5):
    """Display test set examples that model never saw during training"""
    
    if test_df is None or len(test_df) == 0:
        print("\n❌ ERROR: No test data available")
        return
    
    required_columns = ['Exercise Name', 'Full_Desc']
    missing_columns = [col for col in required_columns if col not in test_df.columns]
    
    if missing_columns:
        print(f"\n❌ ERROR: Test data missing required columns: {', '.join(missing_columns)}")
        return
    
    try:
        import random
        num_to_sample = min(num_examples, len(test_df))
        samples = test_df.sample(n=num_to_sample, random_state=42)
    except Exception as e:
        print(f"\n⚠️  Warning: Could not sample randomly ({str(e)})")
        samples = test_df.head(num_examples)
    
    print(f"\n{'='*60}")
    print("🧪 TEST SET EXAMPLES (Model NEVER saw these during training)")
    print(f"{'='*60}")
    print(f"📊 Test examples shown at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"📊 Total test samples available: {len(test_df)}")
    print(f"\n💡 These are REAL examples the model has never seen.")
    print("   Use them as inspiration for your searches!\n")
    print(f"{'-'*60}\n")
    
    for i, (_, row) in enumerate(samples.iterrows(), 1):
        try:
            exercise_name = str(row['Exercise Name'])
            description = str(row['Full_Desc'])
            
            if len(description) > 120:
                description = description[:120] + "..."
            
            print(f"{i}. 🏋️  {exercise_name.upper()}")
            print(f"   📝 {description}")
            print(f"   {'─'*50}")
        except Exception as e:
            print(f"{i}. ⚠️  Error displaying example: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print(f"✅ Displayed {len(samples)} test examples successfully")
    print(f"{'='*60}\n")

def interactive_search_menu(brain, test_df=None):
    """
    Interactive search mode with adaptive menu
    Allows user to search exercises and view examples
    
    Args:
        brain: GymBrain instance
        test_df: Optional test dataframe for showing examples
    """
    print("\n" + "="*60)
    print("🔍 GYM AI - INTERACTIVE SEARCH")
    print("="*60)
    
    if not is_model_ready():
        print("\n⚠️  Access denied: Model not ready")
        print("   Please train or load the model first.")
        return
    
    can_show_test_examples = (
        brain.has_test_access and
        is_model_ready() and 
        test_df is not None and 
        len(test_df) > 0 and
        'Exercise Name' in test_df.columns and
        'Full_Desc' in test_df.columns
    )
    
    while True:
        has_data = brain.database is not None and brain.database_names is not None and len(brain.database_names) > 0
        
        print("\n📋 OPTIONS:")
        print("   1. Search exercise")
        if can_show_test_examples:
            print("   3. 🧪 TEST examples (model never saw these)")
        print(f"   {4 if can_show_test_examples else 3 if has_data else 2}. Exit")
        
        max_option = 4 if can_show_test_examples else (3 if has_data else 2)
        choice = input(f"\n👉 Select an option (1-{max_option}): ").strip()
        
        if choice == "1":
            if not has_data:
                print("\n❌ No database loaded. Please train or load the model first.")
                continue
                
            print("\n" + "-"*60)
            print("💪 Find Your Perfect Exercise")
            print("   (Example: chest, back, arms, cardio, etc.)")
            query = input("\n👉 Describe the exercise you are looking for: ").strip()
            
            if query:
                print("\n🔎 Searching...")
                results = brain.search(query, top_k=5)
                
                if results:
                    print("\n🏆 TOP RESULTS:")
                    print("-"*60)
                    for i, result in enumerate(results, 1):
                        similarity_percent = result['similarity'] * 100
                        print(f"\n   {i}. {result['exercise']}")
                        print(f"      Similarity: {similarity_percent:.1f}%")
                else:
                    print("❌ No results found")
            else:
                print("❌ Please enter a valid description")
        
        elif choice == "2":
            if has_data:
                print("\n" + "-"*60)
                print("📚 RANDOM EXERCISE EXAMPLES (from training data):")
                print("-"*60)
                
                import random
                names_list = list(brain.database_names) if not isinstance(brain.database_names, list) else brain.database_names
                num_examples = min(10, len(names_list))
                indices = random.sample(range(len(names_list)), num_examples)
                
                for i, idx in enumerate(indices, 1):
                    exercise_name = names_list[idx]
                    print(f"\n   {i}. {exercise_name}")
                    
                    if brain.database_descriptions and idx < len(brain.database_descriptions):
                        desc = brain.database_descriptions[idx]
                        if len(desc) > 100:
                            desc = desc[:100] + "..."
                        print(f"      📝 {desc}")
            else:
                print("\n👋 Goodbye!")
                break
                
        elif choice == "3":
            if can_show_test_examples:
                show_test_examples(test_df, num_examples=8)
            elif has_data:
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Try again.")
        
        elif choice == "4" and can_show_test_examples:
            print("\n👋 Goodbye!")
            break
        
        elif choice == str(max_option):
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option. Try again.")

def main_menu():
    """Main menu for terminal interface"""
    if not os.path.exists(FILE_NAME):
        print(f"❌ {FILE_NAME} not found")
        return

    print("\n" + "="*60)
    print("🏋️ WELCOME TO GYM AI - MENU")
    print("="*60)
    print("\n📋 MAIN MENU:")
    print("   1. Train Neural Network (Train Only)")
    print("   2. Smart Search (Train or Load & Search)")
    print("   3. Continue Training (Improve Pre-trained Model)")
    print("   4. Exit")
    
    choice = input("\n👉 Select an option (1-4): ").strip()
    
    if choice == "1":
        # Train only mode
        train_df, test_df = load_and_prepare_data()
        brain = GymBrain(load_pretrained_weights=False)
        brain.train(train_df, epochs=6)
        brain.build_database(train_df)
        validate_test_results(brain, test_df, num_samples=5)
        save_model_state(True, "trained")
        print("\n✅ Training completed! Model saved.")
        print("   Use Option 2 to search or improve the model.")
        input("\nPress Enter to return to menu...")
    
    elif choice == "2":
        # Smart search submenu
        print("\n" + "="*60)
        print("🔍 SMART SEARCH")
        print("="*60)
        print("\n📋 SUBMENU:")
        print("   1. Train New Model (Fresh Start)")
        print("   2. Load Pre-trained Model (Search Only)")
        
        sub_choice = input("\n👉 Select an option (1-2): ").strip()
        
        if sub_choice == "1":
            train_df, test_df = load_and_prepare_data()
            brain = GymBrain(load_pretrained_weights=False)
            brain.train(train_df, epochs=6)
            brain.build_database(train_df)
            validate_test_results(brain, test_df, num_samples=5)
            save_model_state(True, "trained")
            brain.has_test_access = True
            print("\n✅ Model trained successfully!")
            interactive_search_menu(brain, test_df)
        
        elif sub_choice == "2":
            action = input("\n👉 Train with loaded weights? (y/n): ").strip().lower()

            if action in ("y", "yes", "s", "si", "sí"):
                if not os.path.exists(MODEL_PATH):
                    print("\n⚠️  Pre-trained model not found")
                    input("\nPress Enter to return to menu...")
                    return

                train_df, test_df = load_and_prepare_data()
                brain = GymBrain(load_pretrained_weights=True)
                brain.train(train_df, epochs=6)
                brain.build_database(train_df)
                save_model_state(True, "trained")
                brain.has_test_access = True
                print("\n✅ Model trained successfully!")
                interactive_search_menu(brain, test_df)
            else:
                if not os.path.exists(DATABASE_PATH):
                    print("\n⚠️  Pre-trained database not found")
                    input("\nPress Enter to return to menu...")
                    return

                brain = GymBrain(load_pretrained_weights=False)
                if brain.load_database():
                    save_model_state(True, "preloaded")
                    brain.has_test_access = True
                    test_df = load_test_data()
                    print("\n✅ Model loaded successfully!")
                    print(f"📚 Indexed exercises: {len(brain.database_names)}")
                    interactive_search_menu(brain, test_df)
                else:
                    print("\n❌ Error loading database")
                    input("\nPress Enter to return to menu...")
    
    elif choice == "3":
        # Continue training mode
        if not os.path.exists(DATABASE_PATH):
            print("\n⚠️  Pre-trained database not found")
            input("\nPress Enter to return to menu...")
            return
        
        brain = GymBrain(load_pretrained_weights=True)
        if not brain.load_database():
            print("\n❌ Error loading database")
            input("\nPress Enter to return to menu...")
            return
        
        train_df, test_df = load_and_prepare_data()
        brain.train(train_df, epochs=3)
        brain.build_database(train_df)
        validate_test_results(brain, test_df, num_samples=5)
        save_model_state(True, "trained")
        brain.has_test_access = True
        print("\n✅ Model improved and saved!")
        interactive_search_menu(brain, test_df)
    
    elif choice == "4":
        print("\n👋 Goodbye!")
    else:
        print("❌ Invalid option")

def start_api_server(port=5000):
    """Start Flask API server for Flutter frontend"""
    from flask import Flask, request, jsonify
    
    print("\n" + "="*60)
    print("🚀 STARTING API SERVER")
    print("="*60)
    
    if not os.path.exists(DATABASE_PATH):
        print("\n⚠️  Database not found. It will be built from CSV on first load.")
    
    VALIDATION_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(VALIDATION_SET_PATH):
        print("\n⚠️  Validation set not found, creating placeholder...")
        default_validation = {'descriptions': ['placeholder'], 'names': ['Placeholder']}
        with open(VALIDATION_SET_PATH, 'w', encoding='utf-8') as f:
            json.dump(default_validation, f, indent=2, ensure_ascii=False)
    
    brain = GymBrain(load_pretrained_weights=False)
    
    if not brain.load_database(force=True):
        print("\n❌ ERROR: Could not load database")
        return
    
    app = Flask(__name__)
    
    @app.route('/api/search', methods=['POST'])
    def api_search():
        try:
            data = request.get_json()
            query = data.get('query', '').strip()
            
            if not query:
                return jsonify({'error': 'Empty query'}), 400
            
            results = brain.search(query, top_k=5)
            
            if not results:
                return jsonify({'success': False, 'results': [], 'message': 'No results'}), 200
            
            return jsonify({'success': True, 'results': results})
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/validation_set.json', methods=['GET'])
    def get_validation_set():
        try:
            if os.path.exists(VALIDATION_SET_PATH):
                with open(VALIDATION_SET_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return jsonify(data)
            else:
                return jsonify({'descriptions': [], 'names': []}), 404
        except Exception as e:
            print(f"❌ Error loading validation set: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    print(f"\n✅ API Server running on http://localhost:{port}")
    print(f"   Endpoint: POST http://localhost:{port}/api/search")
    print(f"\n   Waiting for connections from Flutter...")
    print(f"   Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)

# ============================================================================
# MAIN EXECUTION LOGIC
# ============================================================================
if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'api':
            start_api_server()
        
        elif sys.argv[1] == '1':
            # CLI: Train mode
            train_df, test_df = load_and_prepare_data()
            brain = GymBrain(load_pretrained_weights=False)
            brain.train(train_df, epochs=6)
            brain.build_database(train_df)
            validate_test_results(brain, test_df, num_samples=5)
            save_model_state(True, "trained")
            brain.has_test_access = True
            print(f"✅ Model saved to: {MODEL_PATH.name}")
            print(f"✅ Database saved to: {DATABASE_PATH.name}")
            interactive_search_menu(brain, test_df)
        
        elif sys.argv[1] == '3':
            # CLI: Search mode
            if not os.path.exists(DATABASE_PATH) or not os.path.exists(TRAINING_LOG):
                print("\n⚠️  Database or training log not found")
                print("   Please train the model first")
                sys.exit(1)
            
            brain = GymBrain(load_pretrained_weights=False)
            if not brain.load_database():
                print("\n❌ Error loading database")
                sys.exit(1)
            
            if len(sys.argv) > 2:
                # Direct search from command line
                query = sys.argv[2]
                results = brain.search(query, top_k=5)
                print("🏆 TOP RESULTS:")
                print("-" * 50)
                for i, result in enumerate(results, 1):
                    similarity_percent = result['similarity'] * 100
                    print(f"\n   {i}. {result['exercise']}")
                    print(f"      Similarity: {similarity_percent:.1f}%")
            else:
                # Interactive mode
                test_df = load_test_data() if is_model_ready() else None
                interactive_search_menu(brain, test_df)
        
        elif sys.argv[1] == '4':
            # CLI: Load and optionally continue training
            action = input("\n👉 Train with loaded weights? (y/n): ").strip().lower()

            if action in ("y", "yes", "s", "si", "sí"):
                if not os.path.exists(MODEL_PATH):
                    print("\n⚠️  Pre-trained model not found")
                    sys.exit(1)

                train_df, test_df = load_and_prepare_data()
                brain = GymBrain(load_pretrained_weights=True)
                brain.train(train_df, epochs=6)
                brain.build_database(train_df)
                save_model_state(True, "trained")
                brain.has_test_access = True
                print("\n✅ Training completed successfully!")
                interactive_search_menu(brain, test_df)
            else:
                if not os.path.exists(DATABASE_PATH):
                    print("\n⚠️  Pre-trained database not found")
                    sys.exit(1)

                brain = GymBrain(load_pretrained_weights=True)
                if brain.load_database():
                    save_model_state(True, "preloaded")
                    brain.has_test_access = True
                    test_df = load_test_data()
                    print("\n✅ Model loaded successfully!")
                    print(f"📚 Indexed exercises: {len(brain.database_names)}")
                    interactive_search_menu(brain, test_df)
                else:
                    print("\n❌ Error loading database")
                    sys.exit(1)
    else:
        main_menu()