"""
Model Registry System - Manages multiple model versions with persistent storage
Handles versioning, metadata, and model lifecycle management
"""

import json
import torch
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class ModelRegistry:
    """
    Manages multiple trained models with version control and metadata tracking.
    
    Structure:
    models/
    ├── gym_model_v1.pt          # Model weights
    ├── gym_model_v1_database.pkl # Indexed exercises database
    ├── gym_model_v1_metadata.json # Metadata (accuracy, date, etc.)
    ├── gym_model_v2.pt
    ├── gym_model_v2_database.pkl
    ├── gym_model_v2_metadata.json
    └── models_registry.json      # Registry of all models
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.registry_file = self.models_dir / "models_registry.json"
        self.registry = self.load_registry()
        
    def load_registry(self) -> Dict:
        """Load registry from JSON file"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_registry(self):
        """Save registry to JSON file"""
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def _prune_missing_models(self) -> bool:
        """Remove registry entries whose files are missing on disk"""
        if not self.registry:
            return False

        removed = False
        for model_id, info in list(self.registry.items()):
            model_file = info.get("model_file")
            database_file = info.get("database_file")

            model_path = self.models_dir / model_file if model_file else None
            db_path = self.models_dir / database_file if database_file else None

            if not model_path or not model_path.exists() or not db_path or not db_path.exists():
                self.registry.pop(model_id, None)
                removed = True

        if removed:
            self.save_registry()
        return removed
    
    def get_next_model_id(self) -> str:
        """Generate next model version ID"""
        if not self.registry:
            return "gym_model_v1"
        
        # Extract version numbers from existing model IDs
        versions = []
        for model_id in self.registry.keys():
            if model_id.startswith("gym_model_v"):
                try:
                    version = int(model_id.split("v")[-1])
                    versions.append(version)
                except:
                    pass
        
        next_version = max(versions) + 1 if versions else 1
        return f"gym_model_v{next_version}"
    
    def create_new_model(self, 
                        brain,
                        model_state_dict,
                        database_tensor,
                        database_names,
                        database_descriptions,
                        test_accuracy: float,
                        epochs_trained: int,
                        training_samples: int,
                        training_type: str = "full") -> Tuple[str, float]:
        """
        Create and register a new model with all its files
        
        Args:
            brain: GymBrain instance (for model compatibility)
            model_state_dict: Model weights to save
            database_tensor: Database embeddings tensor
            database_names: Exercise names in database
            database_descriptions: Exercise descriptions
            test_accuracy: Accuracy achieved on test set
            epochs_trained: Number of epochs trained
            training_samples: Number of training samples
            training_type: 'full' or 'finetuning'
        
        Returns:
            (model_id, accuracy)
        """
        model_id = self.get_next_model_id()
        
        # Save model weights
        model_path = self.models_dir / f"{model_id}.pt"
        torch.save(model_state_dict, model_path)
        print(f"   ✅ Model saved: {model_path.name}")
        
        # Save database
        db_data = {
            'tensor': database_tensor.cpu(),
            'names': database_names,
            'descriptions': database_descriptions,
            'timestamp': datetime.now().isoformat()
        }
        db_path = self.models_dir / f"{model_id}_database.pkl"
        with open(db_path, 'wb') as f:
            pickle.dump(db_data, f)
        print(f"   ✅ Database saved: {db_path.name}")
        
        # Save metadata
        metadata = {
            "model_id": model_id,
            "accuracy": round(test_accuracy, 2),
            "created": datetime.now().isoformat(),
            "epochs_trained": epochs_trained,
            "training_samples": training_samples,
            "training_type": training_type,
            "database_size": len(database_names),
            "model_file": str(model_path.name),
            "database_file": str(db_path.name)
        }
        
        # Register in main registry
        self.registry[model_id] = metadata
        self.save_registry()
        
        print(f"   ✅ Registered: {model_id}")
        
        return model_id, test_accuracy
    
    def overwrite_model(self, 
                       model_id: str,
                       brain,
                       model_state_dict,
                       database_tensor,
                       database_names,
                       database_descriptions,
                       test_accuracy: float,
                       epochs_trained: int,
                       training_samples: int,
                       training_type: str = "finetuning") -> str:
        """
        Overwrite an existing model with new weights and database
        
        Args:
            model_id: Model ID to overwrite
            brain: GymBrain instance (for model compatibility)
            model_state_dict: New model weights to save
            database_tensor: New database embeddings tensor
            database_names: New exercise names
            database_descriptions: New exercise descriptions
            test_accuracy: New accuracy achieved
            epochs_trained: Additional epochs trained
            training_samples: Training samples used
            training_type: 'finetuning' or 'full'
        
        Returns:
            model_id (same as input)
        """
        if model_id not in self.registry:
            print(f"❌ Model not found: {model_id}")
            return None
        
        # Save model weights
        model_path = self.models_dir / f"{model_id}.pt"
        torch.save(model_state_dict, model_path)
        print(f"   ✅ Model updated: {model_path.name}")
        
        # Save database
        db_data = {
            'tensor': database_tensor.cpu(),
            'names': database_names,
            'descriptions': database_descriptions,
            'timestamp': datetime.now().isoformat()
        }
        db_path = self.models_dir / f"{model_id}_database.pkl"
        with open(db_path, 'wb') as f:
            pickle.dump(db_data, f)
        print(f"   ✅ Database updated: {db_path.name}")
        
        # Update metadata
        metadata = self.registry[model_id]
        metadata['accuracy'] = round(test_accuracy, 2)
        metadata['epochs_trained'] = metadata.get('epochs_trained', 0) + epochs_trained
        metadata['training_samples'] = training_samples
        metadata['last_updated'] = datetime.now().isoformat()
        metadata['database_size'] = len(database_names)
        
        self.save_registry()
        print(f"   ✅ Model updated: {model_id}")
        
        return model_id
    
    def load_model(self, model_id: str, brain, device):
        """
        Load a specific model and its database
        
        Args:
            model_id: Model version to load
            brain: GymBrain instance to load weights into
            device: torch device (cuda/cpu)
        
        Returns:
            (brain, metadata) or (None, None) on failure
        """
        if model_id not in self.registry:
            print(f"❌ Model not found: {model_id}")
            return None, None
        
        metadata = self.registry[model_id]
        
        # Load model weights
        model_path = self.models_dir / f"{model_id}.pt"
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            return None, None
        
        try:
            weights = torch.load(model_path, map_location=device)
            brain.model.load_state_dict(weights)
            print(f"   ✅ Model weights loaded: {model_id}")
        except Exception as e:
            print(f"❌ Error loading model weights: {e}")
            return None, None
        
        # Load database
        db_path = self.models_dir / f"{model_id}_database.pkl"
        if not db_path.exists():
            print(f"❌ Database file not found: {db_path}")
            return None, None
        
        try:
            with open(db_path, 'rb') as f:
                db_data = pickle.load(f)
            
            brain.database = db_data['tensor'].to(device)
            brain.database_names = db_data['names']
            brain.database_descriptions = db_data.get('descriptions')
            print(f"   ✅ Database loaded: {len(brain.database_names)} exercises")
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            return None, None
        
        return brain, metadata
    
    def get_all_models(self) -> List[Tuple[str, Dict]]:
        """Get all models sorted by creation date (newest first)"""
        self._prune_missing_models()
        models = list(self.registry.items())
        models.sort(
            key=lambda x: x[1].get('created', ''),
            reverse=True
        )
        return models
    
    def get_best_model(self) -> Optional[Tuple[str, Dict]]:
        """Get model with highest accuracy"""
        if not self.registry:
            return None
        
        best = max(
            self.registry.items(),
            key=lambda x: x[1].get('accuracy', 0)
        )
        return best
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model and all its files"""
        if model_id not in self.registry:
            return False
        
        # Delete files
        model_path = self.models_dir / f"{model_id}.pt"
        db_path = self.models_dir / f"{model_id}_database.pkl"
        
        for path in [model_path, db_path]:
            try:
                if path.exists():
                    path.unlink()
            except:
                pass
        
        # Remove from registry
        del self.registry[model_id]
        self.save_registry()
        
        return True
    
    def get_accuracy_progression(self) -> List[Tuple[str, float]]:
        """Get models sorted by creation with accuracy for graph"""
        models = self.get_all_models()
        # Return in chronological order
        models.reverse()
        return [(model_id, info['accuracy']) for model_id, info in models]
    
    def print_model_list(self):
        """Print formatted list of all models"""
        models = self.get_all_models()
        
        if not models:
            print("\n📭 No trained models found yet")
            print("   Train a new model to get started!")
            return
        
        print("\n📚 AVAILABLE MODELS:")
        print("=" * 70)
        
        for i, (model_id, info) in enumerate(models, 1):
            created_date = info.get('created', '')[:10]
            accuracy = info.get('accuracy', 0)
            epochs = info.get('epochs_trained', 0)
            samples = info.get('training_samples', 0)
            training_type = info.get('training_type', 'unknown')
            
            print(f"\n{i}. {model_id}")
            print(f"   📅 Created: {created_date}")
            print(f"   🎯 Accuracy: {accuracy}%")
            print(f"   📊 Epochs: {epochs} | Samples: {samples}")
            print(f"   🔧 Type: {training_type}")
            print(f"   {'─' * 65}")
