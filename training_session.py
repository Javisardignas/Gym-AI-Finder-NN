"""
Training Session Management - Persistent session state across application runs
Tracks model usage, test queries, accuracy evolution, and training history
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


class TrainingSession:
    """
    Manages persistent session state across application runs.
    
    Saves to: current_session.json
    
    Structure:
    {
        "current_model": "gym_model_v2",
        "last_updated": "2024-01-15T14:30:00",
        "tests": [
            {
                "query": "bench press",
                "accuracy": 65.5,
                "timestamp": "2024-01-15T14:30:00",
                "model": "gym_model_v2"
            }
        ],
        "trainings": [
            {
                "model_created": "gym_model_v1",
                "accuracy_before": 40.0,
                "accuracy_after": 52.3,
                "improvement": 12.3,
                "epochs": 6,
                "timestamp": "2024-01-15T14:15:00"
            }
        ]
    }
    """
    
    def __init__(self, session_file: str = "current_session.json"):
        self.session_file = Path(session_file)
        self.session = self.load_session()
    
    def load_session(self) -> Dict[str, Any]:
        """Load session from JSON or create new one"""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return self._create_empty_session()
        return self._create_empty_session()
    
    def _create_empty_session(self) -> Dict[str, Any]:
        """Create empty session structure"""
        return {
            "current_model": None,
            "last_updated": None,
            "tests": [],
            "trainings": []
        }
    
    def save_session(self):
        """Save session to JSON"""
        self.session['last_updated'] = datetime.now().isoformat()
        with open(self.session_file, 'w', encoding='utf-8') as f:
            json.dump(self.session, f, indent=2, ensure_ascii=False)
    
    def set_current_model(self, model_id: Optional[str]):
        """Set the currently active model"""
        self.session['current_model'] = model_id
        self.save_session()
        print(f"✅ Session updated: Current model = {model_id or 'None'}")
    
    def get_current_model(self) -> Optional[str]:
        """Get the currently active model"""
        return self.session.get('current_model')
    
    def record_test(self, query: str, accuracy: float, model_id: Optional[str] = None):
        """
        Record a test query with accuracy result
        
        Args:
            query: Search query performed
            accuracy: Accuracy achieved (0-100)
            model_id: Model used (defaults to current model)
        """
        if 'tests' not in self.session:
            self.session['tests'] = []
        
        test_record = {
            "query": query,
            "accuracy": round(accuracy, 2),
            "timestamp": datetime.now().isoformat(),
            "model": model_id or self.session.get('current_model', 'base')
        }
        
        self.session['tests'].append(test_record)
        self.save_session()
    
    def record_training(self,
                       model_id: str,
                       accuracy_before: float,
                       accuracy_after: float,
                       epochs: int,
                       training_type: str = "full"):
        """
        Record a training session
        
        Args:
            model_id: ID of the model created
            accuracy_before: Accuracy before training
            accuracy_after: Accuracy after training
            epochs: Number of epochs trained
            training_type: 'full' or 'finetuning'
        """
        if 'trainings' not in self.session:
            self.session['trainings'] = []
        
        training_record = {
            "model_created": model_id,
            "accuracy_before": round(accuracy_before, 2),
            "accuracy_after": round(accuracy_after, 2),
            "improvement": round(accuracy_after - accuracy_before, 2),
            "epochs": epochs,
            "training_type": training_type,
            "timestamp": datetime.now().isoformat()
        }
        
        self.session['trainings'].append(training_record)
        self.save_session()
    
    def get_test_count(self) -> int:
        """Get total number of tests performed"""
        return len(self.session.get('tests', []))
    
    def get_training_count(self) -> int:
        """Get total number of trainings performed"""
        return len(self.session.get('trainings', []))
    
    def get_last_training(self) -> Optional[Dict]:
        """Get last training session"""
        trainings = self.session.get('trainings', [])
        return trainings[-1] if trainings else None
    
    def get_average_accuracy_improvement(self) -> float:
        """Get average accuracy improvement across all trainings"""
        trainings = self.session.get('trainings', [])
        if not trainings:
            return 0.0
        
        total_improvement = sum(t.get('improvement', 0) for t in trainings)
        return total_improvement / len(trainings)
    
    def get_training_history(self) -> List[Dict]:
        """Get all training sessions"""
        return self.session.get('trainings', [])
    
    def get_test_history(self) -> List[Dict]:
        """Get all test queries"""
        return self.session.get('tests', [])
    
    def print_session_summary(self):
        """Print formatted session summary"""
        current_model = self.get_current_model()
        test_count = self.get_test_count()
        training_count = self.get_training_count()
        
        print("\n📊 SESSION SUMMARY:")
        print("=" * 60)
        print(f"   🧠 Current Model: {current_model or 'Base Embeddings'}")
        print(f"   🧪 Total Tests: {test_count}")
        print(f"   📚 Total Trainings: {training_count}")
        
        if training_count > 0:
            avg_improvement = self.get_average_accuracy_improvement()
            print(f"   📈 Average Improvement: {avg_improvement:.2f}%")
        
        last_training = self.get_last_training()
        if last_training:
            print(f"   ⏰ Last Training: {last_training['timestamp'][:10]}")
            print(f"      Model: {last_training['model_created']}")
            print(f"      Accuracy: {last_training['accuracy_after']}%")
        
        print("=" * 60)
    
    def print_detailed_history(self):
        """Print detailed training and testing history"""
        print("\n📚 DETAILED SESSION HISTORY:")
        print("=" * 70)
        
        # Training history
        trainings = self.get_training_history()
        if trainings:
            print("\n🚀 TRAINING SESSIONS:")
            print("-" * 70)
            for i, training in enumerate(trainings, 1):
                print(f"\n{i}. Model Created: {training['model_created']}")
                print(f"   📅 {training['timestamp'][:10]}")
                print(f"   📈 Accuracy: {training['accuracy_before']}% → {training['accuracy_after']}%")
                print(f"   📊 Improvement: +{training['improvement']}%")
                print(f"   🔧 Epochs: {training['epochs']} ({training['training_type']})")
        else:
            print("\n   No training sessions yet")
        
        # Test history
        tests = self.get_test_history()
        if tests:
            print("\n\n🧪 TEST QUERIES:")
            print("-" * 70)
            for i, test in enumerate(tests[-5:], 1):  # Show last 5
                print(f"\n{i}. Query: {test['query'][:50]}")
                print(f"   📅 {test['timestamp'][:10]}")
                print(f"   🎯 Accuracy: {test['accuracy']}%")
                print(f"   🧠 Model: {test['model']}")
        else:
            print("\n   No tests performed yet")
        
        print("\n" + "=" * 70)
