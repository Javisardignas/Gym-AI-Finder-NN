"""
Visualization Module for GYM AI
Provides progress tracking and evolution visualization
"""

import json
import os
from pathlib import Path
from datetime import datetime

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
TRAINING_LOG = SCRIPT_DIR / "training_log.json"

class ProgressVisualizer:
    """Visualize training progress and model evolution"""
    
    def __init__(self):
        """Initialize the visualizer"""
        self.training_history = self._load_training_log()
    
    def _load_training_log(self):
        """Load training history from JSON file"""
        if os.path.exists(TRAINING_LOG):
            try:
                with open(TRAINING_LOG, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def show_summary(self):
        """Display training summary"""
        if not self.training_history:
            print("\n⚠️  No training history found")
            return
        
        print("\n" + "="*60)
        print("📊 TRAINING HISTORY SUMMARY")
        print("="*60)
        print(f"\n📈 Total training sessions: {len(self.training_history)}")
        
        if self.training_history:
            latest = self.training_history[-1]
            print(f"🕐 Last training: {latest.get('timestamp', 'Unknown')}")
            print(f"📉 Final train loss: {latest.get('train_loss', 'N/A')}")
            print(f"✅ Final validation accuracy: {latest.get('val_accuracy', 'N/A')}%")
    
    def show_detailed_progress(self):
        """Display detailed epoch-by-epoch progress"""
        if not self.training_history:
            print("\n⚠️  No training history found")
            return
        
        print("\n" + "="*60)
        print("📊 DETAILED TRAINING PROGRESS")
        print("="*60)
        
        for i, epoch_data in enumerate(self.training_history, 1):
            timestamp = epoch_data.get('timestamp', 'Unknown')
            epoch = epoch_data.get('epoch', i)
            train_loss = epoch_data.get('train_loss', 'N/A')
            val_loss = epoch_data.get('val_loss', 'N/A')
            val_acc = epoch_data.get('val_accuracy', 'N/A')
            lr = epoch_data.get('learning_rate', 'N/A')
            
            print(f"\n📈 Epoch {epoch} - {timestamp}")
            print(f"   ├─ Train Loss: {train_loss}")
            print(f"   ├─ Val Loss: {val_loss}")
            print(f"   ├─ Val Accuracy: {val_acc}%")
            print(f"   └─ Learning Rate: {lr}")
    
    def show_best_performance(self):
        """Display best performance achieved"""
        if not self.training_history:
            print("\n⚠️  No training history found")
            return
        
        best_epoch = max(self.training_history, 
                        key=lambda x: x.get('val_accuracy', 0))
        
        print("\n" + "="*60)
        print("🏆 BEST PERFORMANCE")
        print("="*60)
        print(f"\n🎯 Best validation accuracy: {best_epoch.get('val_accuracy', 'N/A')}%")
        print(f"📈 Achieved in epoch: {best_epoch.get('epoch', 'N/A')}")
        print(f"🕐 Date: {best_epoch.get('timestamp', 'Unknown')}")
        print(f"📉 Train loss: {best_epoch.get('train_loss', 'N/A')}")
        print(f"📊 Val loss: {best_epoch.get('val_loss', 'N/A')}")

def show_main_evolution_menu():
    """Display main evolution and progress menu"""
    visualizer = ProgressVisualizer()
    
    while True:
        print("\n" + "="*60)
        print("📊 MODEL EVOLUTION & PROGRESS")
        print("="*60)
        print("\n📋 OPTIONS:")
        print("   1. Show Training Summary")
        print("   2. Show Detailed Progress")
        print("   3. Show Best Performance")
        print("   4. Return to Main Menu")
        
        choice = input("\n👉 Select an option (1-4): ").strip()
        
        if choice == "1":
            visualizer.show_summary()
            input("\nPress Enter to continue...")
        
        elif choice == "2":
            visualizer.show_detailed_progress()
            input("\nPress Enter to continue...")
        
        elif choice == "3":
            visualizer.show_best_performance()
            input("\nPress Enter to continue...")
        
        elif choice == "4":
            print("\n👋 Returning to main menu...")
            break
        
        else:
            print("❌ Invalid option. Try again.")
