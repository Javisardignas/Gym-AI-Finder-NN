#!/usr/bin/env python3
"""
Gym AI - Multi-Model Training System
Terminal Mode v2 - Adaptive menu with persistent model management

Sistema de entrenamiento neuronal con múltiples modelos, sesiones persistentes
y visualización de evolución de accuracy.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Import our new modules
from nngym_v2 import GymBrain, load_test_df_from_csv
from model_registry import ModelRegistry
from training_session import TrainingSession
from visualization import ProgressVisualizer, show_main_evolution_menu


class AdaptiveTerminalMenu:
    """Adaptive menu system that changes based on current model state"""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.session = TrainingSession()
        self.brain = None
        self.test_df = None
        self.current_model_id = None
        # Draft model (unsaved)
        self.draft_model_data = None
        self.draft_ready = False
        self.FILE_NAME = Path(__file__).parent / "gym_exercise_dataset.csv"
        
        # Load current state
        self.load_current_state()
    
    def load_current_state(self):
        """Load base embeddings by default (new model each session)"""
        # Always start with a new model, don't load existing ones automatically
        self.current_model_id = None
        self.brain = None
        print(f"\n✨ New session started - ready to train a new model!")
    
    def load_test_data(self):
        """Load test data if not already loaded"""
        if self.test_df is None:
            self.test_df = load_test_df_from_csv()
    
    def display_main_menu(self):
        """Display adaptive main menu based on current state"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("   =======================================================")
        print("              SYSTEM STATUS: BULKING IN PROGRESS")
        print("   =======================================================")
        print()
        print("\n" + "=" * 70)
        
        # Show current model status
        if self.current_model_id:
            models = self.registry.get_all_models()
            for model_id, info in models:
                if model_id == self.current_model_id:
                    accuracy = info.get('accuracy', 0)
                    
                    # Try to calculate improvement from training history
                    trainings = self.session.get_training_history()
                    improvement_text = ""
                    for training in reversed(trainings):
                        if training.get('model_id') == self.current_model_id:
                            acc_before = training.get('accuracy_before', 40)
                            improvement = accuracy - acc_before
                            improvement_text = f" | Improvement: +{improvement:.1f}%"
                            break
                    
                    print(f"🧠 CURRENT MODEL: {self.current_model_id} (Accuracy: {accuracy})")
                    break
        elif self.draft_ready:
            # Draft model with training data
            accuracy = self.draft_model_data.get('accuracy_after', 0)
            print(f"📝 DRAFT MODEL: Unsaved (accuracy: {accuracy:.1f}%) - Use Save option to persist")
        else:
            print(f"🧠 CURRENT MODEL: New/Empty - Ready to train")
        
        print("=" * 70)
        print("\n📋 MAIN MENU:")
        print("1. 🚀 Train Model")
        print("2. 🔍 Insert Description")
        print("3. 📋 Try Random Example")
        print("4. 📂 Load Different Model")
        print("5. 💾 Save Current Model")
        print("6. 🗑️ Delete Saved Models")
        print("7. 📊 View Training History")
        print("8. 📊 View Model Evolution")
        print("9. ❌ Exit")
        
        return input("\n👉 Select an option (1-9): ").strip()
    
    def test_search(self):
        """Test search functionality"""
        if self.current_model_id is None and self.brain is None:
            # Load base model
            print("\n📖 Loading base model...")
            self.brain = GymBrain()
            
            # Try to load any database
            if not self.brain.load_database(force=True):
                print("⚠️  No database available. Please train a model first.")
                return
        
        if self.brain.database is None:
            print("❌ No database loaded. Train a model first.")
            return
        
        print("\n" + "=" * 60)
        print("🔍 SEARCH MODE")
        print("=" * 60)
        
        query = input("\n👉 Describe the exercise you're looking for: ").strip()
        
        if not query:
            print("❌ Empty query")
            return
        
        print("\n🔎 Searching...")
        results = self.brain.search(query, top_k=5)
        
        if results:
            print("\n🏆 TOP RESULTS:")
            print("-" * 60)
            for i, result in enumerate(results, 1):
                similarity = result['similarity'] * 100
                print(f"{i}. {result['exercise']:<40} {similarity:.1f}%")
            print("-" * 60)
            
            # Record test in session
            accuracy = results[0]['similarity'] * 100 if results else 0
            self.session.record_test(query, accuracy, self.current_model_id)
        else:
            print("❌ No results found")
    
    def train_new_model(self):
        """Train or modify current draft model (does not auto-save)"""
        if self.brain is None:
            self.brain = GymBrain()
        
        print("\n" + "=" * 60)
        if self.draft_ready:
            print("🚀 CONTINUE TRAINING DRAFT MODEL")
            print("=" * 60)
            print(f"\nModifying existing draft...")
            epochs = 3
            training_type = "finetuning"
            accuracy_before = self.draft_model_data.get('accuracy_after', 40.0)
        elif self.current_model_id:
            print("🚀 FINE-TUNE CURRENT MODEL")
            print("=" * 60)
            print(f"\nCurrent model: {self.current_model_id}")
            epochs = 3
            training_type = "finetuning"
            # Get current accuracy
            models_info = self.registry.get_all_models()
            accuracy_before = 40.0
            for model_id, info in models_info:
                if model_id == self.current_model_id:
                    accuracy_before = info.get('accuracy', 40)
                    break
        else:
            print("🚀 TRAIN NEW MODEL FROM SCRATCH")
            print("=" * 60)
            epochs = 6
            training_type = "full"
            accuracy_before = 40.0
        
        # Load data
        print("\n📖 Loading data...")
        if not self.FILE_NAME.exists():
            print(f"❌ Dataset not found: {self.FILE_NAME}")
            return
        
        df = pd.read_csv(self.FILE_NAME)
        df['Full_Desc'] = (df['Preparation'].fillna('') + " " + df['Execution'].fillna('')).str.strip()
        df = df[df['Full_Desc'] != '']
        
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        print(f"✅ Loaded {len(df)} exercises ({len(train_df)} train, {len(test_df)} test)")
        
        # Get accuracy before training
        accuracy_before = 40.0  # Base embeddings
        if self.current_model_id:
            models_info = self.registry.get_all_models()
            for model_id, info in models_info:
                if model_id == self.current_model_id:
                    accuracy_before = info.get('accuracy', 40)
                    break
        
        # Train
        print(f"\n📚 Training for {epochs} epochs...")
        self.brain.train_real(train_df, epochs=epochs)
        
        # Build database
        print("\n📚 Building database...")
        self.brain.build_database(train_df)
        
        # Calculate accuracy on test set
        print("\n🧪 Calculating accuracy...")
        correct = 0
        for _, row in test_df.head(20).iterrows():
            query = row['Full_Desc']
            correct_answer = row['Exercise Name']
            results = self.brain.search(query, top_k=3)
            
            if results and results[0]['exercise'].lower() == correct_answer.lower():
                correct += 1
        
        accuracy_after = (correct / min(20, len(test_df))) * 100
        
        # Store as draft (NOT saved to registry)
        self.draft_model_data = {
            "model_state_dict": self.brain.model.state_dict(),
            "database_tensor": self.brain.database,
            "database_names": self.brain.database_names,
            "database_descriptions": self.brain.database_descriptions,
            "accuracy_after": accuracy_after,
            "epochs": epochs,
            "training_samples": len(train_df),
            "training_type": training_type,
            "accuracy_before": accuracy_before
        }
        self.draft_ready = True
        
        print("\n✅ Training completed!")
        print(f"   Draft Model: Unsaved")
        print(f"   Accuracy: {accuracy_before}% → {accuracy_after:.1f}% (+{accuracy_after - accuracy_before:.1f}%)")
        print("\n💾 Use option 4 to SAVE this model (required to persist).")
    
    def save_current_model_manual(self):
        """Save the draft model: create new if empty, overwrite if loaded"""
        if not self.draft_ready or not self.draft_model_data:
            print("\n⚠️  No model to save. Train a model first using option 2.")
            return
        
        print("\n💾 Saving model...")
        data = self.draft_model_data
        
        if self.current_model_id:
            # Overwrite existing model
            print(f"   Overwriting {self.current_model_id}...")
            model_id = self.registry.overwrite_model(
                self.current_model_id,
                self.brain,
                data["model_state_dict"],
                data["database_tensor"],
                data["database_names"],
                data["database_descriptions"],
                data["accuracy_after"],
                data["epochs"],
                data["training_samples"],
                data["training_type"]
            )
        else:
            # Create new model
            print(f"   Creating new model...")
            model_id, _ = self.registry.create_new_model(
                self.brain,
                data["model_state_dict"],
                data["database_tensor"],
                data["database_names"],
                data["database_descriptions"],
                data["accuracy_after"],
                data["epochs"],
                data["training_samples"],
                data["training_type"]
            )
        
        # Update session
        self.session.set_current_model(model_id)
        self.session.record_training(
            model_id,
            data["accuracy_before"],
            data["accuracy_after"],
            data["epochs"],
            data["training_type"]
        )
        
        self.current_model_id = model_id
        self.draft_ready = False
        self.draft_model_data = None
        
        print(f"\n✅ Model saved successfully as {model_id}!")
        print(f"   Final Accuracy: {data['accuracy_after']:.1f}%")
        print(f"   Improvement: +{data['accuracy_after'] - data['accuracy_before']:.1f}%")
    
    def load_model_interactive(self):
        """Interactive model selection and loading"""
        models = self.registry.get_all_models()
        
        if not models:
            print("\n📭 No trained models found")
            print("   Train a model first using option 2")
            return
        
        print("\n📚 AVAILABLE MODELS:")
        print("=" * 70)
        
        for i, (model_id, info) in enumerate(models, 1):
            accuracy = info.get('accuracy', 0)
            created = info.get('created', '')[:10]
            epochs = info.get('epochs_trained', 0)
            
            marker = "✅ CURRENT" if model_id == self.current_model_id else ""
            print(f"{i}. {model_id:<18} | Validation score: {accuracy:5.1f}% | {created} | {epochs} epochs {marker}")
        
        print("=" * 70)
        choice = input(f"\n👉 Select model (1-{len(models)}): ").strip()
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                if self.draft_ready:
                    print("\n⚠️  Unsaved draft discarded because you loaded another model.")
                    self.draft_ready = False
                    self.draft_model_data = None
                
                model_id, _ = models[idx]
                
                print(f"\n🔄 Loading {model_id}...")
                self.brain = GymBrain()
                self.brain, _ = self.registry.load_model(model_id, self.brain, self.brain.device)
                
                self.current_model_id = model_id
                self.session.set_current_model(model_id)
                
                print(f"✅ Model loaded successfully!")
                input("\nPress Enter to return to menu...")
            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Invalid input")
    
    def view_training_history(self):
        """Show training history and session summary"""
        print("\n📊 TRAINING HISTORY")
        print("=" * 70)
        
        self.session.print_session_summary()
        
        trainings = self.session.get_training_history()
        if trainings:
            print("\n📚 RECENT TRAININGS:")
            print("-" * 70)
            for i, training in enumerate(trainings[-5:], 1):
                model = training.get('model_created', 'unknown')
                before = training.get('accuracy_before', 0)
                after = training.get('accuracy_after', 0)
                improvement = training.get('improvement', 0)
                date = training.get('timestamp', '')[:10]
                
                print(f"\n{i}. {date} - {model}")
                print(f"\nVALIDATION TEST ACCURACY:")
                print(f"   {before}% → {after}% ({improvement:+.1f}%)")
        
        tests = self.session.get_test_history()
        if tests:
            print(f"\nMANUAL TESTS PERFORMED: {len(tests)}")
            if tests:
                avg_acc = sum(t['accuracy'] for t in tests) / len(tests)
                print(f"   Average Accuracy: {avg_acc:.1f}%")
        
        print("\n" + "=" * 70)
    
    def try_example_search(self):
        """Show random example exercises and perform search"""
        if self.brain is None:
            self.brain = GymBrain()
            if not self.brain.load_database(force=True):
                print("⚠️  No database available. Please train a model first.")
                return
        
        if self.brain.database is None:
            print("❌ No database loaded. Train a model first.")
            return
        
        # Load examples from main dataset (random sample)
        if not self.FILE_NAME.exists():
            print(f"\n❌ Dataset not found: {self.FILE_NAME}")
            return
        
        import random
        df = pd.read_csv(self.FILE_NAME)
        df['Full_Desc'] = (df['Preparation'].fillna('') + " " + df['Execution'].fillna('')).str.strip()
        df = df[df['Full_Desc'] != '']
        
        # Get 10 random examples
        if len(df) < 10:
            print(f"❌ Need at least 10 exercises in dataset, found {len(df)}")
            return
        
        examples_df = df.sample(n=10, random_state=None).reset_index(drop=True)
        
        print("\n" + "=" * 70)
        print("📋 RANDOM EXAMPLE EXERCISES (Select one to search):")
        print("=" * 70)
        
        for idx, row in examples_df.iterrows():
            desc = row['Full_Desc'][:60] + "..." if len(row['Full_Desc']) > 60 else row['Full_Desc']
            print(f"\n{idx + 1}. {desc}")
        
        print("\n" + "=" * 70)
        choice = input("\n👉 Select example (1-10): ").strip()
        
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(examples_df):
                query = examples_df.iloc[choice_idx]['Full_Desc']
                
                print("\n" + "=" * 60)
                print("🔍 SEARCH RESULTS FOR EXAMPLE:")
                print("=" * 60)
                print(f"\n📝 Query:\n{query}\n")
                
                results = self.brain.search(query, top_k=5)
                
                if results:
                    print("🏆 TOP MATCHES:")
                    print("-" * 60)
                    for i, result in enumerate(results, 1):
                        similarity = result['similarity'] * 100
                        print(f"{i}. {result['exercise']:<40} {similarity:.1f}%")
                    print("-" * 60)
                    
                    # Record test in session
                    accuracy = results[0]['similarity'] * 100 if results else 0
                    current_model = self.current_model_id if self.current_model_id else "base"
                    self.session.record_test(query, accuracy, current_model)
                else:
                    print("❌ No results found")
            else:
                print("❌ Invalid selection. Please enter 1-10")
        except ValueError:
            print("❌ Invalid input. Please enter a number between 1-10")
    
    def view_evolution(self):
        """Show model evolution menu"""
        show_main_evolution_menu()
    
    def delete_models_interactive(self):
        """Delete saved models interactively"""
        print("\n" + "=" * 70)
        print("🗑️  DELETE SAVED MODELS")
        print("=" * 70)
        
        models = self.registry.get_all_models()
        
        if not models:
            print("\n📭 No saved models found")
            return
        
        print("\n📚 AVAILABLE MODELS:")
        print("-" * 70)
        
        for i, (model_id, info) in enumerate(models, 1):
            created_date = info.get('created', '')[:10]
            accuracy = info.get('accuracy', 0)
            epochs = info.get('epochs_trained', 0)
            current_marker = " (CURRENT)" if model_id == self.current_model_id else ""
            
            print(f"{i}. {model_id}{current_marker}")
            print(f"   📅 {created_date} | 🎯 {accuracy}% | 📊 {epochs} epochs")
        
        print("-" * 70)
        print("\n⚠️  WARNING: This action cannot be undone!")
        print("\nOptions:")
        print("  • Enter model number to delete")
        print("  • Type 'all' to delete ALL models")
        print("  • Press Enter to cancel")
        
        choice = input("\n👉 Your choice: ").strip().lower()
        
        if not choice:
            print("❌ Cancelled")
            return
        
        if choice == 'all':
            confirm = input("\n⚠️  Delete ALL models? Type 'yes' to confirm: ").strip().lower()
            if confirm != 'yes':
                print("❌ Cancelled")
                return
            
            deleted_count = 0
            for model_id, _ in models:
                if self.registry.delete_model(model_id):
                    deleted_count += 1
                    print(f"   ✅ Deleted: {model_id}")
            
            print(f"\n✅ Successfully deleted {deleted_count} model(s)")
            
            # Clear current model if it was deleted
            if self.current_model_id:
                self.current_model_id = None
                self.brain = None
                self.session.set_current_model(None)
            return
        
        # Try to parse as number
        try:
            index = int(choice) - 1
            if 0 <= index < len(models):
                model_id, info = models[index]
                
                confirm = input(f"\n⚠️  Delete {model_id}? Type 'yes' to confirm: ").strip().lower()
                if confirm != 'yes':
                    print("❌ Cancelled")
                    return
                
                if self.registry.delete_model(model_id):
                    print(f"\n✅ Successfully deleted: {model_id}")
                    
                    # Clear current model if it was the one deleted
                    if self.current_model_id == model_id:
                        self.current_model_id = None
                        self.brain = None
                        self.session.set_current_model(None)
                        print("   ℹ️  Current model cleared")
                else:
                    print(f"\n❌ Failed to delete: {model_id}")
            else:
                print("❌ Invalid model number")
        except ValueError:
            print("❌ Invalid input")
    
    def run(self):
        """Main loop"""
        while True:
            choice = self.display_main_menu()
            
            if choice == "1":
                self.train_new_model()
                input("\nPress Enter to return to menu...")
            elif choice == "2":
                self.test_search()
                input("\nPress Enter to return to menu...")
            elif choice == "3":
                self.try_example_search()
                input("\nPress Enter to return to menu...")
            elif choice == "4":
                self.load_model_interactive()
            elif choice == "5":
                self.save_current_model_manual()
                input("\nPress Enter to return to menu...")
            elif choice == "6":
                self.delete_models_interactive()
                input("\nPress Enter to return to menu...")
            elif choice == "7":
                self.view_training_history()
                input("\nPress Enter to return to menu...")
            elif choice == "8":
                self.view_evolution()
                input("\nPress Enter to return to menu...")
            elif choice == "9":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Try again.")


def main():
    """Entry point"""
    try:
        if len(sys.argv) > 1 and sys.argv[1] in ("--load-model", "load-model"):
            menu = AdaptiveTerminalMenu()
            menu.load_model_interactive()
            return

        menu = AdaptiveTerminalMenu()
        menu.run()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
