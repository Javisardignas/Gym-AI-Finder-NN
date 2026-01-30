#!/usr/bin/env python
"""
Script to generate exercise database
Trains model and saves gym_database.pkl
"""

import os
import sys

# Add current path to sys.path
sys.path.insert(0, os.path.dirname(__file__))

from nngym_v2 import GymBrain, pd, train_test_split, FILE_NAME

def generate_database():
    """Generates database by training the model"""
    
    if not os.path.exists(FILE_NAME):
        print(f"\n❌ Error: {FILE_NAME} not found")
        return False
    
    try:
        print("\n" + "="*60)
        print("🏋️  GENERATING DATABASE - GYM AI")
        print("="*60)
        
        print("\n📖 Reading data...")
        df = pd.read_csv(FILE_NAME)
        df['Full_Desc'] = (df['Preparation'].fillna('') + " " + df['Execution'].fillna('')).str.strip()
        df = df[df['Full_Desc'] != '']
        
        print(f"✅ Data loaded: {len(df)} exercises")
        
        train_df, _ = train_test_split(df, test_size=0.2, random_state=42)
        
        print("\n🧠 Initializing model...")
        brain = GymBrain()
        
        print("\n🚀 Training model...")
        brain.train_real(train_df, epochs=6)
        
        print("\n📚 Building database...")
        brain.build_database(train_df)
        
        print("\n" + "="*60)
        print("✅ DATABASE GENERATED SUCCESSFULLY")
        print("="*60)
        print("\nFile created: gym_database.pkl")
        print("Now you can run: servidor_api.bat")
        print("\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    generate_database()
