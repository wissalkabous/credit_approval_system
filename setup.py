"""
Script de configuration initiale
Exécutez: python setup.py
"""
import os
from pathlib import Path

def setup():
    """Configure le projet."""
    
    print("\n" + "="*60)
    print("⚙️  CONFIGURATION DU PROJET")
    print("="*60)
    
    # Créer les répertoires
    dirs = [
        "data/raw",
        "data/processed",
        "models",
        "src",
        "api"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Répertoire créé: {dir_path}")
    
    # Vérifier les fichiers importants
    print("\n📁 Vérification des fichiers...")
    files = [
        "requirements.txt",
        "train.py",
        "src/__init__.py",
        "src/model.py",
        "src/data_loader.py",
        "src/predictor.py",
        "src/decorators.py",
        "api/main.py",
        "api/schemas.py",
        "data/raw/credit_data.csv"
    ]
    
    for file_path in files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MANQUANT")
    
    print("\n" + "="*60)
    print("✅ SETUP TERMINÉ!")
    print("="*60)
    print("\n📝 Prochaines étapes:")
    print("1. Installer les dépendances:")
    print("   pip install -r requirements.txt")
    print("\n2. Télécharger un dataset (optionnel):")
    print("   - UCI German Credit Dataset")
    print("   - Lending Club dataset")
    print("   - Placer dans: data/raw/credit_data.csv")
    print("\n3. Entraîner le modèle:")
    print("   python train.py")
    print("\n4. Lancer l'API:")
    print("   python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")
    print("\n5. Ouvrir le navigateur:")
    print("   http://localhost:8000")


if __name__ == "__main__":
    setup()
