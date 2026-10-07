"""
Script d'entraînement du modèle - Respecte DIP
Utilise l'Inversion of Control Container pour l'injection de dépendances
Exécutez: python train.py
"""
import os
import sys
import pandas as pd
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent))

from src.container import initialize_container
from src.interfaces import IDataProcessor, IModel, IRepository, ILogger

# Configuration
DATA_PATH = "data/raw/credit_data.csv"
MODEL_PATH = "credit_model.pkl"
MODEL_INFO_PATH = "credit_model_info.json"


def main():
    """Fonction principale avec injection de dépendances."""
    
    # Initialiser le conteneur IoC
    container = initialize_container()
    logger = container.get_service('logger')
    repository = container.get_service('repository')
    
    print("=" * 60)
    logger.info("DÉMARRAGE DE L'ENTRAÎNEMENT DU MODÈLE")
    print("=" * 60)
    
    # Vérifier que les données existent
    if not os.path.exists(DATA_PATH):
        logger.error(f"Fichier non trouvé: {DATA_PATH}")
        logger.info("Veuillez télécharger un dataset depuis Kaggle et le placer dans data/raw/credit_data.csv")
        return
    
    # Créer les dossiers nécessaires
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    
    try:
        # ========== ÉTAPE 1: Créer et préparer les données ==========
        print("\n" + "="*60)
        logger.info("ÉTAPE 1: Chargement et Préparation des Données")
        print("="*60)
        
        # Utiliser le conteneur pour créer le DataProcessor (injection de dépendances)
        data_processor: IDataProcessor = container.create_data_processor(DATA_PATH)
        
        # Utiliser l'interface IDataProcessor
        data_processor.load()
        data_processor.clean()
        data_processor.encode()
        
        # Afficher les infos
        logger.info(f"Exploration des données...")
        info = data_processor.explore_data()
        logger.info(f"  Shape: {info['shape']}")
        logger.info(f"  Colonnes: {list(info['dtypes'].keys())}")
        
        # Préparer train/test
        X_train, X_test, y_train, y_test = data_processor.prepare()
        
        # Sauvegarder les données
        logger.info("Sauvegarde des données traitées...")
        X_train_df = pd.DataFrame(X_train, columns=data_processor.get_feature_names())
        X_train_df['Approval'] = y_train.values
        X_train_df.to_csv("data/processed/train.csv", index=False)
        
        X_test_df = pd.DataFrame(X_test, columns=data_processor.get_feature_names())
        X_test_df['Approval'] = y_test.values
        X_test_df.to_csv("data/processed/test.csv", index=False)
        logger.info("Données sauvegardées")
        
        # ========== ÉTAPE 2: Créer et entraîner le modèle ==========
        print("\n" + "="*60)
        logger.info("ÉTAPE 2: Entraînement du Modèle")
        print("="*60)
        
        # Utiliser le conteneur pour créer le CreditModel (injection de dépendances)
        model: IModel = container.create_model(model_type="random_forest")
        model.feature_names = data_processor.get_feature_names()
        
        # Utiliser l'interface IModel
        model.train(X_train, y_train)
        
        # ========== ÉTAPE 3: Évaluer le modèle ==========
        print("\n" + "="*60)
        logger.info("ÉTAPE 3: Évaluation du Modèle")
        print("="*60)
        
        metrics = model.evaluate(X_test, y_test)
        
        # ========== ÉTAPE 4: Sauvegarder (MLOps) ==========
        print("\n" + "="*60)
        logger.info("ÉTAPE 4: Sauvegarde du Modèle (MLOps)")
        print("="*60)
        
        # Utiliser le repository (injection de dépendances) pour sauvegarder
        model.save(MODEL_PATH)
        
        # Sauvegarder les infos du modèle
        model_info = model.get_model_info()
        model_info['feature_names'] = data_processor.get_feature_names()
        repository.save_metadata(model_info, MODEL_INFO_PATH)
        
        # ========== RÉSUMÉ ==========
        print("\n" + "="*60)
        logger.info("ENTRAÎNEMENT RÉUSSI!")
        print("="*60)
        
        logger.info(f"Métriques finales:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info(f"Fichiers créés:")
        logger.info(f"  ✓ Modèle: models/{MODEL_PATH}")
        logger.info(f"  ✓ Infos: models/{MODEL_INFO_PATH}")
        logger.info(f"  ✓ Train: data/processed/train.csv")
        logger.info(f"  ✓ Test: data/processed/test.csv")
        
        logger.info(f"Prochaine étape: Lancer l'API")
        logger.info(f"  python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")
    
    except Exception as e:
        logger.error(f"Erreur durant l'entraînement: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

