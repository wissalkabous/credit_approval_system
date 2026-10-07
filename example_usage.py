"""
Exemple d'utilisation complet avec DIP et IoC Container
Ce script démontre tous les concepts: OOP, DIP, ML, Décorateurs, MLOps
"""
import sys
from pathlib import Path

# Ajouter le répertoire parent
sys.path.insert(0, str(Path(__file__).parent))

from src import (
    initialize_container,
    IDataProcessor,
    IModel,
    IPredictor,
    pipe,
    map_transform,
    filter_by,
    ConsoleLogger,
    HybridLogger
)


def example_complete_workflow():
    """Exemple complet du workflow avec DIP et injection de dépendances."""
    
    print("\n" + "=" * 70)
    print("EXEMPLE COMPLET: Credit Approval System avec DIP")
    print("=" * 70)
    
    # ========== ÉTAPE 1: Initialiser le conteneur IoC ==========
    print("\n" + "-" * 70)
    print("ÉTAPE 1: Initialiser le Conteneur IoC (Dependency Injection)")
    print("-" * 70)
    
    container = initialize_container()
    logger = container.get_service('logger')
    
    logger.info("Conteneur IoC initialisé avec injection de dépendances")
    logger.info("Services disponibles: logger, repository, et factories")
    
    # ========== ÉTAPE 2: Créer les dépendances via le conteneur ==========
    print("\n" + "-" * 70)
    print("ÉTAPE 2: Créer les Dépendances via le Conteneur")
    print("-" * 70)
    
    # ✅ DIP: Les dépendances sont créées via le conteneur, pas directement
    data_processor: IDataProcessor = container.create_data_processor("data/raw/credit_data.csv")
    model: IModel = container.create_model(model_type="random_forest")
    
    logger.info("DataProcessor créé (implémente IDataProcessor)")
    logger.info("CreditModel créé (implémente IModel)")
    
    # ========== ÉTAPE 3: Utiliser les interfaces abstraites ==========
    print("\n" + "-" * 70)
    print("ÉTAPE 3: Utiliser les Interfaces Abstraites (OOP + DIP)")
    print("-" * 70)
    
    # ✅ OOP: On utilise l'interface IDataProcessor, pas la classe directe
    logger.info("Chargement des données (utilise IDataProcessor.load())")
    data_processor.load()
    
    logger.info("Nettoyage des données (utilise IDataProcessor.clean())")
    data_processor.clean()
    
    logger.info("Encoding (utilise IDataProcessor.encode())")
    data_processor.encode()
    
    # ========== ÉTAPE 4: Préparation des données ==========
    print("\n" + "-" * 70)
    print("ÉTAPE 4: Préparation des Données pour ML")
    print("-" * 70)
    
    # ✅ ML: Préparer les données avec train/test split et scaling
    X_train, X_test, y_train, y_test = data_processor.prepare()
    
    logger.info(f"Train set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    
    # ========== ÉTAPE 5: Entraîner le modèle (avec décorateurs) ==========
    print("\n" + "-" * 70)
    print("ÉTAPE 5: Entraînement du Modèle (Décorateurs + ML)")
    print("-" * 70)
    
    # ✅ Décorateurs: @log_execution, @timing appliqués automatiquement
    model.feature_names = data_processor.get_feature_names()
    model.train(X_train, y_train)
    
    logger.info("Modèle entraîné avec succès")
    
    # ========== ÉTAPE 6: Évaluation (Classification) ==========
    print("\n" + "-" * 70)
    print("ÉTAPE 6: Évaluation du Modèle (Classification)")
    print("-" * 70)
    
    # ✅ Classification: Évaluer avec accuracy, precision, recall, f1, roc_auc
    metrics = model.evaluate(X_test, y_test)
    
    logger.info("Métriques de classification:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    # ========== ÉTAPE 7: Sauvegarde (MLOps) ==========
    print("\n" + "-" * 70)
    print("ÉTAPE 7: Sauvegarde du Modèle (MLOps - MLflow Pattern)")
    print("-" * 70)
    
    # ✅ MLOps: Repository pattern pour persistance + versioning
    model.save("credit_model.pkl")
    
    logger.info("Modèle et métadonnées sauvegardés (MLOps)")
    logger.info("Version: 1")
    
    # ========== ÉTAPE 8: Créer le prédicteur ==========
    print("\n" + "-" * 70)
    print("ÉTAPE 8: Créer le Prédicteur (DIP)")
    print("-" * 70)
    
    # ✅ DIP: Injecter les dépendances dans le prédicteur
    predictor: IPredictor = container.create_predictor(model, data_processor)
    
    logger.info("CreditPredictor créé avec injection des dépendances")
    
    # ========== ÉTAPE 9: Faire des prédictions ==========
    print("\n" + "-" * 70)
    print("ÉTAPE 9: Faire des Prédictions")
    print("-" * 70)
    
    # Données de test
    test_client = {
        'Age': 35,
        'MonthlyIncome': 5000,       # MAD
        'EmploymentStatus': 0,
        'EmploymentDuration': 5,
        'TotalDebt': 20000,          # MAD
        'LoanAmount': 80000,         # MAD
        'LoanDuration': 36,
        'CreditScore': 700,
        'RepaymentHistory': 7,
        'NumLatePayments': 1,
        'NumOpenAccounts': 3,
        'AccountAgeYears': 5,
    }
    
    logger.info(f"Prédiction pour client: {test_client}")
    prediction, probability = predictor.predict_single(test_client)
    
    result = predictor.format_result(prediction, probability, test_client)
    logger.info(f"Résultat: {result['statut']}")
    logger.info(f"Confiance: {result['confiance_pct']}")
    logger.info(f"Risque: {result['risque']}")
    
    # ========== ÉTAPE 10: Programmation Fonctionnelle ==========
    print("\n" + "-" * 70)
    print("ÉTAPE 10: Programmation Fonctionnelle Avancée")
    print("-" * 70)
    
    # ✅ Fonctionnel: Fonctions d'ordre supérieur, composition, etc.
    
    # Exemple 1: Map transform
    ages = [25, 30, 35, 40, 45]
    
    add_five = map_transform(lambda x: x + 5)
    ages_plus_five = add_five(ages)
    
    logger.info(f"Map transform: {ages} -> {ages_plus_five}")
    
    # Exemple 2: Filter
    is_adult = filter_by(lambda x: x >= 30)
    adults = is_adult(ages)
    
    logger.info(f"Filter (age >= 30): {ages} -> {adults}")
    
    # Exemple 3: Pipe (composition)
    process = pipe(
        add_five,
        is_adult
    )
    
    result = process(ages)
    logger.info(f"Pipe (add_five + filter): {ages} -> {result}")
    
    # ========== RÉSUMÉ ==========
    print("\n" + "=" * 70)
    logger.info("RÉSUMÉ DES CONCEPTS APPLIQUÉS")
    print("=" * 70)
    
    concepts = {
        "OOP": "✅ Classes abstraites (IDataProcessor, IModel, IPredictor)",
        "DIP": "✅ Dependency Inversion Principle avec interfaces",
        "ML": "✅ Scikit-learn, train/test split, scaling",
        "Classification": "✅ Binary classification, metrics (accuracy, precision, f1, roc_auc)",
        "FastAPI": "✅ API REST (voir api/main.py)",
        "Programmation Fonctionnelle": "✅ map, filter, pipe, composition",
        "Décorateurs": "✅ @log_execution, @timing, @validate_input, @with_retry",
        "MLOps": "✅ Repository pattern, versioning, métadonnées",
        "IoC Container": "✅ DIContainer avec injection de dépendances",
        "Logging": "✅ ConsoleLogger, FileLogger, HybridLogger"
    }
    
    logger.info("\nConcepts implémentés:")
    for concept, status in concepts.items():
        logger.info(f"  {status}: {concept}")
    
    print("\n" + "=" * 70)
    logger.info("EXEMPLE COMPLÉTÉ AVEC SUCCÈS!")
    print("=" * 70)


def example_testing_with_dip():
    """Exemple de testing avec DIP (mocking)."""
    
    print("\n\n" + "=" * 70)
    print("EXEMPLE: Testing avec DIP (Mock Dépendances)")
    print("=" * 70)
    
    from src import ILogger
    
    # Créer un mock logger pour les tests
    class MockLogger(ILogger):
        def __init__(self):
            self.messages = []
        
        def info(self, msg: str) -> None:
            self.messages.append(("INFO", msg))
        
        def error(self, msg: str) -> None:
            self.messages.append(("ERROR", msg))
        
        def warning(self, msg: str) -> None:
            self.messages.append(("WARNING", msg))
    
    print("\n" + "-" * 70)
    print("Création d'un MockLogger pour les tests")
    print("-" * 70)
    
    mock_logger = MockLogger()
    logger = ConsoleLogger()
    
    logger.info("Mock logger créé")
    logger.info("Dépendance mockée au lieu d'une vraie")
    logger.info("Facile de contrôler et vérifier les logs")
    
    # Utiliser le mock
    mock_logger.info("Test message")
    mock_logger.error("Test error")
    
    logger.info(f"Messages collectés: {len(mock_logger.messages)}")
    for level, msg in mock_logger.messages:
        logger.info(f"  [{level}] {msg}")


if __name__ == "__main__":
    try:
        example_complete_workflow()
        example_testing_with_dip()
    except Exception as e:
        print(f"\n❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

