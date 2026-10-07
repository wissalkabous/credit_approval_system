"""
Tests Unitaires - Démonstration du DIP et Injection de Dépendances
Ce fichier montre comment tester avec des mocks grace au DIP.

Exécutez: python -m pytest tests/ -v
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ajouter le répertoire parent
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    ILogger,
    IModel,
    IDataProcessor,
    IRepository,
    ConsoleLogger,
    ModelRepository,
    DataProcessor,
    CreditModel,
    CreditPredictor,
    DIContainer
)


# ============================================================================
# MOCKS POUR TESTING
# ============================================================================

class MockLogger(ILogger):
    """Mock Logger pour les tests."""
    
    def __init__(self):
        self.messages = []
    
    def info(self, message: str) -> None:
        self.messages.append(("INFO", message))
    
    def error(self, message: str) -> None:
        self.messages.append(("ERROR", message))
    
    def warning(self, message: str) -> None:
        self.messages.append(("WARNING", message))
    
    def has_message(self, level: str, text: str) -> bool:
        """Vérifie si un message existe."""
        return any(msg_level == level and text in msg_text for msg_level, msg_text in self.messages)


class MockRepository(IRepository):
    """Mock Repository pour les tests."""
    
    def __init__(self):
        self.saved_models = {}
        self.saved_metadata = {}
    
    def save_model(self, model, filepath: str) -> None:
        self.saved_models[filepath] = model
    
    def load_model(self, filepath: str):
        return self.saved_models.get(filepath)
    
    def save_metadata(self, metadata, filepath: str) -> None:
        self.saved_metadata[filepath] = metadata
    
    def load_metadata(self, filepath: str):
        return self.saved_metadata.get(filepath)


class MockDataProcessor(IDataProcessor):
    """Mock DataProcessor pour les tests."""
    
    def __init__(self):
        self.loaded = False
        self.cleaned = False
        self.encoded = False
        self.prepared = False
        self.feature_names = ['age', 'income', 'debt', 'monthly_payment', 'repayment_history', 'credit_score']
    
    def load(self):
        self.loaded = True
        return pd.DataFrame({
            'age': [25, 30, 35],
            'income': [30000, 40000, 50000],
            'debt': [5000, 10000, 15000],
            'monthly_payment': [500, 800, 1200],
            'repayment_history': [2, 3, 4],
            'credit_score': [650, 700, 750],
            'Approval': [0, 1, 1]
        })
    
    def clean(self) -> None:
        self.cleaned = True
    
    def encode(self) -> None:
        self.encoded = True
    
    def prepare(self):
        self.prepared = True
        X_train = np.random.randn(10, 6)
        X_test = np.random.randn(3, 6)
        y_train = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
        y_test = np.array([1, 0, 1])
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self):
        return self.feature_names


# ============================================================================
# TESTS
# ============================================================================

class TestDIP:
    """Tests pour démontrer le Dependency Inversion Principle."""
    
    def test_logger_dependency_injection(self):
        """Test: Logger peut être injecté dans CreditModel."""
        print("\n" + "=" * 70)
        print("TEST 1: Injection de dépendance Logger")
        print("=" * 70)
        
        # Créer un mock logger
        mock_logger = MockLogger()
        
        # Créer le modèle avec le mock logger injecté
        model = CreditModel(logger=mock_logger)
        
        # Vérifier que le mock logger a été injecté
        assert model.logger is mock_logger
        print("✅ Mock logger injecté avec succès dans CreditModel")
    
    def test_repository_dependency_injection(self):
        """Test: Repository peut être injecté dans CreditModel."""
        print("\n" + "=" * 70)
        print("TEST 2: Injection de dépendance Repository")
        print("=" * 70)
        
        # Créer un mock repository
        mock_repo = MockRepository()
        
        # Créer le modèle avec le mock repository injecté
        model = CreditModel(repository=mock_repo)
        
        # Vérifier que le mock repository a été injecté
        assert model.repository is mock_repo
        print("✅ Mock repository injecté avec succès dans CreditModel")
    
    def test_multiple_implementations(self):
        """Test: Différentes implémentations de ILogger peuvent être injectées."""
        print("\n" + "=" * 70)
        print("TEST 3: Plusieurs implémentations d'ILogger")
        print("=" * 70)
        
        # Logger original
        console_logger = ConsoleLogger()
        
        # Mock logger
        mock_logger = MockLogger()
        
        # Créer deux modèles avec des loggers différents
        model1 = CreditModel(logger=console_logger)
        model2 = CreditModel(logger=mock_logger)
        
        # Vérifier que chaque modèle a son propre logger
        assert isinstance(model1.logger, ConsoleLogger)
        assert isinstance(model2.logger, MockLogger)
        
        print("✅ Différentes implémentations injectées avec succès")
        print("  Model1 -> ConsoleLogger")
        print("  Model2 -> MockLogger")
    
    def test_predictor_with_mock_dependencies(self):
        """Test: CreditPredictor avec dépendances mockées."""
        print("\n" + "=" * 70)
        print("TEST 4: Prédicteur avec dépendances mockées")
        print("=" * 70)
        
        # Créer les mocks
        mock_logger = MockLogger()
        mock_data_processor = MockDataProcessor()
        mock_model = CreditModel(logger=mock_logger)
        
        # Créer le prédicteur avec les mocks
        predictor = CreditPredictor(
            model=mock_model,
            data_processor=mock_data_processor,
            logger=mock_logger
        )
        
        # Vérifier que toutes les dépendances ont été injectées
        assert predictor.model is mock_model
        assert predictor.data_processor is mock_data_processor
        assert predictor.logger is mock_logger
        
        print("✅ Toutes les dépendances injectées dans CreditPredictor")
        print("  - Model (IModel)")
        print("  - DataProcessor (IDataProcessor)")
        print("  - Logger (ILogger)")


class TestDataProcessorInterface:
    """Tests pour vérifier que DataProcessor implémente IDataProcessor."""
    
    def test_implements_interface(self):
        """Test: DataProcessor implémente IDataProcessor."""
        print("\n" + "=" * 70)
        print("TEST 5: DataProcessor implémente IDataProcessor")
        print("=" * 70)
        
        # Créer un DataProcessor avec un logger
        logger = MockLogger()
        processor = DataProcessor("data/raw/credit_data.csv", logger=logger)
        
        # Vérifier que c'est une instance de IDataProcessor
        assert isinstance(processor, IDataProcessor)
        
        # Vérifier que toutes les méthodes existent
        assert hasattr(processor, 'load')
        assert hasattr(processor, 'clean')
        assert hasattr(processor, 'encode')
        assert hasattr(processor, 'prepare')
        assert hasattr(processor, 'get_feature_names')
        
        print("✅ DataProcessor implémente correctement IDataProcessor")


class TestLogging:
    """Tests pour vérifier le logging avec injection de dépendances."""
    
    def test_mock_logger_captures_messages(self):
        """Test: MockLogger capture les messages."""
        print("\n" + "=" * 70)
        print("TEST 6: MockLogger capture les messages")
        print("=" * 70)
        
        mock_logger = MockLogger()
        
        # Simuler des logs
        mock_logger.info("Test info")
        mock_logger.error("Test error")
        mock_logger.warning("Test warning")
        
        # Vérifier que les messages ont été capturés
        assert len(mock_logger.messages) == 3
        assert mock_logger.has_message("INFO", "Test info")
        assert mock_logger.has_message("ERROR", "Test error")
        assert mock_logger.has_message("WARNING", "Test warning")
        
        print("✅ MockLogger a capturé tous les messages")
        for level, msg in mock_logger.messages:
            print(f"  [{level}] {msg}")


class TestDIContainer:
    """Tests pour le conteneur IoC."""
    
    def test_container_creates_services(self):
        """Test: DIContainer crée les services."""
        print("\n" + "=" * 70)
        print("TEST 7: DIContainer crée les services")
        print("=" * 70)
        
        container = DIContainer()
        
        # Vérifier que les services sont enregistrés
        logger = container.get_service('logger')
        repository = container.get_service('repository')
        
        assert logger is not None
        assert repository is not None
        
        print("✅ DIContainer a créé les services avec succès")
        print(f"  Logger: {type(logger).__name__}")
        print(f"  Repository: {type(repository).__name__}")
    
    def test_container_factories(self):
        """Test: DIContainer crée les objets via factories."""
        print("\n" + "=" * 70)
        print("TEST 8: DIContainer factories")
        print("=" * 70)
        
        container = DIContainer()
        
        # Créer un DataProcessor via la factory
        data_processor = container.create_data_processor("test.csv")
        
        # Créer un Model via la factory
        model = container.create_model("random_forest")
        
        # Vérifier que les interfaces sont implémentées
        assert isinstance(data_processor, IDataProcessor)
        assert isinstance(model, IModel)
        
        print("✅ DIContainer factories créent les objets correctement")
        print(f"  DataProcessor: {type(data_processor).__name__}")
        print(f"  Model: {type(model).__name__}")


# ============================================================================
# RÉSUMÉ
# ============================================================================

def run_all_tests():
    """Exécute tous les tests."""
    print("\n" + "=" * 70)
    print("EXÉCUTION DE TOUS LES TESTS DIP")
    print("=" * 70)
    
    # Instancier les classes de test
    test_dip = TestDIP()
    test_processor = TestDataProcessorInterface()
    test_logging = TestLogging()
    test_container = TestDIContainer()
    
    # Exécuter les tests
    tests = [
        test_dip.test_logger_dependency_injection,
        test_dip.test_repository_dependency_injection,
        test_dip.test_multiple_implementations,
        test_dip.test_predictor_with_mock_dependencies,
        test_processor.test_implements_interface,
        test_logging.test_mock_logger_captures_messages,
        test_container.test_container_creates_services,
        test_container.test_container_factories,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ ÉCHOUÉ: {str(e)}")
            failed += 1
        except Exception as e:
            print(f"❌ ERREUR: {str(e)}")
            failed += 1
    
    # Résumé
    print("\n" + "=" * 70)
    print("RÉSUMÉ DES TESTS")
    print("=" * 70)
    print(f"✅ Réussis: {passed}/{len(tests)}")
    print(f"❌ Échoués: {failed}/{len(tests)}")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
