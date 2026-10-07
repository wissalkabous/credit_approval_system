"""
Conteneur IoC (Inversion of Control) - Respecte DIP
Gère l'injection de dépendances pour le projet.
"""
from typing import Dict, Any
from src.interfaces import IModel, IDataProcessor, IPredictor, ILogger, IRepository
from src.model import CreditModel
from src.data_loader import DataProcessor
from src.predictor import CreditPredictor
from src.logger import ConsoleLogger, HybridLogger
from src.repository import ModelRepository


class DIContainer:
    """
    Conteneur d'Injection de Dépendances.
    Respecte le Dependency Inversion Principle (DIP).

    Design Pattern: Singleton — une seule instance du conteneur dans toute l'application.
    """

    _instance = None  # Singleton: référence à l'unique instance

    def __new__(cls):
        """Singleton Pattern: retourne toujours la même instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialise le conteneur avec les dépendances par défaut."""
        if hasattr(self, '_initialized'):  # évite la réinitialisation
            return
        self._services: Dict[str, Any] = {}
        self._register_defaults()
        self._initialized = True
    
    def _register_defaults(self) -> None:
        """Enregistre les services par défaut."""
        # Logger
        self.register_service('logger', HybridLogger(verbose=True))
        
        # Repository
        self.register_service('repository', ModelRepository())
    
    def register_service(self, name: str, service: Any) -> None:
        """
        Enregistre un service dans le conteneur.
        
        Args:
            name: Nom du service
            service: Instance du service
        """
        self._services[name] = service
        print(f"[OK] Service registered: {name}")
    
    def get_service(self, name: str) -> Any:
        """
        Récupère un service du conteneur.
        
        Args:
            name: Nom du service
            
        Returns:
            Instance du service
        """
        if name not in self._services:
            raise ValueError(f"Service not found: {name}")
        
        return self._services[name]
    
    def create_data_processor(self, data_path: str) -> IDataProcessor:
        """
        Factory pour créer un DataProcessor avec injection de dépendances.
        
        Args:
            data_path: Chemin vers les données
            
        Returns:
            Instance de DataProcessor
        """
        logger = self.get_service('logger')
        return DataProcessor(data_path, logger=logger)
    
    def create_model(self, model_type: str = "random_forest") -> IModel:
        """
        Factory pour créer un CreditModel avec injection de dépendances.
        
        Args:
            model_type: Type de modèle ("logistic" ou "random_forest")
            
        Returns:
            Instance de CreditModel
        """
        logger = self.get_service('logger')
        repository = self.get_service('repository')
        return CreditModel(model_type=model_type, logger=logger, repository=repository)
    
    def create_predictor(self, 
                        model: IModel,
                        data_processor: IDataProcessor) -> IPredictor:
        """
        Factory pour créer un CreditPredictor avec injection de dépendances.
        
        Args:
            model: Instance de IModel
            data_processor: Instance de IDataProcessor
            
        Returns:
            Instance de CreditPredictor
        """
        logger = self.get_service('logger')
        return CreditPredictor(model=model, data_processor=data_processor, logger=logger)


# Instance globale du conteneur
_container = None


def get_container() -> DIContainer:
    """
    Fonction globale pour obtenir l'instance du conteneur.
    
    Returns:
        Instance du DIContainer
    """
    global _container
    if _container is None:
        _container = DIContainer()
    return _container


def initialize_container() -> DIContainer:
    """
    Initialise (ou réinitialise) le conteneur Singleton.

    Returns:
        Instance unique du DIContainer
    """
    global _container
    # Réinitialiser le Singleton pour permettre un fresh start
    DIContainer._instance = None
    _container = DIContainer()
    return _container
