"""
Interfaces Abstraites - Dependency Inversion Principle (DIP)
Les classes de haut niveau ne dépendent pas des classes de bas niveau.
Les deux dépendent d'abstractions.
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List
from pathlib import Path


class IDataProcessor(ABC):
    """Interface pour le traitement des données."""
    
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Charge les données."""
        pass
    
    @abstractmethod
    def clean(self) -> None:
        """Nettoie les données."""
        pass
    
    @abstractmethod
    def encode(self) -> None:
        """Encode les variables catégoriques."""
        pass
    
    @abstractmethod
    def prepare(self) -> Tuple:
        """Prépare les données pour l'entraînement."""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Retourne les noms des features."""
        pass


class IModel(ABC):
    """Interface pour le modèle ML."""
    
    @abstractmethod
    def train(self, X_train, y_train) -> None:
        """Entraîne le modèle."""
        pass
    
    @abstractmethod
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """Évalue le modèle."""
        pass
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Prédit les classes."""
        pass
    
    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Retourne les probabilités."""
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Sauvegarde le modèle."""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Charge le modèle."""
        pass


class IPredictor(ABC):
    """Interface pour les prédictions."""
    
    @abstractmethod
    def predict_single(self, data: Dict) -> Tuple[int, float]:
        """Prédit pour un client."""
        pass
    
    @abstractmethod
    def predict_batch(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prédit pour plusieurs clients."""
        pass
    
    @abstractmethod
    def format_result(self, prediction: int, probability: float, client_data: Dict) -> Dict:
        """Formate le résultat."""
        pass


class IScaler(ABC):
    """Interface pour le scaling des données."""
    
    @abstractmethod
    def fit_transform(self, X) -> np.ndarray:
        """Fit et transform les données."""
        pass
    
    @abstractmethod
    def transform(self, X) -> np.ndarray:
        """Transform les données."""
        pass


class IRepository(ABC):
    """Interface pour la persistance (MLOps)."""
    
    @abstractmethod
    def save_model(self, model: Any, filepath: str) -> None:
        """Sauvegarde un modèle."""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> Any:
        """Charge un modèle."""
        pass
    
    @abstractmethod
    def save_metadata(self, metadata: Dict, filepath: str) -> None:
        """Sauvegarde les métadonnées."""
        pass
    
    @abstractmethod
    def load_metadata(self, filepath: str) -> Dict:
        """Charge les métadonnées."""
        pass


class ILogger(ABC):
    """Interface pour le logging."""
    
    @abstractmethod
    def info(self, message: str) -> None:
        """Log un message info."""
        pass
    
    @abstractmethod
    def error(self, message: str) -> None:
        """Log une erreur."""
        pass
    
    @abstractmethod
    def warning(self, message: str) -> None:
        """Log un avertissement."""
        pass
