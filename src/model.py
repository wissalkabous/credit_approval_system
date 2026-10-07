"""
Modèle Machine Learning - Respecte DIP
Implémente IModel pour l'inversion des dépendances
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.decorators import log_execution, timing
from src.interfaces import IModel, IRepository, ILogger
from src.logger import ConsoleLogger
from src.repository import ModelRepository
from typing import Dict


class CreditModel(IModel):
    """
    Classe pour le modèle de prédiction de crédit.
    Respecte le Dependency Inversion Principle (DIP).
    """
    
    def __init__(self, 
                 model_type: str = "logistic",
                 logger: ILogger = None,   # interface — not HybridLogger
                 repository: IRepository = None): # interface — not ModelRepository
        """
        Initialise le modèle.
        
        Args:
            model_type: "logistic" ou "random_forest"
            logger: Implémentation de ILogger (injection de dépendance)
            repository: Implémentation de IRepository (injection de dépendance)
        """
        self.model_type = model_type
        self.model = None
        self.metrics = {}
        self.feature_names = None
        self.logger = logger or ConsoleLogger()  # DIP
        self.repository = repository or ModelRepository()  # DIP
        
        if model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='lbfgs'
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Type de modèle inconnu: {model_type}")
    
    def train(self, X_train, y_train) -> None:
        """Implémente IModel.train() - Entraîne le modèle."""
        @log_execution
        @timing
        def _train():
            self.logger.info(f"Entraînement du modèle {self.model_type}...")
            self.model.fit(X_train, y_train)
            self.logger.info("Entraînement terminé")
        
        return _train()
    
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """Implémente IModel.evaluate() - Évalue le modèle."""
        @log_execution
        @timing
        def _evaluate():
            self.logger.info("Évaluation du modèle...")
            
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            self.metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, y_pred_proba))
            }
            
            self.logger.info(f"Résultats:")
            for metric, value in self.metrics.items():
                self.logger.info(f"  {metric}: {value:.4f}")
            
            return self.metrics
        
        return _evaluate()
    
    def predict(self, X) -> np.ndarray:
        """Prédit les classes."""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def predict_proba(self, X) -> np.ndarray:
        """Retourne les probabilités."""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)
    
    def save(self, filepath: str) -> None:
        """Implémente IModel.save() - Sauvegarde le modèle via le repository."""
        self.repository.save_model(self.model, filepath)
        
        # Sauvegarder aussi les métadonnées
        metadata = self.get_model_info()
        metadata_path = filepath.replace('.pkl', '_info.json')
        self.repository.save_metadata(metadata, metadata_path)
    
    def load(self, filepath: str) -> None:
        """Implémente IModel.load() - Charge le modèle via le repository."""
        self.model = self.repository.load_model(filepath)
    
    def get_feature_importance(self):
        """Retourne l'importance des features (Random Forest only)."""
        if self.model_type != "random_forest":
            return None
        
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        return self.model.feature_importances_
    
    def get_model_info(self) -> Dict:
        """Retourne les infos du modèle."""
        return {
            "type": self.model_type,
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "is_trained": self.model is not None
        }
