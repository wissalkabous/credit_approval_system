"""
Chargement et Préparation des Données - Respecte DIP
Implémente IDataProcessor pour l'inversion des dépendances
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from src.decorators import log_execution, timing
from src.interfaces import IDataProcessor, ILogger
from src.logger import ConsoleLogger


class DataProcessor(IDataProcessor):
    """
    Classe pour charger et préparer les données.
    Respecte le Dependency Inversion Principle (DIP).
    """
    
    def __init__(self, data_path: str, logger: ILogger = None):
        """
        Initialise le DataProcessor.
        
        Args:
            data_path: Chemin vers le fichier CSV
            logger: Implémentation de ILogger (injection de dépendance)
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.logger = logger or ConsoleLogger()  # DIP: dépend de l'interface
        self.feature_names = []
        
    def load(self) -> pd.DataFrame:
        """Implémente IDataProcessor.load() - Charge le CSV."""
        @log_execution
        @timing
        def _load():
            self.data = pd.read_csv(self.data_path)
            self.logger.info(f"Données chargées: {self.data.shape}")
            return self.data
        
        return _load()
    
    def explore_data(self) -> Dict:
        """Explore les données - stats basiques."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        info = {
            "shape": self.data.shape,
            "missing": self.data.isnull().sum().to_dict(),
            "dtypes": self.data.dtypes.astype(str).to_dict(),
            "stats": self.data.describe().to_dict()
        }
        return info
    
    def clean(self) -> None:
        """Implémente IDataProcessor.clean() - Nettoie les données."""
        @log_execution
        def _clean():
            self.logger.info("Nettoyage des données...")
            
            # Supprimer les doublons
            self.data = self.data.drop_duplicates()
            
            # Remplir les valeurs manquantes
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                self.data[col] = self.data[col].fillna(self.data[col].mean())

            categorical_cols = self.data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col != 'Approval':
                    self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
            
            self.logger.info(f"Nettoyage terminé. Shape: {self.data.shape}")
        
        return _clean()
    
    def encode(self) -> None:
        """Implémente IDataProcessor.encode() - Encode les variables catégoriques."""
        @log_execution
        def _encode():
            self.logger.info("Encoding des variables catégoriques...")
            
            for col in self.data.columns:
                if col == 'Approval':
                    continue
                
                if self.data[col].dtype == 'object':
                    le = LabelEncoder()
                    self.data[col] = le.fit_transform(self.data[col].astype(str))
                    self.label_encoders[col] = le
                    self.logger.info(f"  encoded: {col}")
        
        return _encode()
    
    def prepare(self, target_column: str = 'Approval', test_size: float = 0.2) -> Tuple:
        """Implémente IDataProcessor.prepare() - Prépare les données pour l'entraînement."""
        @log_execution
        @timing
        def _prepare():
            self.logger.info("Préparation des données...")
            
            # Séparer features et target
            X = self.data.drop(columns=[target_column])
            y = self.data[target_column]
            
            # Sauvegarder les noms des features
            self.feature_names = list(X.columns)
            
            # Train/Test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Scaling
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
            self.logger.info(f"Train set: {self.X_train.shape}")
            self.logger.info(f"Test set: {self.X_test.shape}")
            
            return self.X_train, self.X_test, self.y_train, self.y_test
        
        return _prepare()
    
    def get_feature_names(self) -> List[str]:
        """Implémente IDataProcessor.get_feature_names()."""
        if not self.feature_names:
            return [col for col in self.data.columns if col != 'Approval']
        return self.feature_names
    
    def explore_data(self) -> Dict:
        """Explore les données - stats basiques (non requis par l'interface)."""
        if self.data is None:
            raise ValueError("Données non chargées. Appelez load() d'abord.")
        
        info = {
            "shape": self.data.shape,
            "missing": self.data.isnull().sum().to_dict(),
            "dtypes": self.data.dtypes.astype(str).to_dict(),
            "stats": self.data.describe().to_dict()
        }
        return info
