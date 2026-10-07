"""
Prédicteur - Respecte DIP
Implémente IPredictor pour l'inversion des dépendances
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from src.interfaces import IPredictor, IModel, IDataProcessor, ILogger
from src.logger import ConsoleLogger
from src.decorators import timing, validate_input


class CreditPredictor(IPredictor):
    """
    Classe pour faire des prédictions avec le modèle.
    Respecte le Dependency Inversion Principle (DIP).
    """
    
    def __init__(self, 
                 model: IModel,
                 data_processor: IDataProcessor,
                 logger: ILogger = None):
        """
        Initialise le prédicteur.
        
        Args:
            model: Implémentation de IModel
            data_processor: Implémentation de IDataProcessor
            logger: Implémentation de ILogger
        """
        self.model = model  # DIP: dépend de l'interface IModel
        self.data_processor = data_processor  # DIP: dépend de l'interface IDataProcessor
        self.logger = logger or ConsoleLogger()  # DIP: dépend de l'interface ILogger
    
    def predict_single(self, data: Dict) -> Tuple[int, float]:
        """Implémente IPredictor.predict_single() - Prédiction pour un client."""
        @timing
        @validate_input
        def _predict():
            # Créer un DataFrame avec une seule ligne
            df = pd.DataFrame([data])
            
            # Assurer l'ordre des colonnes
            feature_names = self.data_processor.get_feature_names()
            df = df[feature_names]
            
            # Scaling
            df_scaled = self.data_processor.scaler.transform(df)
            
            # Prédiction
            prediction = self.model.predict(df_scaled)[0]
            probability = self.model.predict_proba(df_scaled)[0][1]
            
            return int(prediction), float(probability)
        
        return _predict()
    
    def predict_batch(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Implémente IPredictor.predict_batch() - Prédictions pour plusieurs clients."""
        @timing
        def _predict():
            # Assurer l'ordre des colonnes
            feature_names = self.data_processor.get_feature_names()
            data_reordered = data[feature_names]
            
            # Scaling
            data_scaled = self.data_processor.scaler.transform(data_reordered)
            
            # Prédictions
            predictions = self.model.predict(data_scaled)
            probabilities = self.model.predict_proba(data_scaled)[:, 1]
            
            return predictions, probabilities
        
        return _predict()
    
    def format_result(self, prediction: int, probability: float, client_data: Dict) -> Dict:
        """Implémente IPredictor.format_result() - Formate le résultat pour l'affichage."""
        status = "APPROVED" if prediction == 1 else "REJECTED"
        confidence = probability * 100
        
        # Évaluer le risque
        if probability >= 0.75:
            risk = "Low"
        elif probability >= 0.5:
            risk = "Medium"
        else:
            risk = "High"
        
        self.logger.info(f"Prédiction: {status}, Confiance: {confidence:.2f}%, Risque: {risk}")
        
        return {
            "statut": status,
            "prediction": prediction,
            "confiance_pct": f"{confidence:.2f}%",
            "risque": risk,
            "data_client": client_data
        }
