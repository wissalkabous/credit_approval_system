"""
Pattern Repository pour la persistance - MLOps
Respecte DIP: Les dépendances pointent vers les abstractions
"""
import pickle
import json
from pathlib import Path
from typing import Any, Dict
from src.interfaces import IRepository


class ModelRepository(IRepository):
    """Repository pour la gestion des modèles et métadonnées."""
    
    def __init__(self, base_path: str = "models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: Any, filepath: str) -> None:
        """
        Sauvegarde un modèle en pickle.
        
        Args:
            model: Le modèle scikit-learn
            filepath: Chemin où sauvegarder
        """
        path = self.base_path / filepath
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"[SAVED] Model: {path}")
    
    def load_model(self, filepath: str) -> Any:
        """
        Charge un modèle depuis pickle.
        
        Args:
            filepath: Chemin du modèle
            
        Returns:
            Le modèle chargé
        """
        path = self.base_path / filepath
        
        if not path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {path}")
        
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"[LOADED] Model: {path}")
        return model
    
    def save_metadata(self, metadata: Dict, filepath: str) -> None:
        """
        Sauvegarde les métadonnées du modèle.
        
        Args:
            metadata: Dictionnaire avec infos du modèle
            filepath: Chemin où sauvegarder
        """
        path = self.base_path / filepath
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
        
        print(f"[SAVED] Metadata: {path}")
    
    def load_metadata(self, filepath: str) -> Dict:
        """
        Charge les métadonnées du modèle.
        
        Args:
            filepath: Chemin des métadonnées
            
        Returns:
            Dictionnaire avec les infos
        """
        path = self.base_path / filepath
        
        if not path.exists():
            raise FileNotFoundError(f"Métadonnées non trouvées: {path}")
        
        with open(path, 'r') as f:
            metadata = json.load(f)
        
        print(f"[LOADED] Metadata: {path}")
        return metadata
    
    def get_model_version(self) -> int:
        """Retourne la version du dernier modèle sauvegardé."""
        pkl_files = list(self.base_path.glob("*.pkl"))
        return len(pkl_files)
