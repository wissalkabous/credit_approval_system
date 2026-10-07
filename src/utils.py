"""
Utilitaires - Programmation Fonctionnelle
"""
import json
import functools
from typing import Any, Callable, List, Dict
from pathlib import Path


def apply_to_list(data: List[Any], func: Callable) -> List[Any]:
    """
    Applique une fonction à chaque élément d'une liste (map).
    """
    return list(map(func, data))


def filter_by_condition(data: List[Dict], key: str, condition: Callable) -> List[Dict]:
    """
    Filtre une liste selon une condition.
    
    Exemple:
        filter_by_condition(data, 'age', lambda x: x > 25)
    """
    return list(filter(lambda item: condition(item.get(key)), data))


def compose(*functions: Callable) -> Callable:
    """
    Compose plusieurs fonctions.
    
    Exemple:
        process = compose(normalize, validate, transform)
        result = process(data)
    """
    def composed(x):
        return functools.reduce(lambda val, f: f(val), reversed(functions), x)
    return composed


def ensure_directory(path: str) -> Path:
    """Crée un répertoire s'il n'existe pas."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_json(data: Dict, filepath: str) -> None:
    """Sauvegarde un dictionnaire en JSON."""
    ensure_directory(Path(filepath).parent)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"[SAVED] {filepath}")


def load_json(filepath: str) -> Dict:
    """Charge un fichier JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_prediction_result(prediction: int, probability: float, client_info: Dict) -> Dict:
    """Formate le résultat de prédiction pour l'affichage."""
    status = "APPROVED" if prediction == 1 else "REJECTED"
    confidence = probability * 100

    return {
        "statut": status,
        "prediction": int(prediction),
        "confiance": f"{confidence:.2f}%",
        "client": client_info,
        "risque": "Low" if probability > 0.7 else "Medium" if probability > 0.5 else "High"
    }
