"""
Décorateurs pour logging et timing - concepts fonctionnels
"""
import time
import functools
from datetime import datetime
from typing import Callable, Any


def log_execution(func: Callable) -> Callable:
    """
    Décorateur pour logger l'exécution d'une fonction.
    
    Exemple:
        @log_execution
        def train_model():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] RUN   {func.__name__}")
        try:
            result = func(*args, **kwargs)
            print(f"[{timestamp}] OK    {func.__name__}")
            return result
        except Exception as e:
            print(f"[{timestamp}] ERROR {func.__name__}: {str(e)}")
            raise
    return wrapper


def timing(func: Callable) -> Callable:
    """
    Décorateur pour mesurer le temps d'exécution.
    
    Exemple:
        @timing
        def predict(data):
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"[TIME] {func.__name__} took {elapsed_time:.4f}s")
        return result
    return wrapper


def cache_result(func: Callable) -> Callable:
    """
    Simple cache decorator pour éviter calculs répétés.
    
    Exemple:
        @cache_result
        def expensive_operation():
            pass
    """
    cache_dict = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Créer une clé simple
        key = (args, tuple(sorted(kwargs.items())))
        
        if key in cache_dict:
            print(f"[CACHE] hit for {func.__name__}")
            return cache_dict[key]
        
        result = func(*args, **kwargs)
        cache_dict[key] = result
        return result
    
    return wrapper


def validate_input(func: Callable) -> Callable:
    """
    Décorateur pour valider que les inputs ne sont pas None.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        if None in args or None in kwargs.values():
            raise ValueError("Input cannot be None")
        return func(*args, **kwargs)
    return wrapper
