"""
Package source principal - Respecte le DIP
"""
# Interfaces abstraites
from .interfaces import (
    IDataProcessor,
    IModel,
    IPredictor,
    IScaler,
    IRepository,
    ILogger
)

# Implémentations concrètes
from .data_loader import DataProcessor
from .model import CreditModel
from .predictor import CreditPredictor
from .logger import ConsoleLogger, FileLogger, HybridLogger
from .repository import ModelRepository

# Décorateurs
from .decorators import log_execution, timing, cache_result, validate_input

# Utilitaires
from .utils import (
    apply_to_list,
    filter_by_condition,
    ensure_directory,
    save_json,
    load_json,
    format_prediction_result
)

# Programmation Fonctionnelle
from .functional import (
    map_transform,
    filter_by,
    reduce_items,
    compose,
    pipe,
    curry,
    memoize,
    pipeline_data,
    create_classifier
)

# Inversion of Control Container
from .container import DIContainer, get_container, initialize_container

__all__ = [
    # Interfaces
    'IDataProcessor',
    'IModel',
    'IPredictor',
    'IScaler',
    'IRepository',
    'ILogger',
    # Implémentations
    'DataProcessor',
    'CreditModel',
    'CreditPredictor',
    'ConsoleLogger',
    'FileLogger',
    'HybridLogger',
    'ModelRepository',
    # Décorateurs
    'log_execution',
    'timing',
    'cache_result',
    'validate_input',
    # Utilitaires
    'apply_to_list',
    'filter_by_condition',
    'ensure_directory',
    'save_json',
    'load_json',
    'format_prediction_result',
    # Fonctionnel
    'map_transform',
    'filter_by',
    'reduce_items',
    'compose',
    'pipe',
    'curry',
    'memoize',
    'pipeline_data',
    'create_classifier',
    # IoC
    'DIContainer',
    'get_container',
    'initialize_container',
]

