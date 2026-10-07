"""
Programmation Fonctionnelle Avancée - Fonctions d'ordre supérieur
Démontre: map, filter, reduce, compose, pipeline, currying, etc.
"""
import functools
from typing import Callable, TypeVar, List, Dict, Any, Tuple
import operator

T = TypeVar('T')
U = TypeVar('U')


# ============= FONCTIONS D'ORDRE SUPÉRIEUR =============

def map_transform(transform_fn: Callable[[T], U]) -> Callable[[List[T]], List[U]]:
    """
    Curried function pour transformer une liste.
    
    Exemple:
        double = map_transform(lambda x: x * 2)
        result = double([1, 2, 3])  # [2, 4, 6]
    """
    def mapper(items: List[T]) -> List[U]:
        return list(map(transform_fn, items))
    return mapper


def filter_by(predicate: Callable[[T], bool]) -> Callable[[List[T]], List[T]]:
    """
    Curried function pour filtrer une liste.
    
    Exemple:  
        is_even = filter_by(lambda x: x % 2 == 0)
        result = is_even([1, 2, 3, 4])  # [2, 4]
    """
    def filterer(items: List[T]) -> List[T]:
        return list(filter(predicate, items))
    return filterer


def reduce_items(reducer: Callable[[T, U], T], initial: T) -> Callable[[List[U]], T]:
    """
    Curried function pour réduire une liste.
    
    Exemple:
        sum_all = reduce_items(lambda acc, x: acc + x, 0)
        result = sum_all([1, 2, 3, 4])  # 10
    """
    def reducer_fn(items: List[U]) -> T:
        return functools.reduce(reducer, items, initial)
    return reducer_fn


# ============= COMPOSITION DE FONCTIONS =============

def compose(*functions: Callable) -> Callable:
    """
    Compose plusieurs fonctions (droite à gauche).
    
    Exemple:
        add_one = lambda x: x + 1
        double = lambda x: x * 2
        process = compose(double, add_one)
        result = process(5)  # (5 + 1) * 2 = 12
    """
    def composed(x):
        return functools.reduce(lambda val, f: f(val), reversed(functions), x)
    return composed


def pipe(*functions: Callable) -> Callable:
    """
    Pipe plusieurs fonctions (gauche à droite).
    
    Exemple:
        add_one = lambda x: x + 1
        double = lambda x: x * 2
        process = pipe(add_one, double)
        result = process(5)  # (5 + 1) * 2 = 12
    """
    def piped(x):
        return functools.reduce(lambda val, f: f(val), functions, x)
    return piped


# ============= CURRYING =============

def curry(fn: Callable) -> Callable:
    """
    Transforme une fonction en version currifiée.
    
    Exemple:
        def add(a, b, c):
            return a + b + c
        
        curried_add = curry(add)
        result = curried_add(1)(2)(3)  # 6
    """
    @functools.wraps(fn)
    def curried(*args):
        if len(args) >= fn.__code__.co_argcount:
            return fn(*args)
        return functools.partial(curried, *args)
    return curried


# ============= MEMOIZATION =============

def memoize(fn: Callable) -> Callable:
    """
    Cache les résultats des appels de fonction.
    
    Exemple:
        @memoize
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
    """
    cache = {}
    
    @functools.wraps(fn)
    def memoized(*args):
        if args in cache:
            return cache[args]
        result = fn(*args)
        cache[args] = result
        return result
    
    return memoized


# ============= FONCTIONS POUR ML =============

def normalize(features: List[float]) -> List[float]:
    """Normalise les features entre 0 et 1."""
    min_val = min(features)
    max_val = max(features)
    range_val = max_val - min_val
    
    if range_val == 0:
        return [0] * len(features)
    
    return [(x - min_val) / range_val for x in features]


def standardize(features: List[float]) -> List[float]:
    """Standardise les features (z-score)."""
    mean = sum(features) / len(features)
    variance = sum((x - mean) ** 2 for x in features) / len(features)
    std_dev = variance ** 0.5
    
    if std_dev == 0:
        return [0] * len(features)
    
    return [(x - mean) / std_dev for x in features]


# ============= PIPELINE FONCTIONNEL =============

def pipeline_data(
    data: List[Dict[str, Any]],
    transformations: List[Callable[[List[Dict]], List[Dict]]]
) -> List[Dict]:
    """
    Applique une série de transformations fonctionnelles aux données.
    
    Exemple:
        pipeline_data(
            data,
            [
                filter_by(lambda x: x['age'] > 25),
                map_transform(lambda x: {**x, 'age': x['age'] * 2})
            ]
        )
    """
    return functools.reduce(
        lambda data, transform: transform(data),
        transformations,
        data
    )


# ============= PARTIAL APPLICATION =============

def create_classifier(threshold: float) -> Callable[[float], bool]:
    """
    Crée un classifieur avec un seuil spécifique.
    
    Exemple:
        approves = create_classifier(0.5)
        result = approves(0.7)  # True
    """
    return lambda probability: probability >= threshold


# ============= DECORATEURS FONCTIONNELS =============

def with_logging(fn: Callable) -> Callable:
    """Décorateur qui log les entrées et sorties."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"[CALL] {fn.__name__}({args}, {kwargs})")
        result = fn(*args, **kwargs)
        print(f"[RESULT] {result}")
        return result
    return wrapper


def with_retry(max_attempts: int = 3) -> Callable:
    """Décorateur qui réessaye en cas d'erreur."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    print(f"[WARN] Attempt {attempt}/{max_attempts} failed. Retrying...")
            return None
        return wrapper
    return decorator


# ============= EXEMPLE D'UTILISATION =============

if __name__ == "__main__":
    # Exemple 1: Composition
    print("=" * 60)
    print("Exemple 1: Composition de fonctions")
    print("=" * 60)
    
    add_ten = lambda x: x + 10
    multiply_by_two = lambda x: x * 2
    
    process = pipe(add_ten, multiply_by_two)
    result = process(5)
    print(f"pipe(add_ten, multiply_by_two)(5) = {result}")  # (5 + 10) * 2 = 30
    
    # Exemple 2: Currying
    print("\n" + "=" * 60)
    print("Exemple 2: Currying")
    print("=" * 60)
    
    def add(a, b, c):
        return a + b + c
    
    curried_add = curry(add)
    result = curried_add(1)(2)(3)
    print(f"curry(add)(1)(2)(3) = {result}")  # 6
    
    # Exemple 3: Map/Filter
    print("\n" + "=" * 60)
    print("Exemple 3: Map et Filter")
    print("=" * 60)
    
    numbers = [1, 2, 3, 4, 5]
    
    double = map_transform(lambda x: x * 2)
    is_even = filter_by(lambda x: x % 2 == 0)
    
    doubled = double(numbers)
    evens = is_even(numbers)
    
    print(f"double({numbers}) = {doubled}")  # [2, 4, 6, 8, 10]
    print(f"is_even({numbers}) = {evens}")   # [2, 4]
