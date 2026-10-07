"""
DOCUMENTATION: Dependency Inversion Principle (DIP) & Architecture
====================================================================

Ce document explique comment le DIP est implémenté dans le projet.
"""

# ============================================================================
# 1. QU'EST-CE QUE LE DEPENDENCY INVERSION PRINCIPLE (DIP) ?
# ============================================================================

"""
Le DIP est un principe SOLID qui stipule:

    "Les modules de haut niveau ne doivent pas dépendre des modules de bas niveau.
     Les deux doivent dépendre d'abstractions."

En Python, les abstractions sont les INTERFACES (classes abstraites avec ABC).

MAUVAIS (Sans DIP):
-------------------
class DataLoader:  # Bas niveau
    def load(self):
        pass

class Model:  # Haut niveau
    def __init__(self):
        self.loader = DataLoader()  # ❌ Dépend directement de DataLoader
    
    def train(self):
        data = self.loader.load()
        # ...

PROBLÈME: Si on veut changer DataLoader, il faut modifier Model !

BON (Avec DIP):
---------------
class IDataProcessor(ABC):  # ABSTRACTION
    @abstractmethod
    def load(self):
        pass

class DataLoader(IDataProcessor):  # Bas niveau - implémente l'interface
    def load(self):
        pass

class Model:  # Haut niveau
    def __init__(self, data_processor: IDataProcessor):  # ✅ Dépend de l'interface
        self.data_processor = data_processor
    
    def train(self):
        data = self.data_processor.load()
        # ...

AVANTAGE: On peut passer n'importe quelle implémentation de IDataProcessor !
"""


# ============================================================================
# 2. STRUCTURE DU PROJET AVEC DIP
# ============================================================================

"""
src/
├── interfaces.py          ← ABSTRACTIONS (IDataProcessor, IModel, ILogger, etc.)
├── data_loader.py         ← CONCRÈTE: Implémente IDataProcessor
├── model.py               ← CONCRÈTE: Implémente IModel
├── predictor.py           ← CONCRÈTE: Implémente IPredictor
├── logger.py              ← CONCRÈTE: Implémente ILogger (ConsoleLogger, FileLogger)
├── repository.py          ← CONCRÈTE: Implémente IRepository (MLOps)
├── container.py           ← IoC Container (Injection de Dépendances)
├── decorators.py          ← Décorateurs fonctionnels
├── functional.py          ← Programmation fonctionnelle avancée
└── utils.py               ← Fonctions utilitaires

FLUX DE DÉPENDANCES:
====================

                    IDataProcessor (Abstract)
                          ▲
                          │ implements
                          │
                    DataProcessor (Concrete)

                    IModel (Abstract)
                          ▲
                          │ implements
                          │
                    CreditModel (Concrete)

                    ILogger (Abstract)
                          ▲
                          │ implements
                          │
        ┌───────────────┬──┴──────────┬──────────────┐
        │               │             │              │
    ConsoleLogger  FileLogger  HybridLogger

                    DIContainer
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
create_data_processor  create_model       create_predictor
    │                     │                     │
    ▼                     ▼                     ▼
DataProcessor         CreditModel         CreditPredictor
(IDataProcessor)      (IModel)            (IPredictor)
"""


# ============================================================================
# 3. INJECTION DE DÉPENDANCES (DEPENDENCY INJECTION)
# ============================================================================

"""
L'injection de dépendances est la technique pour réaliser le DIP.

TROIS FORMES D'INJECTION:
=========================

1. CONSTRUCTOR INJECTION (PRÉFÉRÉE) - Ce qu'on utilise:
   -------------------------------------------------
   class CreditModel(IModel):
       def __init__(self, logger: ILogger = None, repository: IRepository = None):
           self.logger = logger or ConsoleLogger()
           self.repository = repository or ModelRepository()

   model = CreditModel(
       logger=HybridLogger(),
       repository=ModelRepository()
   )

2. SETTER INJECTION:
   class CreditModel(IModel):
       def __init__(self):
           self._logger = None
       
       def set_logger(self, logger: ILogger):
           self._logger = logger

3. INTERFACE INJECTION:
   class Configurable(ABC):
       @abstractmethod
       def configure(self, container: DIContainer):
           pass


CONTENEUR IoC (INVERSION OF CONTROL):
======================================

Le DIContainer gère automatiquement l'injection de dépendances:

    container = initialize_container()
    
    # Le conteneur crée automatiquement les dépendances
    data_processor = container.create_data_processor("data.csv")
    model = container.create_model("random_forest")
    predictor = container.create_predictor(model, data_processor)

AVANTAGES:
- Couplage faible entre les classes
- Facile de tester (mock les dépendances)
- Facile de changer les implémentations
"""


# ============================================================================
# 4. TECHNOLOGIES UTILISÉES
# ============================================================================

"""
OOP (Object-Oriented Programming):
==================================
✅ Classes abstraites (ABC) pour définir les contrats
✅ Héritage avec les interfaces
✅ Encapsulation des données
✅ Polymorphisme

Exemple:
class IModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train) -> None:
        pass

class CreditModel(IModel):
    def train(self, X_train, y_train) -> None:
        self.model.fit(X_train, y_train)


Machine Learning:
==================
✅ scikit-learn pour les modèles (LogisticRegression, RandomForest)
✅ sklearn.preprocessing (StandardScaler, LabelEncoder)
✅ sklearn.metrics pour l'évaluation

Exemple:
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, y_pred)


Classification:
================
✅ Problème de classification binaire (Approuvé/Rejeté)
✅ Modèles: Logistic Regression, Random Forest
✅ Probabilités pour évaluer le risque

Exemple:
predictions = model.predict(X)  # [0, 1, 1, 0, ...]
probabilities = model.predict_proba(X)  # [[0.7, 0.3], [0.2, 0.8], ...]


FastAPI:
=========
✅ API REST avec endpoints
✅ Pydantic pour la validation
✅ Interface Web interactive

Exemple:
@app.post("/api/predict")
async def predict(data: PredictionInput):
    return predictor.predict_single(data.dict())


Programmation Fonctionnelle:
=============================
✅ Fonctions d'ordre supérieur (map, filter, reduce)
✅ Composition de fonctions (pipe, compose)
✅ Currying et partial application
✅ Memoization

Exemple:
process = pipe(
    map_transform(lambda x: x * 2),
    filter_by(lambda x: x > 10)
)
result = process([1, 5, 10, 15])  # [12, 20, 30]


Décorateurs:
=============
✅ @log_execution - Log les appels
✅ @timing - Mesure le temps
✅ @cache_result - Cache les résultats
✅ @validate_input - Valide les inputs
✅ @with_retry - Réessaye en cas d'erreur
✅ @with_logging - Log entrées/sorties

Exemple:
@log_execution
@timing
def train(self, X_train, y_train):
    self.model.fit(X_train, y_train)


MLOps:
======
✅ Repository Pattern pour persister les modèles
✅ Versioning automatique des modèles
✅ Sauvegarde des métadonnées
✅ Logging structuré

Exemple:
repository = ModelRepository()
repository.save_model(model, "model_v1.pkl")
repository.save_metadata({"accuracy": 0.95}, "model_v1_info.json")
"""


# ============================================================================
# 5. EXEMPLE: FLUX COMPLET AVEC DIP
# ============================================================================

"""
AVANT (Sans DIP):
=================
def main():
    loader = DataLoader("data.csv")  # ❌ Couplage direct
    loader.load_data()
    loader.clean_data()
    
    model = CreditModel()  # ❌ Couplage direct
    model.train(X_train, y_train)
    
    predictor = Predictor(model, loader)  # ❌ Couplage direct

PROBLÈMES:
- Difficile à tester (pas de mocks)
- Difficile de changer les implémentations
- Beaucoup de dépendances manuelles


APRÈS (Avec DIP et IoC):
=========================
def main():
    # Initialiser le conteneur (IoC)
    container = initialize_container()
    logger = container.get_service('logger')
    
    # Créer les dépendances via le conteneur
    data_processor: IDataProcessor = container.create_data_processor("data.csv")
    model: IModel = container.create_model("random_forest")
    predictor: IPredictor = container.create_predictor(model, data_processor)
    
    # Utiliser les interfaces
    data_processor.load()
    data_processor.clean()
    data_processor.encode()
    
    X_train, X_test, y_train, y_test = data_processor.prepare()
    
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    
    model.save("model.pkl")

AVANTAGES:
✅ Couplage faible
✅ Facile à tester
✅ Facile de changer les implémentations
✅ Code maintenable et extensible
"""


# ============================================================================
# 6. TESTING AVEC DIP
# ============================================================================

"""
AVANT (Difficile à tester):
============================
class Model:
    def __init__(self):
        self.logger = ConsoleLogger()  # ❌ Impossible à mocker
    
    def train(self, X, y):
        self.logger.info("Training...")  # ❌ Affiche partout

# Test difficile parce qu'on ne peut pas contrôler le logger


APRÈS (Facile à tester):
=========================
class MockLogger(ILogger):
    def __init__(self):
        self.messages = []
    
    def info(self, msg):
        self.messages.append(msg)

class Model(IModel):
    def __init__(self, logger: ILogger = None):
        self.logger = logger or ConsoleLogger()

# Test
def test_model_logging():
    mock_logger = MockLogger()
    model = CreditModel(logger=mock_logger)
    model.train(X, y)
    
    assert "Training..." in mock_logger.messages

AVANTAGE: On contrôle totalement les dépendances dans les tests !
"""


# ============================================================================
# 7. HIÉRARCHIE DES INTERFACES
# ============================================================================

"""
                        ABC (Abstract Base Class)
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
    ILogger              IModel              IDataProcessor
        │                    │                    │
    ┌───┴──────┐          │                    │
    │   │      │          │                    │
Console File Hybrid      │                    │
Logger  Logger Logger     │                    │
                          │                    │
                     CreditModel         DataProcessor
                          │                    │
    ┌──────────────────────┼────────────────────┘
    │                      │
    └──────────┬───────────┘
               │
         IPredictor
               │
        CreditPredictor
               │
          FastAPI
             Route
"""


if __name__ == "__main__":
    print(__doc__)
