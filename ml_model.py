import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from data_ingestion import get_data
import warnings

warnings.filterwarnings("ignore")

def prepare_causal_features(df):
    print("Construction des variables prédictives basées sur le graphe causal...")
    
    # Cible : La direction de MTUM à T+1 (1 si positif, 0 si négatif)
    y = np.where(df['MTUM'].shift(-1) > df['MTUM'], 1, 0)
    
    # Variables prédictives : L'état du monde à T (lag 1)
    X = pd.DataFrame(index=df.index)
    X['TNX_lag1'] = df['^TNX'].shift(1)  # Impact des taux d'intérêt
    X['IVW_lag1'] = df['IVW'].shift(1)   # Impact de la croissance
    X['MTUM_lag1'] = df['MTUM'].shift(1) # Inertie (autocorrélation)
    
    X['Target'] = y
    X = X.dropna()
    
    y_clean = X.pop('Target')
    return X, y_clean

def run_trading_model():
    df = get_data()
    X, y = prepare_causal_features(df)
    
    # Séparation temporelle stricte : 80% Entraînement, 20% Test (Futur)
    # Règle d'or en finance : ne jamais mélanger aléatoirement des séries temporelles
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nEntraînement de l'algorithme XGBoost sur {len(X_train)} jours de marché...")
    
    # Calibrage strict pour éviter d'apprendre par cœur le bruit du marché
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print("\n--- ÉVALUATION SUR LES DONNÉES INCONNUES ---")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Précision directionnelle (Accuracy) : {accuracy:.4f}\n")
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))
    
    print("\nPoids des variables causales dans la décision :")
    importances = model.feature_importances_
    for col, imp in zip(X.columns, importances):
        print(f"{col}: {imp*100:.1f}%")

if __name__ == "__main__":
    run_trading_model()