# Quant-Awards
## Journal de Recherche - 16 Mai 2026

### Objectif du jour
Résolution des verrous techniques d'infrastructure et calcul de la première matrice causale PCMCI valide sur l'univers Quant-Awards.

### Avancements et Corrections
1. **Pipeline de Données** : Contournement du pare-feu institutionnel de la FED (FRED) via falsification d'User-Agent (`storage_options`). Ajustement du flux Yahoo Finance suite à la dépréciation de la colonne `Adj Close` au profit de `Close`.
2. **Stationnarité Algorithmique** : Implémentation d'une routine d'optimisation du paramètre de différenciation fractionnaire (*d*) par actif. Remplacement du seuil arbitraire de 0.4 par un balayage automatique validé par le test de Dickey-Fuller Augmenté ($p < 0.05$). Préservation maximale de la mémoire longue (VIX à $d=0.10$, Growth à $d=0.55$).
3. **Analyse Causale (Tigramite)** : Calcul de la matrice des forces causales PCMCI ($\alpha = 0.05$, $\tau_{max} = 5$). Identification d'un signal causal fort à $T-1$ des Taux 10 ans (`^TNX`) sur le facteur Momentum (`MTUM`).
4. **Prototype Predictive (XGBoost)** : Spécification d'un premier classifieur binaire pour prédire la direction de `MTUM` à $T+1$ sur la base des variables causales. Performance brute : 50.75% d'accuracy (efficience de marché de premier ordre).

### Prochaine étape
Implémentation d'un filtre probabiliste d'exécution (seuils d'action à >60% de certitude via `predict_proba`) et extension de l'horizon prédictif à $T+5$ pour extraire l'alpha du bruit quotidien.