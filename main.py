from data_ingestion import get_data
from preprocessing import align_and_clean
from causal_discovery import discover_causality

if __name__ == "__main__":
    print("Début de l'extraction des données...")
    df_fractionnel = get_data()
    
    print("\nNettoyage et alignement...")
    df_propre = align_and_clean(df_fractionnel)
    
    print("\nCalcul de la matrice causale PCMCI...")
    resultats_causaux = discover_causality(df_propre)