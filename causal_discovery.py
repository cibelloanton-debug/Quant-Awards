import numpy as np
import matplotlib.pyplot as plt
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import tigramite.plotting as tp

def discover_causality(df):
    print("Initialisation de l'algorithme PCMCI...")
    
    # Conversion stricte au format Tigramite
    dataframe = pp.DataFrame(df.values, 
                             datatime=np.arange(len(df)), 
                             var_names=df.columns)
    
    # Test de corrélation partielle linéaire
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
    
    # Paramétrage de la purge statistique
    tau_max = 5
    pc_alpha = 0.05
    print(f"Exécution avec filtrage du bruit (alpha = {pc_alpha})...")
    
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)
    
    print("\nGénération du graphe directionnel...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    tp.plot_graph(
        val_matrix=results['val_matrix'],
        graph=results['graph'],
        var_names=df.columns,
        link_colorbar_label='Force causale croisée (Cross-MCI)',
        node_colorbar_label='Inertie (Auto-MCI)',
        fig_ax=(fig, ax)
    )
    
    # Sauvegarde physique pour contourner les limitations d'Onyxia
    fichier_sortie = 'graphe_causal_pur.png'
    plt.savefig(fichier_sortie, dpi=300, bbox_inches='tight')
    print(f"Succès. Le graphe a été gravé dans le fichier : {fichier_sortie}")
    
    return results