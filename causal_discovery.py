import numpy as np
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

def discover_causality(df_final):
    noms_variables = df_final.columns.tolist()
    dataframe = pp.DataFrame(
        df_final.values, 
        datatime=np.arange(len(df_final)), 
        var_names=noms_variables
    )

    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

    tau_max = 5
    resultats = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=0.01)

    matrice_valeurs = resultats['val_matrix']
    print("Matrice des forces causales :")
    print(np.round(matrice_valeurs, 3))
    
    return resultats