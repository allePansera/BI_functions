import numpy as np
import pandas as pd

from SchemaMapping.BottomUp import local_matching_table
from SchemaMapping.TopDown import global_matching_table
from Validation.ValidationMetrics import complete_validation, get_validation_metric
from utils.utils import to_GMM, generate_LAT, confronta_LAT_GoldStandard, missed_from_sources

# 15-Lab-ESAME) DataIntegration_SchemaMatchingSchemaIntegration.ipynb
# ESERIZIO 30 OTTOBRE (Bottom-Up)

src_links = [
    'https://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/EsempioBook/BookA1.csv',
    'https://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/EsempioBook/BookB1.csv',
    'https://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/EsempioBook/BookC1.csv'
]

SOURCES = {'S' + str(i + 1): pd.read_csv(link).astype(str) for i, link in enumerate(src_links)}

GlobalSchema = pd.read_csv('https://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/EsempioBook/GlobalSchema.csv').astype(str)

# questo è il gold standard dato
GMT_GS = pd.read_csv('https://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/EsempioBook/GMT_GSesempio.csv').astype(str)
# print(to_GMM(GMT_GS))

# confronta_LAT_GoldStandard(GMT_GS, generate_LAT(SOURCES))

SUPPORTED_METHODS = {
    'label': ['LEV', 'JAR', 'JAC', 'OVC', 'J_W', 'M_E'],
    'instance': ['JAC', 'G_J', 'EJL', 'SIM']
}

# GMTcalcolata = global_matching_table(sources=SOURCES,
#                                      global_schema=GlobalSchema,
#                                      methods={'label': ['LEV'], 'instance': ['SIM']}, functions={'SIM': 'JAC'}, sim_thresholds={'SIM': 0.3},
#                                      combiner='AvgSim',
#                                      combiner_threshold=0.3,
#                                      weights=None,
#                                      exec_stable_matching=True,
#                                      exec_symmetric_best_match=False,
#                                      K=None,
#                                      column='A')
# print(to_GMM(GMTcalcolata))
# results = complete_validation(GoldStandard=GMT_GS, GlobalMatchingTable=GMTcalcolata, id_cols=['GAT', 'SLAT'], title="1-1")
# print(f"\n{get_validation_metric(GMT_GS, GMTcalcolata, metrics='FP', id_cols=['GAT', 'SLAT'])[['GAT', 'SLAT', '_merge']]}\n")

######################################################################################################################################################
# Tentativo 1, essendo una N-N non ha senso applicare stable_marriage che va a cercare un mapping 1-1.
######################################################################################################################################################
# GMTcalcolata2 = global_matching_table(sources=SOURCES,
#                                       global_schema=GlobalSchema,
#                                       methods={'label': ['LEV'], 'instance': ['SIM']}, functions={'SIM': 'JAC'}, sim_thresholds={'SIM': 0.3},
#                                       combiner='AvgSim',
#                                       combiner_threshold=0.3,
#                                       weights=None,
#                                       exec_stable_matching=False,
#                                       exec_symmetric_best_match=False,
#                                       K=None,
#                                       column='A')
# print(to_GMM(GMTcalcolata2))
# results = complete_validation(GoldStandard=GMT_GS, GlobalMatchingTable=GMTcalcolata2, id_cols=['GAT', 'SLAT'], title="N-N",
#                               ValidationTable=results)
# print(results)
# print(f"\n{get_validation_metric(GMT_GS, GMTcalcolata2, metrics='FP', id_cols=['GAT', 'SLAT'])[['GAT', 'SLAT', '_merge']]}\n")

######################################################################################################################################################
# Tentativo 3, noto un gran numero di FP, quindi alzo la soglia.
######################################################################################################################################################
# GMTcalcolata3 = global_matching_table(sources=SOURCES,
#                                       global_schema=GlobalSchema,
#                                       methods={'label': ['LEV'], 'instance': ['SIM']}, functions={'SIM': 'JAC'}, sim_thresholds={'SIM': 0.5},
#                                       combiner='AvgSim',
#                                       combiner_threshold=0.3,
#                                       weights=None,
#                                       exec_stable_matching=False,
#                                       exec_symmetric_best_match=False,
#                                       K=None,
#                                       column='A')
# print(to_GMM(GMTcalcolata2))
# results = complete_validation(GoldStandard=GMT_GS, GlobalMatchingTable=GMTcalcolata3, id_cols=['GAT', 'SLAT'], title="N-N_0.5",
#                               ValidationTable=results)
# print(results)
# print(f"\n{get_validation_metric(GMT_GS, GMTcalcolata3, metrics='FP', id_cols=['GAT', 'SLAT'])[['GAT', 'SLAT', '_merge']]}\n")

######################################################################################################################################################
# Tentativo 4, noto un gran numero di FP, quindi alzo la soglia.
######################################################################################################################################################
# GMTcalcolata4 = global_matching_table(sources=SOURCES,
#                                       global_schema=GlobalSchema,
#                                       methods={'label': ['LEV', 'JAC'], 'instance': ['SIM']}, functions={'SIM': 'JAC'}, sim_thresholds={'SIM': 0.3},
#                                       combiner='AvgSim',
#                                       combiner_threshold=0.3,
#                                       weights=None,
#                                       exec_stable_matching=False,
#                                       exec_symmetric_best_match=False,
#                                       K=None,
#                                       column='A')
# print(to_GMM(GMTcalcolata4))
# results = complete_validation(GoldStandard=GMT_GS, GlobalMatchingTable=GMTcalcolata4, id_cols=['GAT', 'SLAT'], title="N-N+JAC")
# print(results)
# print(f"\n{get_validation_metric(GMT_GS, GMTcalcolata4, metrics='FP', id_cols=['GAT', 'SLAT'])[['GAT', 'SLAT', '_merge']]}\n")

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################

# vediamo un altro esempio, cambiando gold standard
GMT_GS = pd.read_csv('http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/EsempioBook/GS_1_1_B.csv').astype(str)
print(f"\n{to_GMM(GMT_GS)}\n\n{'-•-' * 50}\n")

# confronta_LAT_GoldStandard(GMT_GS, generate_LAT(SOURCES))

GMTcalcolata = global_matching_table(sources=SOURCES,
                                     global_schema=GlobalSchema,
                                     methods={'label': ['LEV', 'JAC'], 'instance': ['SIM']}, functions={'SIM': 'JAC'}, sim_thresholds={'SIM': 0.3},
                                     combiner='MinSim',
                                     combiner_threshold=0.3,
                                     weights=None,
                                     exec_stable_matching=False,
                                     exec_symmetric_best_match=True,
                                     K=None,
                                     column='A')

# print(f"\n{'-•-' * 20} GMT CALCOLATA {'-•-' * 20}\n\n{to_GMM(GMTcalcolata)}\n\n")
results = complete_validation(GoldStandard=GMT_GS, GlobalMatchingTable=GMTcalcolata, id_cols=['GAT', 'SLAT'], title="Min")
# print(f"\n{results}\n\n")
# print(f"\n{get_validation_metric(GMT_GS, GMTcalcolata, metrics='FN', id_cols=['GAT', 'SLAT'])[['GAT', 'SLAT', '_merge']]}\n\n")

#      MT  TP  FP FN       P       R       F
# Min  14  14   0  4  1.0000  0.7778  0.8750

######################################################################################################################################################
# Tentativo 2, noto un gran numero di FN, quindi provo a cambiare il combiner, in modo da alzare il risultato delle similarity (infatti prendo
# la media e non il minimo). Rischio di ottenere dei FP.
######################################################################################################################################################
GMTcalcolata = global_matching_table(sources=SOURCES,
                                     global_schema=GlobalSchema,
                                     methods={'label': ['LEV', 'JAC'], 'instance': ['SIM']}, functions={'SIM': 'JAC'}, sim_thresholds={'SIM': 0.3},
                                     combiner='AvgSim',
                                     combiner_threshold=0.3,
                                     weights=None,
                                     exec_stable_matching=False,
                                     exec_symmetric_best_match=True,
                                     K=None,
                                     column='A')

# print(f"\n{'-•-' * 20} GMT CALCOLATA {'-•-' * 20}\n\n{to_GMM(GMTcalcolata)}\n\n")
results = complete_validation(GoldStandard=GMT_GS, GlobalMatchingTable=GMTcalcolata, id_cols=['GAT', 'SLAT'], title="Avg_0.3",
                              ValidationTable=results)
# print(f"\n{results}\n\n")
# print(f"\nFalse Negative (FN):\n{get_validation_metric(GMT_GS, GMTcalcolata, metrics='FN', id_cols=['GAT', 'SLAT'])[['GAT', 'SLAT', '_merge']]}\n\n")
# print(f"\nFalse Positive (FP):\n{get_validation_metric(GMT_GS, GMTcalcolata, metrics='FP', id_cols=['GAT', 'SLAT'])[['GAT', 'SLAT', '_merge']]}\n\n")

#          MT  TP  FP FN       P       R       F
# Avg_0.3  27  18   9  0  0.6667  1.0000  0.8000


######################################################################################################################################################
# Tentativo 3, ho ottenuto 0 FN, bene. Il problema però, come previsto, è che aumenta il numero di FP.
# Per risolvere il problema posso aumentare la soglia del combiner.
######################################################################################################################################################
GMTcalcolata = global_matching_table(sources=SOURCES,
                                     global_schema=GlobalSchema,
                                     methods={'label': ['LEV', 'JAC'], 'instance': ['SIM']}, functions={'SIM': 'JAC'}, sim_thresholds={'SIM': 0.3},
                                     combiner='AvgSim',
                                     combiner_threshold=0.4,
                                     weights=None,
                                     exec_stable_matching=True,
                                     exec_symmetric_best_match=False,
                                     K=None,
                                     column='A')

# print(f"\n{'-•-' * 20} GMT CALCOLATA {'-•-' * 20}\n\n{to_GMM(GMTcalcolata)}\n\n")
results = complete_validation(GoldStandard=GMT_GS, GlobalMatchingTable=GMTcalcolata, id_cols=['GAT', 'SLAT'], title="Avg_0.4",
                              ValidationTable=results)
# print(f"\n{results}\n\n")
# print(f"\nFalse Negative (FN):\n{get_validation_metric(GMT_GS, GMTcalcolata, metrics='FN', id_cols=['GAT', 'SLAT'])[['GAT', 'SLAT', '_merge']]}\n\n")
# print(f"\nFalse Positive (FP):\n{get_validation_metric(GMT_GS, GMTcalcolata, metrics='FP', id_cols=['GAT', 'SLAT'])[['GAT', 'SLAT', '_merge']]}\n\n")

#          MT  TP  FP FN       P       R       F
# Avg_0.4  21  16   5  2  0.7619  0.8889  0.8205

######################################################################################################################################################
# Tentativo 4, Uso MaxSim con soglia 0.4. Mi aspetto 0 FN, ma eventualemente qualche FP in più.
######################################################################################################################################################
GMTcalcolata = global_matching_table(sources=SOURCES,
                                     global_schema=GlobalSchema,
                                     methods={'label': ['LEV', 'JAC'], 'instance': ['SIM']}, functions={'SIM': 'JAC'}, sim_thresholds={'SIM': 0.35},
                                     combiner='MaxSim',
                                     combiner_threshold=0.3,
                                     weights=None,
                                     exec_stable_matching=True,
                                     exec_symmetric_best_match=False,
                                     K=None,
                                     column='A')

# print(f"\n{'-•-' * 20} GMT CALCOLATA {'-•-' * 20}\n\n{to_GMM(GMTcalcolata)}\n\n")
results = complete_validation(GoldStandard=GMT_GS, GlobalMatchingTable=GMTcalcolata, id_cols=['GAT', 'SLAT'], title="Max_0.3",
                              ValidationTable=results)
# print(f"\n{results}\n\n")
# print(f"\nFalse Negative (FN):\n{get_validation_metric(GMT_GS, GMTcalcolata, metrics='FN', id_cols=['GAT', 'SLAT'])[['GAT', 'SLAT', '_merge']]}\n\n")
# print(f"\nFalse Positive (FP):\n{get_validation_metric(GMT_GS, GMTcalcolata, metrics='FP', id_cols=['GAT', 'SLAT'])[['GAT', 'SLAT', '_merge']]}\n\n")

#          MT  TP  FP FN       P       R       F
# Max_0.3  29  18  11  0  0.6207  1.0000  0.7660

######################################################################################################################################################
# Tentativo 5, tento con una grid search tra le soglie.
# N.B. Come si vede il combiner è fisso su Avg_Sim. Questo perché si è notato che i risultati erano sempre gli stessi tra Max, Min ed Avg,
# ciò che cambiava erano i range nei quali si verificano le somiglianze.
######################################################################################################################################################
for combiner in ['Avg_Sim']:  # , 'MinSim', 'MaxSim']:
    for combiner_threshold in np.linspace(0.3, 0.5, 5):
        for sim_threshold in np.linspace(0.4, 0.8, 4):
            GMTcalcolata = global_matching_table(sources=SOURCES,
                                                 global_schema=GlobalSchema,
                                                 methods={'label': ['LEV', 'JAC'], 'instance': ['SIM']}, functions={'SIM': 'JAC'},
                                                 sim_thresholds={'SIM': sim_threshold},
                                                 combiner=combiner,
                                                 combiner_threshold=combiner_threshold,
                                                 weights=None,
                                                 exec_stable_matching=True,
                                                 exec_symmetric_best_match=False,
                                                 K=None,
                                                 column='A')

            results = complete_validation(GoldStandard=GMT_GS, GlobalMatchingTable=GMTcalcolata, id_cols=['GAT', 'SLAT'],
                                          title=f"{combiner}_{combiner_threshold:.2f}_SIM_{sim_threshold:.2f}",
                                          ValidationTable=results)

print(f"\n{'-•-' * 20} RESULTS {'-•-' * 20}\n\n{results}\n\n")

# Risultati:
#                        MT  TP  FP FN       P       R       F
# Avg_Sim_0.30_SIM_0.40  27  18   9  0  0.6667  1.0000  0.8000
# Avg_Sim_0.30_SIM_0.53  27  18   9  0  0.6667  1.0000  0.8000
# Avg_Sim_0.30_SIM_0.67  25  18   7  0  0.7200  1.0000  0.8372
# Avg_Sim_0.30_SIM_0.80  25  18   7  0  0.7200  1.0000  0.8372
# Avg_Sim_0.35_SIM_0.40  25  18   7  0  0.7200  1.0000  0.8372
# Avg_Sim_0.35_SIM_0.53  23  18   5  0  0.7826  1.0000  0.8780
# Avg_Sim_0.35_SIM_0.67  21  18   3  0  0.8571  1.0000  0.9231 # <-- best R (a parità di P)
# Avg_Sim_0.35_SIM_0.80  21  18   3  0  0.8571  1.0000  0.9231 # <-- best R (a parità di P)
# Avg_Sim_0.40_SIM_0.40  19  16   3  2  0.8421  0.8889  0.8649
# Avg_Sim_0.40_SIM_0.53  19  16   3  2  0.8421  0.8889  0.8649
# Avg_Sim_0.40_SIM_0.67  17  15   2  3  0.8824  0.8333  0.8571
# Avg_Sim_0.40_SIM_0.80  17  15   2  3  0.8824  0.8333  0.8571
# Avg_Sim_0.45_SIM_0.40  16  15   1  3  0.9375  0.8333  0.8824
# Avg_Sim_0.45_SIM_0.53  14  14   0  4  1.0000  0.7778  0.8750 # <-- best P
# Avg_Sim_0.45_SIM_0.67  14  14   0  4  1.0000  0.7778  0.8750 # <-- best P
# Avg_Sim_0.45_SIM_0.80  14  14   0  4  1.0000  0.7778  0.8750 # <-- best P
# Avg_Sim_0.50_SIM_0.40  14  14   0  4  1.0000  0.7778  0.8750 # <-- best P
# Avg_Sim_0.50_SIM_0.53  14  14   0  4  1.0000  0.7778  0.8750 # <-- best P
# Avg_Sim_0.50_SIM_0.67  14  14   0  4  1.0000  0.7778  0.8750 # <-- best P
# Avg_Sim_0.50_SIM_0.80  14  14   0  4  1.0000  0.7778  0.8750 # <-- best P

# Le possibili conclusioni/soluzioni sono 2:
# 1) eseguire una grid search sui metodi, in modo da osservare se ci siano metodi più efficaci;
# 2) non è possibile raggiungere un (P,R) = (1,1), ed è dunque possibile prendere come risultati ottimali i 2 casi seguenti (la scelta cade sul caso
# in esame: è necessario dare maggiore importanza alla P o alla R?
#                        MT  TP  FP FN       P       R       F
# Avg_Sim_0.35_SIM_0.80  21  18   3  0  0.8571  1.0000  0.9231
# Avg_Sim_0.45_SIM_0.53  14  14   0  4  1.0000  0.7778  0.8750
