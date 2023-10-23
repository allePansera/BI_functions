# Da Blocking_Matching.py
# https://colab.research.google.com/drive/167jd9hHV6pEqr1JpafxxeIFwoceJLCOy?hl=it#scrollTo=7_EvEWPPmrh8

import pandas as pd
import py_entitymatching as em
import os

current_dir = os.getcwd()  # Get the current directory
output_dir = os.path.join(current_dir, 'csv')
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

######################## ESEMPIO COSMETICS ########################
####################### Testo dell'esercizio ######################

A = pd.read_csv('http://pages.cs.wisc.edu/~anhai/data/wisc_em_benchmark/839_spring19/Cosmetics/csv/table_a.csv')
B = pd.read_csv('http://pages.cs.wisc.edu/~anhai/data/wisc_em_benchmark/839_spring19/Cosmetics/csv/table_b.csv')
GoldStandard = pd.read_csv('http://pages.cs.wisc.edu/~anhai/data/wisc_em_benchmark/839_spring19/Cosmetics/csv/predicted_matches.csv')
CandidateSetDato = pd.read_csv('http://pages.cs.wisc.edu/~anhai/data/wisc_em_benchmark/839_spring19/Cosmetics/csv/candidate_pairs.csv')

# per uniformare i nomi
GoldStandard.columns = ['l_id', 'r_id', 'conf']
CandidateSetDato.columns = ['l_id', 'r_id']
A = A.rename(columns={'_id': 'l_id'})
B = B.rename(columns={'_id': 'r_id'})

# si aggiunge un id a GoldStandard
GoldStandard['id'] = range(len(GoldStandard))
# e si setta come key
em.set_key(GoldStandard, 'id')
# si settano le key dei due dataset
em.set_key(A, 'l_id')
em.set_key(B, 'r_id')
# si settano le foreign key in GS
em.set_ltable(GoldStandard, A)
em.set_rtable(GoldStandard, B)
em.set_fk_ltable(GoldStandard, 'l_id')
em.set_fk_rtable(GoldStandard, 'r_id')

# per applicare il RuleBasedBlocker, generiamo automaticamente le features
block_features = em.get_features_for_blocking(A, B, validate_inferred_attr_types=True)

A["rating"] = A["rating"].astype(str)
B["rating"] = B["rating"].astype(str)
block_c = em.get_attr_corres(A, B)

atypesA = em.get_attr_types(A)
atypesB = em.get_attr_types(B)

atypesA['product name'] = atypesB['product name']
atypesA['rating'] = atypesB['rating']
block_features = em.get_features(A, B,
                                 atypesA, atypesB, block_c,
                                 em.get_tokenizers_for_blocking(),
                                 em.get_sim_funs_for_blocking())
[print(f'{i + 1}) {feature}') for i, feature in enumerate(block_features["feature_name"].to_list())]

# Validation:
#                      A     B  BlockSize       R       C       Q
# CandidateSetDato  3212  3089     162226  0.9836  1.0000  0.0059

rb = em.RuleBasedBlocker()
rb.add_rule(['product_name_product_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.3'], block_features)

C = rb.block_tables(A, B, l_output_attrs=['product name', 'brand', 'price'],
                    r_output_attrs=['product name', 'brand', 'price'], show_progress=True)

# Validation:
#       A     B  BlockSize       R       C       Q
# C  3212  3089      15163  0.9985  0.9633  0.0606

# per analizzare quelli missing MissedMatches: quelli che sono nel GoldStandard e non nel Candidate Set
C = C.rename(columns={'ltable_l_id': 'l_id', 'rtable_r_id': 'r_id'})
MissedMatches = pd.merge(GoldStandard, C, how='left', indicator=True).query("_merge=='left_only'")[['l_id', 'r_id']]
# print(len(MissedMatches))   ->   35
pd.merge(pd.merge(MissedMatches, A), B, on='r_id').to_csv(f'{output_dir}/MissedMatches.csv')

####################### DOMANDE ######################


## DOMANDA 1
# Mi accorgo che molte delle MissedMatches hanno un nome abbastanza diverso
# ma simile brand-price allora al candidate set C devo unire le coppie con simile brand-price:
# come includerle?


## DOMANDA 2
# Altre modalità per effettuare Blocking sulla coppia ['brand', 'price'],
# cioè per confrontare ASSIEME sia brand che price dei due dataset?


## DOMANDA 3
# Come aumentare la qualità? Cosa perdo?


## DOMANDA 4
# Effettuare il blocking tramite similarity join sul seguente attributo mix.
# Poi confrontarne il risultato rispetto al CandidateSetDato
mix = ['brand', 'price', 'product name']

## DOMANDA 5
# Modificare la rule precedente per aumentare la precisione


## DOMANDA 6
# Ha senso applicare in questo caso il Global Matching, cioè limitarsi a fissate cardinalità del maching tra i dataset?
# Tipologia del matching stabilito nel GS? one-to-one, one-to-many or many-to-many ?
# Valuta la cardinalità della Matching Table ottenuta alla domanda precedente (domanda 5) per rispondere alla domanda
print(GoldStandard.groupby('r_id')[['l_id']].count().sort_values('l_id', ascending=False).head(2))
print(GoldStandard.groupby('l_id')[['r_id']].count().sort_values('r_id', ascending=False).head(2))

## DOMANDA 7
# Supponendo che il Gold Standard non sia più disponibile.
# Effettuare il data matching usando il Candidate Set dato.
