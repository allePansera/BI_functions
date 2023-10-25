# Da BlockingMatching_Clustering.ipynb
# https://colab.research.google.com/drive/1VwpHgiqDfaQku0NK0DrI2idN8voCUj7T?hl=it
import sys
import warnings
from argparse import ArgumentParser
from copy import deepcopy

import pandas as pd
import os
import py_entitymatching as em
import py_stringmatching as sm
import py_stringsimjoin as ssj

from examples.ESERCIZI.utils.funzioni_prof import ClusterComponentiConnessi, IdSOURCES, Aggregazione, stable_marriage, _VisualizzaDistribuzioneCluster, CalcolaMatchIndottiCluster, Valuta

warnings.filterwarnings('ignore')

current_dir = os.getcwd()  # Get the current directory
output_dir = os.path.join(current_dir, 'csv')
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

parser = ArgumentParser()
parser.add_argument('esercizio', type=int, help="Number of exercise wanted")
args = parser.parse_args()
esercizio = args.esercizio

esercizi = [1, 2, 3]

if esercizio not in esercizi:
    print(f"Esercizio non valido, inserire un esercizio valido: {esercizio} not in {esercizi}")
    sys.exit()
print(f"Executing exercise {esercizio}")
print(f'{"=" * 60}\n')


##################################################################
######################## Funzioni esercizio #######################
##################################################################
# Vengono date una serie di funzioni, riportate:
# - nel file utils/funzioni_prof.py (funzioni generali)
# - di seguito  le funzioni con implementazione specifica

def MatchTableSOURCES(Sources: list):
    MatchTable = pd.DataFrame(columns=['l_id', 'r_id', 'sim'])

    for x in Sources.keys():
        for y in Sources.keys():
            if (x < y):  # x<=y nel caso dirty

                MTxy = BlockingMatchingRule(Sources[x], Sources[y])

                # global mapping
                MTxy = stable_marriage(MTxy.query("l_id!=r_id"), 'l_id', 'r_id')
                # MTxy = simmetric_best_match(MTxy.query("l_id!=r_id"), 'l_id', 'r_id')

                MatchTable = MatchTable.append(MTxy[['l_id', 'r_id', 'sim']], sort=True)
    return MatchTable


def BlockingMatchingRule(A, B):
    A = deepcopy(A)
    B = deepcopy(B)
    A = A.rename(columns={'id': "l_id"})
    B = B.rename(columns={'id': "r_id"})

    em.set_key(A, 'l_id')
    em.set_key(B, 'r_id')

    # BLOCKING
    A['mix'] = A['first_name'] + ' ' + A['last_name']
    B['mix'] = B['first_name'] + ' ' + B['last_name']

    C_SimJoin_mix1 = ssj.jaccard_join(A, B, 'l_id', 'r_id',
                                      'mix', 'mix', sm.QgramTokenizer(qval=3), threshold=0.2,
                                      l_out_attrs=['first_name', 'last_name', 'sex', 'age', 'birth_place', 'zip_code'],
                                      r_out_attrs=['first_name', 'last_name', 'sex', 'age', 'birth_place', 'zip_code']
                                      )
    C_SimJoin_mix1 = C_SimJoin_mix1.rename(columns={'l_l_id': 'l_id'})
    C_SimJoin_mix1 = C_SimJoin_mix1.rename(columns={'r_r_id': 'r_id'})
    C_SimJoin_mix1 = C_SimJoin_mix1.rename(columns={'_sim_score': 'sim'})
    em.set_key(C_SimJoin_mix1, '_id')
    em.set_ltable(C_SimJoin_mix1, A)
    em.set_rtable(C_SimJoin_mix1, B)
    em.set_fk_ltable(C_SimJoin_mix1, 'l_id')
    em.set_fk_rtable(C_SimJoin_mix1, 'r_id')

    brm = em.BooleanRuleMatcher()
    brm.add_rule(['last_name_last_name_lev_sim(ltuple, rtuple) >= .7', 'zip_code_zip_code_exm(ltuple, rtuple) == 1'], FixedFeatures)
    brm.add_rule(['last_name_last_name_lev_sim(ltuple, rtuple) >= .3',
                  'first_name_first_name_lev_sim(ltuple, rtuple) >= .3'], FixedFeatures)
    predictions = brm.predict(C_SimJoin_mix1, target_attr='pred_label', append=True)
    MT = predictions[predictions.pred_label == 1]

    return MT


if esercizio == 1:
    ##################################################################
    ######################## Testo esercizio 1 #######################
    ##################################################################
    path = 'http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/NCVR2/'

    src_links = [
        path + 'NCVR_AF_clean.csv',
        path + 'NCVR_BF_clean.csv',
        path + 'NCVR_CF_clean.csv']

    SOURCES = {'S' + str(i + 1): pd.read_csv(link).astype(str) for i, link in enumerate(src_links)}

    GoldStandardCLEAN = pd.read_csv(path + "GoldStandardCLEAN2.csv")

    # Consideriamo la loro unione nel dataframe UNIONE il cui schema sarà quello di una sorgente (hanno tutti lo stesso schema)
    UNIONE = pd.DataFrame(columns=SOURCES['S2'].columns)
    # quindi effettuo unione tramite append
    for x in SOURCES.keys():
        UNIONE = UNIONE.append(SOURCES[x])

    # Con il seguente codice si fissano e si verificano le features nel matching
    A = UNIONE
    B = UNIONE
    A = A.rename(columns={'id': "l_id"})
    B = B.rename(columns={'id': "r_id"})
    em.set_key(A, 'l_id')
    em.set_key(B, 'r_id')
    F = em.get_features_for_matching(A, B, validate_inferred_attr_types=False)
    FeaturesList = F['feature_name'].to_list()
    print("Si fissano e si verificano le features nel matching:")
    [print(f'{i + 1}) {feature}') for i, feature in enumerate(FeaturesList)]

    # Features da considerare nel matching
    FixedFeatures = F[F.feature_name.isin(['last_name_last_name_lev_sim',
                                           'zip_code_zip_code_exm',
                                           'first_name_first_name_lev_sim'])]

    ####################### Domande esercizio 1 ######################
    print(f"\n\n{'- • ' * 10}- ESECUZIONE RISPOSTE {'- • ' * 10}-\n\n")

    # TODO DOMANDA 1: Partendo (si consiglia di guardare il notebook di riferimento):
    #  https://colab.research.google.com/drive/1j7yD-uzHKV3K_vxvcl7q9bv9Qo9Zaflx#scrollTo=4Afuqnd2XRGh
    #       - dalla regola data nella funzione BlockingMatchingRule
    #       - dalla visualizzazione dei relativi cluster
    #       - dalla valutazione dei relativi cluster
    #  fare e commentare opportune modifiche alla funzione BlockingMatchingRule per migliorare:
    #       - precizione
    #       - recall
    #  Si possono considerare solo le FixedFeatures o eventualmente anche le altre disponibili

elif esercizio == 2:
    ##################################################################
    ######################## Testo esercizio 1 #######################
    ##################################################################
    path = 'http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/NCVR2/'

    src_links = [
        path + 'NCVR_AF.csv',
        path + 'NCVR_BF.csv',
        path + 'NCVR_CF.csv']

    SOURCES = {'S' + str(i + 1): pd.read_csv(link).astype(str) for i, link in enumerate(src_links)}
    GoldStandardDIRTY = pd.read_csv(path + "GoldStandardDIRTY2.csv")

    # Consideriamo la loro unione nel dataframe UNIONE
    # il cui schema sarà quello di una sorgente (hanno tutti lo stesso schema)
    UNIONE = pd.DataFrame(columns=SOURCES['S2'].columns)
    # quindi effettuo unione tramite append
    for x in SOURCES.keys():
        UNIONE = UNIONE.append(SOURCES[x])

    ClusterGoldStandardDIRTY = ClusterComponentiConnessi(GoldStandardDIRTY[['l_id', 'r_id']],
                                                         IdSOURCES(SOURCES))

    ####################### Domande esercizio 1 ######################

    # TODO DOMANDA 1:
    #  Confrontare quanto fatto in precedenza nel caso clean con quello che si ottiene nel caso dirty: ripetere le stesse funzioni e fare opportune considerazioni

elif esercizio == 3:
    ##################################################################
    ######################## Testo esercizio 1 #######################
    ##################################################################

    # DATASET CLEAN
    path = 'http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/PPRL/'

    src_links = [
        path + 'cleanDatasetA3.csv',
        path + 'cleanDatasetB3.csv',
        path + 'cleanDatasetC3.csv']

    SOURCES = {'S' + str(i + 1): pd.read_csv(link).astype(str) for i, link in enumerate(src_links)}

    GoldStandard = pd.read_csv(path + 'cleanGoldStandard3.csv')

    GoldStandard.columns = ['l_id', 'r_id']
    ClusterGoldStandardCLEAN = ClusterComponentiConnessi(GoldStandard[['l_id', 'r_id']],
                                                         IdSOURCES(SOURCES))

    print(_VisualizzaDistribuzioneCluster(ClusterGoldStandardCLEAN))
    # si effettua l'unione di tutte le sources (ricordiamo che hanno tutte le stesso schema)
    UNIONE = pd.DataFrame(columns=SOURCES['S2'].columns)
    for x in SOURCES.keys():
        UNIONE = UNIONE.append(SOURCES[x])
    # si generano le features tra Unione e se stessa
    A = UNIONE
    B = UNIONE
    A = A.rename(columns={'id': "l_id"})
    B = B.rename(columns={'id': "r_id"})

    em.set_key(A, 'l_id')
    em.set_key(B, 'r_id')

    F = em.get_features_for_matching(A, B, validate_inferred_attr_types=False)
    FeaturesList = F['feature_name'].to_list()
    print("Si fissano e si verificano le features nel matching:")
    [print(f'{i + 1}) {feature}') for i, feature in enumerate(FeaturesList)]
    # Features da considerare nel matching
    FixedFeatures = F[F.feature_name.isin(['surname_surname_jac_qgm_3_qgm_3',
                                           'given_name_given_name_jwn',
                                           'date_of_birth_date_of_birth_jac_qgm_3_qgm_3'])]


    ####################### Domande esercizio 1 ######################

    # TODO DOMANDA 1:
    #  Partendo dalla seguente regola data fare opportune modifiche per migliorare:
    #  - precizione
    #  - recall
    #  considerando solo le FixedFeatures

    def BlockingMatchinRule(A, B):
        A = A.rename(columns={'id': "l_id"})
        B = B.rename(columns={'id': "r_id"})

        em.set_key(A, 'l_id')
        em.set_key(B, 'r_id')

        A['mix'] = A['given_name'] + ' ' + A['surname']
        B['mix'] = B['given_name'] + ' ' + B['surname']

        C_SimJoin_mix = ssj.jaccard_join(A, B, 'l_id', 'r_id',
                                         'mix', 'mix', sm.QgramTokenizer(qval=3), threshold=0.3,
                                         l_out_attrs=['given_name', 'surname', 'date_of_birth'],
                                         r_out_attrs=['given_name', 'surname', 'date_of_birth'])
        C_SimJoin_mix = C_SimJoin_mix.rename(columns={'l_l_id': 'l_id'})
        C_SimJoin_mix = C_SimJoin_mix.rename(columns={'r_r_id': 'r_id'})
        C_SimJoin_mix = C_SimJoin_mix.rename(columns={'_sim_score': 'sim'})

        em.set_key(C_SimJoin_mix, '_id')
        em.set_ltable(C_SimJoin_mix, A)
        em.set_rtable(C_SimJoin_mix, B)
        em.set_fk_ltable(C_SimJoin_mix, 'l_id')
        em.set_fk_rtable(C_SimJoin_mix, 'r_id')

        MT = C_SimJoin_mix  # In questo modo se non applico nessuna rule matching,
        #                   il risultato dell'Entity Resolution è il Candidate Set

        # F = em.get_features_for_matching(A, B, validate_inferred_attr_types=False)
        # viene escluso perchè consideriamo solo le FixedFeatures

        brm = em.BooleanRuleMatcher()
        brm.add_rule(['surname_surname_jac_qgm_3_qgm_3(ltuple, rtuple) >= 0.3'], FixedFeatures)
        predictions = brm.predict(C_SimJoin_mix, target_attr='pred_label', append=True)
        MT = predictions[predictions.pred_label == 1]

        return MT


    def MatchTableSOURCES(Sources: list):
        MatchTable = pd.DataFrame(columns=['l_id', 'r_id', 'sim'])

        for x in Sources.keys():
            for y in Sources.keys():
                if (x < y):  # x<=y nel caso dirty

                    MTxy = BlockingMatchinRule(Sources[x], Sources[y])

                    # global mapping
                    # MTxy = stable_marriage(MTxy.query("l_id!=r_id"))
                    # MTxy = simmetric_best_match(MTxy.query("l_id!=r_id"))

                    MatchTable = MatchTable.append(MTxy[['l_id', 'r_id', 'sim']], sort=True)
        return MatchTable
