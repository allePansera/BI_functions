# Da EsercizioEsameTopDown234.py
# https://colab.research.google.com/drive/1VwpHgiqDfaQku0NK0DrI2idN8voCUj7T?hl=it
import sys
import warnings
from argparse import ArgumentParser

import pandas as pd

from ESERCIZI.utils.funzioni_prof import levenshtein_label_based_similarity, value_overlap_simjoin_jaccard, min_sim_table, max_sim_table, avg_sim_table, thresholding, simmetric_best_match, stable_marriage, ChiusuraRiflessivaTransitiva, CalcoloDeiCluster, fromClusterToGMT, generaLAT, Valuta, toA_B, to_GMM

warnings.filterwarnings('ignore')

parser = ArgumentParser()
parser.add_argument('esercizio', type=int, help="Number of exercise wanted")
args = parser.parse_args()
esercizio = args.esercizio
esercizi = [1, 2, 3]
if esercizio not in esercizi:
    print(f"Esercizio non valido, inserire un esercizio valido: {esercizio} not in {esercizi}")
    sys.exit()
print(f"Executing exercise {esercizio}")
print("=" * 60)
print()


##################################################################
######################### ESERICIZO ESAME #########################
######################## Funzioni esercizio #######################
##################################################################
# Vengono date una serie di funzioni, riportate:
# - nel file utils/funzioni_prof.py (funzioni generali)
# - di seguito  le funzioni con implementazione specifica

def CalcoloGlobalMatchingTable(Sources: list, GlobalSchema: pd.DataFrame,
                               value_overlap_simjoin_jaccardT,
                               combinerMinMaxAvg,
                               thresholdingT,
                               GlobalMapping):
    GlobalMatchingTable = pd.DataFrame(columns=['GAT', 'SOURCE', 'LAT', 'SLAT', 'sim'])
    from tqdm import tqdm
    for y in Sources.keys():
        SimTableA = levenshtein_label_based_similarity(GlobalSchema, Sources[y])
        SimTableC = value_overlap_simjoin_jaccard(GlobalSchema, Sources[y], value_overlap_simjoin_jaccardT)

        # combiner
        if combinerMinMaxAvg == "min":
            SimTable = min_sim_table([SimTableA, SimTableC])
        if combinerMinMaxAvg == "max":
            SimTable = max_sim_table([SimTableA, SimTableC])
        if combinerMinMaxAvg == "avg":
            SimTable = avg_sim_table([SimTableA, SimTableC])
        if combinerMinMaxAvg == "weighed":
            SimTableA = SimTableA.rename(columns={'sim': 'sim_A'})
            SimTableC = SimTableC.rename(columns={'sim': 'sim_C'})
            SimTable = pd.merge(SimTableA, SimTableC)
            SimTable['sim'] = 0.4 * SimTable['sim_A'] + 0.6 * SimTable['sim_C']  # **** cambiare i pesi ****
            SimTable = SimTable.drop(columns=["sim_A", 'sim_C'])

        MatchTable = thresholding(SimTable, thresholdingT)

        # MatchTable = top_K(table,2)
        # MatchTable = top_K(MatchTable,2,'A')
        # MatchTable = top_K(MatchTable,2,'B')

        # global mapping
        if GlobalMapping == "SM":
            MatchTable = stable_marriage(MatchTable)
        if GlobalMapping == "SBM":
            MatchTable = simmetric_best_match(MatchTable)

        MatchTable.columns = ['GAT', 'LAT', 'sim']
        MatchTable['SOURCE'] = str(y)
        MatchTable['SLAT'] = MatchTable['SOURCE'] + '_' + MatchTable['LAT']
        GlobalMatchingTable = GlobalMatchingTable.append(MatchTable, sort=False)
    return GlobalMatchingTable


def CalcoloLocalMatchingTable(Sources: list,
                              value_overlap_simjoin_jaccardT,
                              combinerMinMaxAvg,
                              thresholdingT,
                              GlobalMapping):

    LocalMatchingTable = pd.DataFrame(columns=['SOURCE_A', 'LAT_A', 'SOURCE_B', 'LAT_B', 'SLAT_A', 'SLAT_B', 'sim'])

    for x in Sources.keys():
        for y in Sources.keys():
            if x <= y:
                SimTableA = levenshtein_label_based_similarity(Sources[x], Sources[y])
                # SimTableB = jaro_label_based_similarity(Sources[x], Sources[y])
                SimTableC = value_overlap_simjoin_jaccard(Sources[x], Sources[y], value_overlap_simjoin_jaccardT)

                # combiner
                if combinerMinMaxAvg == "min":
                    SimTable = min_sim_table([SimTableA, SimTableC])
                if combinerMinMaxAvg == "max":
                    SimTable = max_sim_table([SimTableA, SimTableC])
                if combinerMinMaxAvg == "avg":
                    SimTable = avg_sim_table([SimTableA, SimTableC])
                if combinerMinMaxAvg == "weighed":
                    SimTableA = SimTableA.rename(columns={'sim': 'sim_A'})
                    SimTableC = SimTableC.rename(columns={'sim': 'sim_C'})
                    SimTable = pd.merge(SimTableA, SimTableC)
                    SimTable['sim'] = 0.4 * SimTable['sim_A'] + 0.6 * SimTable['sim_C']
                    SimTable = SimTable.drop(columns=["sim_A", 'sim_C'])

                MatchTable = thresholding(SimTable, thresholdingT)

                # global mapping
                if GlobalMapping == "SM":
                    MatchTable = stable_marriage(MatchTable.query("A!=B"))
                if GlobalMapping == "SBM":
                    MatchTable = simmetric_best_match(MatchTable.query("A!=B"))

                LocalMatchingTableXY = MatchTable
                LocalMatchingTableXY.columns = ['LAT_A', 'LAT_B', 'sim']
                LocalMatchingTableXY['SOURCE_A'] = x
                LocalMatchingTableXY['SOURCE_B'] = y
                LocalMatchingTableXY['SLAT_A'] = LocalMatchingTableXY['SOURCE_A'] + '_' + LocalMatchingTableXY['LAT_A']
                LocalMatchingTableXY['SLAT_B'] = LocalMatchingTableXY['SOURCE_B'] + '_' + LocalMatchingTableXY['LAT_B']
                LocalMatchingTable = LocalMatchingTable.append(LocalMatchingTableXY, sort=True)
    return LocalMatchingTable


def SchemaIntegration(Sources,
                      value_overlap_simjoin_jaccardT,
                      combinerMinMaxAvg,
                      thresholdingT,
                      GlobalMapping):
    LMT = CalcoloLocalMatchingTable(Sources,
                                    value_overlap_simjoin_jaccardT,
                                    combinerMinMaxAvg,
                                    thresholdingT,
                                    GlobalMapping)
    CRT = ChiusuraRiflessivaTransitiva(LMT[['SLAT_A', 'SLAT_B']],
                                       generaLAT(Sources)).drop_duplicates()

    Cluster = CalcoloDeiCluster(CRT)
    GMT = fromClusterToGMT(Cluster)

    return GMT


# ------------------------------------------------------------------------------------------------------------------------------------------


##################################################################
######################## Testo esercizio 1 #######################
##################################################################

if esercizio == 1:
    # Consideriamo le seguenti Sources
    SOURCES = {}
    DF = pd.DataFrame({'AAAAAAAAAAA': ['a', 'b', 'c', 'd', 'e'],
                       'BBBBBBBBBBB': ['f', 'g', 'h', 'i', 'l']})
    SOURCES['S1'] = DF
    DF = pd.DataFrame({'AAAAA': ['f', 'g', 'h', 'z']})
    SOURCES['S2'] = DF

    # Schema Globale Dato istanziato
    GlobalSchema = pd.DataFrame({'AAAAAAAAAAA': ['a', 'b', 'c', 'd', 'e'],
                                 'BBBBBBBBB': ['f', 'g', 'h', 'i', 'l'],
                                 'CCCCCCC': ['f', 'g', 'h', 'd', 'e']})
    print(f"\nGMT con jaccard_threshold = 0.1, maxCombiner, thresholding = 0.1\n"
          f"{CalcoloGlobalMatchingTable(SOURCES, GlobalSchema, 0.1, 'max', 0.1, 'NO')}")
    # se alzo soglia a 0.4
    print(f"\nGMT con jaccard_threshold = 0.1, maxCombiner, thresholding = 0.5\n"
          f"{CalcoloGlobalMatchingTable(SOURCES, GlobalSchema, 0.1, 'max', 0.4, 'NO')}")
    # se alzo soglia a 0.5
    print(f"\nGMT con jaccard_threshold = 0.1, maxCombiner, thresholding = 0.5\n"
          f"{CalcoloGlobalMatchingTable(SOURCES, GlobalSchema, 0.1, 'max', 0.5, 'NO')}")
    # se utilizzo min
    print(f"\nGMT con jaccard_threshold = 0.1, minCombiner, thresholding = 0.1\n"
          f"{CalcoloGlobalMatchingTable(SOURCES, GlobalSchema, 0.1, 'min', 0.1, 'NO')}")

    ####################### Domande esercizio 1 ######################
    print(f"\n\n{'- • ' * 10}- ESECUZIONE RISPOSTE {'- • ' * 10}-\n\n")

    # Considerando:
    GMT = CalcoloGlobalMatchingTable(SOURCES, GlobalSchema, 0.1, 'max', 0.1, 'NO')
    # TODO DOMANDA 1: Come posso ottenere un mapping 1-N, in cui un attributo locale sia mappato in un solo (il migliore) attributo globale?
    # RISPOSTA 1 - Alessandro
    # E' sufficiente impiegare come tecnica di mapping globale lo stable marriage al contrario del symmetric best match
    # TODO DOMANDA 2: Come posso ottenere un mapping N-N, in cui un attributo locale sia mappato in massimo due (i migliori) attributo globale?
    # RISPOSTA 2 - Alessandro
    # Posso impiegare la top-k strategy con k = 2.

# ------------------------------------------------------------------------------------------------------------------------------------------

##################################################################
######################## Testo esercizio 2 #######################
##################################################################
elif esercizio == 2:
    # consideriamo il seguente schema
    SOURCES = {}
    DF = pd.DataFrame({'AAAAAAAAAAA': ['a', 'b', 'c', 'd', 'e'],
                       'BBBBBBBBBBB': ['f', 'g', 'h', 'i', 'l']})
    SOURCES['S1'] = DF
    DF = pd.DataFrame({'AAAAAAAA': ['f', 'g', 'h', 'z']})
    SOURCES['S2'] = DF

    print(f"\nGMT ottenuta tramite\nSchemaIntegration con jaccard_threshold = 0.1, minCombiner, thresholding = 0.1\n"
          f"{SchemaIntegration(SOURCES, 0.1, 'max', 0.1, 'NO')}")

    # Vediamo il processo nello specifico
    print(f"\n\n{'-' * 50}\nVediamo il processo di schema integration passo passo\n\n")

    LMT = CalcoloLocalMatchingTable(SOURCES, 0.1, 'max', 0.1, 'NO')
    print(f"LMT con jaccard_threshold = 0.1, minCombiner, thresholding = 0.1\n"
          f"{LMT}")

    CRT = ChiusuraRiflessivaTransitiva(LMT[['SLAT_A', 'SLAT_B']], generaLAT(SOURCES)).drop_duplicates()
    print(f"\nChiusuraRiflessivaTransitiva su LMT\n"
          f"{CRT}")

    Cluster = CalcoloDeiCluster(CRT)
    print(f"\nCluster ottenuti sulla chiusura\n"
          f"{Cluster}")

    GMT = fromClusterToGMT(Cluster)
    print(f"\nGMT ottenuta\n"
          f"{GMT}")

    ####################### Domande esercizio 2 ######################
    print(f"\n\n{'- • ' * 10}- ESECUZIONE RISPOSTE {'- • ' * 10}-\n\n")

    # Consideriamo
    SchemaIntegration(SOURCES, 0.1, 'max', 0.1, 'NO')

    # TODO DOMANDA 1: Cosa cambia se vario la soglia value_overlap_simjoin_jaccardT portandola a 0.9
    # RISPOSTA 1 - Alessandro
    # A livello teorico quello che accade è che il numero di match dovrebbe risursi in quanto sale la soglia del coefficiente di similarità
    # TODO DOMANDA 2: Perché è rimasto tutto invariato?
    # Usando come combiner il max con soglia a 0.1 i risultati migliori saranno sempre gli stessi che avevamo prima
    # TODO DOMANDA 3: Perché viene usata per decidere se un valore è uguale ad un altro, e nel nostro caso sono tutti singolo caratteri


# ------------------------------------------------------------------------------------------------------------------------------------------

##################################################################
######################## SIMULAZIONE ESAME #######################
######################## Testo esercizio 3 #######################
##################################################################
elif esercizio == 3:
    # Sorgenti, Global Schema e Gold Standard
    path = "https://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/"

    src_links = [
        path + 'CameraS1_B.csv',
        path + 'CameraS2_B.csv',
        path + 'CameraS3_B.csv',
        path + 'CameraS4_B.csv']

    SOURCES = {'S' + str(i + 1): pd.read_csv(link).astype(str) for i, link in enumerate(src_links)}

    GlobalSchema = pd.read_csv(path + "GlobalSchemaTopDown23.csv").astype(str)
    GoldStandard = pd.read_csv(path + "EE_GoldStandardTopDown234.csv").astype(str)
    GlobalSchemaGAT = pd.DataFrame({"GAT": GlobalSchema.columns})

    # Verifichiamo che gli attributi globali usati nel gold standard siano gli stessi di quelli del Global Schema
    FOJ = pd.merge(GoldStandard, GlobalSchemaGAT, on='GAT', how='outer', indicator=True)
    print(f"Stessi attributi in GoldStandard e in GlobalSchema = {len(FOJ[FOJ._merge != 'both']) == 0}")

    # Verifichiamo che gli attributi locali usati nel gold standard siano gli stessi di quelli delle SOURCES
    FOJ = pd.merge(GoldStandard, generaLAT(SOURCES), on='SLAT', how='outer', indicator=True)
    print(f"Stessi attributi in GoldStandard e nelle SOURCES = {len(FOJ[FOJ._merge != 'both']) == 0}")

    # CONSIDERARE il dataframe per memorizzare le valutazioni dei vari metodi
    ValutazioneMatchTable = pd.DataFrame(columns=['MT', 'TP', 'FP', 'FN', 'P', 'R', 'F'])
    # Partendo dal seguente esempio
    GMT = CalcoloGlobalMatchingTable(SOURCES, GlobalSchema, 0.2, 'max', 0.5, 'NO')
    ValutazioneMatchTable = ValutazioneMatchTable.append(
        Valuta(toA_B(GoldStandard),
               toA_B(GMT))).rename(index={0: " 0.2,'max',0.5,'NO' "})
    print(f"\nValidation della Match Table\n{ValutazioneMatchTable}")

    ####################### Domande esercizio 3 ######################
    print(f"\n\n{'- • ' * 10}- ESECUZIONE RISPOSTE {'- • ' * 10}-\n\n")

    # TODO DOMANDA 1:
    #  Modificare il processo di schema matching (sia parametri che struttura di CalcoloGlobalMatchingTable) per migliorare:
    #  - Precision
    #  - Recall
    # IN QUESTO CASO E' SUFFICIENTE PROCEDERE PER TENTTIVI ANDANDO AD AGGIUNGERE DELLE MISURE OPPURE
    # VARIARE IL TIPO DI MISURAZIONE DA CONSIDERARE NEL COMBINER - > MIN/MAX/AVG

    # TODO DOMANDA 2:
    #  Considerare la seguente Mapping Matching Table data MTdata_TopDown2.
    #  1) Stabilire il tipo di mapping che si ha con tale MTdata_TopDown2:**
    #       1-1: in ogni sorgente i, un local attribute di LSi può essere mappato in 1 global attribute di GS, e viceversa
    #       N-1 : in ogni sorgente i, un local attribute di LSi può essere mappato in n global attribute di GS, MA NON viceversa
    #       1-N, in ogni sorgente i, un global attribute di GS può essere mappato in n local attribute di LSi, MA NON viceversa
    #       N-N: in ogni sorgente i, un local attribute di LSi può essere mappato in N global attribute di GS, E viceversa
    #  2) Modificare il processo di schema matching ottenuto in precedenza (sia parametri che struttura di CalcoloGlobalMatchingTable)
    #  per ottenere una Global Mapping Matrix con le stesse caratteristiche;
    #  ad esempio se il mapping stabilito da MTdata_TopDown2 è *1-N si dovrebbe ottenere lo stesso tipo di mapping.
    #  Non è necessario effettuare la valutazione di precision e recall.
    # ABBIAMO UNA N-N, PER MODIFICARE LA FUNZIONE PRECEDENTE BASTA USARE UNA TOP-K COME METODO DI MAPPING PER GENERARE LE CORRISPONDENZE

    MTdata_TopDown234 = pd.read_csv(path + 'MTdata_TopDown234.csv').astype(str)
    # print(to_GMM(MTdata_TopDown234)) -> CORREGGERE PERCHÉ DA ERRORE

    # Verifichiamo che gli attributi globali usati in MTdata_TopDown234 siano gli stessi di quelli del Global Schema
    FOJ = pd.merge(MTdata_TopDown234, GlobalSchemaGAT, on='GAT', how='outer', indicator=True)
    print(f"Stessi attributi in MTdata_TopDown234 e in GlobalSchema = {len(FOJ[FOJ._merge != 'both']) == 0}")

    # Verifichiamo che gli attributi locali usati nel MTdata_TopDown234 siano gli stessi di quelli delle SOURCES
    FOJ = pd.merge(MTdata_TopDown234, generaLAT(SOURCES), on='SLAT', how='outer', indicator=True)
    print(f"Stessi attributi in MTdata_TopDown234 e nelle SOURCES = {len(FOJ[FOJ._merge != 'both']) == 0}")
