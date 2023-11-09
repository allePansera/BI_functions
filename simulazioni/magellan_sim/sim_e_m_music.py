from Blocking.EntityMatching import EntityMatching
from Evaluation.Evaluation import vedi_valuta, valuta
from ESERCIZI.utils.funzioni_prof import _VisualizzaDistribuzioneCluster, VisualizzaCluster
import pandas as pd, csv


path='C:\\Users\\allep\\Desktop\\studi\\Università\\esami\\20232024\\BI\\repo_db\\music\\csv_files\\'

src_links = [
path+'itunes.csv',
path+'amazon_music.csv']

cand_set_path = path + "candset.csv"
labeled_data_path = path + "labeled_data.csv"


SOURCES = { 'S'+str(i+1) : pd.read_csv(link, encoding="ISO-8859-1").astype(str) for i, link in enumerate(src_links) }

# Il candidate set me lo costruisco io, ci sono dei bug dentro al file
with open(cand_set_path, encoding="ISO-8859-1") as cs_f:
    delimiter = ','
    csvreader = csv.reader(cs_f, delimiter=delimiter)
    header = []
    header = next(csvreader)
    data_struct = {"_id": [], "l_id": [], "r_id": []}
    for row in csvreader:
        row = row[0].split(delimiter)
        if len(row) > 4:
            data_struct["_id"].append(row[1])
            data_struct["l_id"].append(row[2])
            data_struct["r_id"].append(row[3])

    CAND_SET = pd.DataFrame(data_struct)

# Costruisco il mio GS con i soli campi di interesse
# Il candidate set me lo costruisco io, ci sono dei bug dentro al file
with open(labeled_data_path, encoding="ISO-8859-1") as cs_f:
    delimiter = ','
    csvreader = csv.reader(cs_f, delimiter=delimiter)
    header = []
    header = next(csvreader)
    data_struct = {"_id": [], "l_id": [], "r_id": []}
    for row in csvreader:
        row = row[0].split(delimiter)
        if len(row) > 4:
            data_struct["_id"].append(row[1])
            data_struct["l_id"].append(row[2])
            data_struct["r_id"].append(row[3])

    LABELLED_DATA_SET = pd.DataFrame(data_struct)

GoldStandard = LABELLED_DATA_SET
# rinominiamo in id il campo 'Sno'
for x in SOURCES.keys():
    SOURCES[x] = SOURCES[x].rename(columns={"Sno": "id"})

# rimuovo le colonne non comuni, non sono dati necessari per identificare i report.
SOURCES['S2'] = SOURCES['S2'].drop(['Label', 'Copyright'], axis=1)
SOURCES['S1'] = SOURCES['S1'].drop(['Customer_Rating', 'Album_Price', 'CopyRight'], axis=1)

# setto per tutte le tabelle l'id a stringa in modo da non avere problemi in seguito con le add_prefix
SOURCES['S1'] = SOURCES['S1'].astype({'id': str})
SOURCES['S2'] = SOURCES['S2'].astype({'id': str})
GoldStandard = GoldStandard.astype({'l_id': str})
GoldStandard = GoldStandard.astype({'r_id': str})
GoldStandard = GoldStandard.astype({'_id': str})
CAND_SET = CAND_SET.astype({'l_id': str})
CAND_SET = CAND_SET.astype({'l_id': str})
CAND_SET = CAND_SET.astype({'_id': str})
# ri-definisco gli id all'interno della left table (S1) e right table (S2)
SOURCES['S1']['id'] = "S1_"+SOURCES['S1']['id'].astype(str)
SOURCES['S2']['id'] = "S2_"+SOURCES['S2']['id'].astype(str)
# aggiungo i prefissi al candidate e anche al gold standard affinché possa riutilizzare il sistema
GoldStandard['l_id'] = "S1_"+GoldStandard['l_id'].astype(str)
GoldStandard['r_id'] = "S2_"+GoldStandard['r_id'].astype(str)
CAND_SET['l_id'] = "S1_"+CAND_SET['l_id'].astype(str)
CAND_SET['r_id'] = "S2_"+CAND_SET['r_id'].astype(str)
# Creo una var. joint con NomeArt/NomeAlb/NomeCanzone
for s in SOURCES:
    # chiamo mix la variabile che andrò a creare su entrambe le basi di dati
    SOURCES[s]['Song_Name'] = SOURCES[s]['Song_Name'].str.replace(r"\(.*?\)", "", regex=True)
    SOURCES[s]['Song_Name'] = SOURCES[s]['Song_Name'].str.replace(r"\[.*?\]", "", regex=True)
    SOURCES[s]['mix'] = SOURCES[s]['Album_Name']+' '+SOURCES[s]['Artist_Name']+' '+SOURCES[s]['Song_Name']
    SOURCES[s]['AlbumName'] = SOURCES[s]['Album_Name']
    SOURCES[s]['ArtistName'] = SOURCES[s]['Artist_Name']
    SOURCES[s]['SongName'] = SOURCES[s]['Song_Name']

# query per analizzare i record con SciView
# SOURCES['S2'].loc[SOURCES['S2']['Sno'] == X]

# come prima cosa calcolo il cluster del golden standard
cluster_gold_standard=EntityMatching.cluster_componenti_connessi(GoldStandard[['l_id','r_id']],
                                                                 EntityMatching.id_sources(SOURCES))

# visualizzare distribuzione del cluster -> printa in output
_VisualizzaDistribuzioneCluster(cluster_gold_standard)

# visualizzazione dei cluster
grouped_cluster = VisualizzaCluster(cluster_gold_standard)

method = "join_blocker"
# method = "rule_based_diff_type"
blocking_keys = ["mix"]
# blocking_keys = []
blocking_rules = []
# blocking_rules = [
#    [{"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) < {}", "attr": "fullname",
#      "score": "0.2"}]]
# matching rules
matching_rules = [
        [{"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) >= {}", "attr": "SongName", "score": "0.2"},
         # vorrei mettere l'exm ma non è disponibile (secondo me perchè ci sono dei valori nulli)
         {"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) >= {}", "attr": "ArtistName", "score": "0.5"}]
]

# l attrs to exclude
omit_l_attrs = ["l_id"]
# r attrs to exclude
omit_r_attrs = ["r_id"]
# matching method
matching_method = "SYMM" # STAB # SYMM

em = EntityMatching(SOURCES)
entity_match_table = em.process(method, blocking_keys, omit_l_attrs, omit_r_attrs,
                                blocking_rules, matching_rules, matching_method)

# query per verifica del matching all'interno della match table
CorrespMT = entity_match_table.groupby('r_id')[['l_id']].count().sort_values('l_id', ascending=False)


cluster_entity_matching=EntityMatching.cluster_componenti_connessi(entity_match_table[['l_id','r_id']],
                                                                   EntityMatching.id_sources(SOURCES))

# da controllare per verificare Precision e Recall
Valuta = valuta(EntityMatching.calcola_match_indotti_cluster(cluster_gold_standard).rename(columns={"l_id": "A", "r_id": "B"}),
                     EntityMatching.calcola_match_indotti_cluster(cluster_entity_matching).rename(columns={"l_id": "A", "r_id": "B"}))

# da controllare quando si hanno score troppo bassi nel matching
VediValuta_FN = vedi_valuta(EntityMatching.calcola_match_indotti_cluster(cluster_gold_standard).rename(columns={"l_id": "A", "r_id": "B"}),
                     EntityMatching.calcola_match_indotti_cluster(cluster_entity_matching).rename(columns={"l_id": "A", "r_id": "B"}),
                    "FN")
VediValuta_FP = vedi_valuta(EntityMatching.calcola_match_indotti_cluster(cluster_gold_standard).rename(columns={"l_id": "A", "r_id": "B"}),
                     EntityMatching.calcola_match_indotti_cluster(cluster_entity_matching).rename(columns={"l_id": "A", "r_id": "B"}),
                    "FP")
# da controllare per verificare il numero di elementi per ogni cluster, può essere comodo per valutare il mapping
ClusterGrouping = cluster_entity_matching.groupby('ClusterKey').apply(EntityMatching.aggregazione_cluster).reset_index().sort_values('#Elements',
                                                                                                 ascending=False)

# visualizzare distribuzione del cluster
_VisualizzaDistribuzioneCluster(cluster_entity_matching)

# visualizzazione dei cluster
grouped_cluster = VisualizzaCluster(cluster_entity_matching)


Unione = pd.DataFrame(columns=SOURCES['S1'].columns)
for x in SOURCES.keys():
    Unione = Unione.append(SOURCES[x])
vedi_valuta_detail_FP = pd.merge(pd.merge(VediValuta_FP.rename(columns={"A": "l_id", "B": "r_id"}),
                                       Unione,
                                       left_on='l_id', right_on='id'),
                              Unione, left_on='r_id', right_on='id')
vedi_valuta_detail_FP = pd.merge(pd.merge(VediValuta_FN.rename(columns={"A": "l_id", "B": "r_id"}),
                                       Unione,
                                       left_on='l_id', right_on='id'),
                              Unione, left_on='r_id', right_on='id')

exit(1)



