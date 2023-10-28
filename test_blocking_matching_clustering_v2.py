from Blocking.EntityMatching import EntityMatching
from Evaluation.Evaluation import vedi_valuta, valuta
import pandas as pd


# DATASET CLEAN
path='http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/NCVR2/'

src_links = [
path+'NCVR_AF_clean.csv',
path+'NCVR_BF_clean.csv',
path+'NCVR_CF_clean.csv']

SOURCES = { 'S'+str(i+1) : pd.read_csv(link).astype(str) for i, link in enumerate(src_links) }

GoldStandard=pd.read_csv(path + "GoldStandardCLEAN2.csv")

GoldStandard.columns=['l_id','r_id']



# print(GoldStandard.columns)
for s in SOURCES:
    # print(SOURCES[s].columns)
    # creazione di una variabile mix per avere confronto con nome e cognome assieme
    SOURCES[s]['fullname'] = SOURCES[s]['first_name']+' '+SOURCES[s]['last_name']
    SOURCES[s]['fullnamezip'] = SOURCES[s]['first_name']+' '+SOURCES[s]['last_name']+' '+SOURCES[s]['zip_code']


# blocking rules - blocco su zip_code
# method = "join_blocker"
method = "rule_based_diff_type"
blocking_keys = ["fullname"]
blocking_rules = [[{"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) < {}", "attr": "fullname", "score": "0.3"}]]
# matching rules
matching_rules = [
        [{"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) >= {}", "attr": "fullname", "score": "0.3"}]
]
# l attrs to exclude
omit_l_attrs = ["l_id"]
# r attrs to exclude
omit_r_attrs = ["r_id"]
# matching method
matching_method = "STAB" # STAB # SYMM

em = EntityMatching(SOURCES)
entity_match_table = em.process(method, blocking_keys, omit_l_attrs, omit_r_attrs,
                                blocking_rules, matching_rules, matching_method)

cluster_entity_matching=EntityMatching.cluster_componenti_connessi(entity_match_table[['l_id','r_id']],
                                                                   EntityMatching.id_sources(SOURCES))
cluster_gold_standard=EntityMatching.cluster_componenti_connessi(GoldStandard[['l_id','r_id']],
                                                                 EntityMatching.id_sources(SOURCES))

Valuta = valuta(EntityMatching.calcola_match_indotti_cluster(cluster_gold_standard).rename(columns={"l_id": "A", "r_id": "B"}),
                     EntityMatching.calcola_match_indotti_cluster(cluster_entity_matching).rename(columns={"l_id": "A", "r_id": "B"}))
VediValuta = vedi_valuta(EntityMatching.calcola_match_indotti_cluster(cluster_gold_standard).rename(columns={"l_id": "A", "r_id": "B"}),
                     EntityMatching.calcola_match_indotti_cluster(cluster_entity_matching).rename(columns={"l_id": "A", "r_id": "B"}),
                    "FP")
print(Valuta)
print(VediValuta)
# print(entity_match_table)
