import pandas as pd
from tqdm import tqdm
from SchemaMatching import SchemaMatching
from CorrispBuilder.CorrisBuilder import CorrisBuilder

def global_match_table(local_schemas: list, global_schema: pd.DataFrame, sim_methods, corr_method="STAB_MARR", score="SimAvg", weights=[]):
    """

    :param local_schemas: list of local schema to integrate
    :param global_schema: global schema to compare
    :param sim_methods: methods to build ensemble similarity method
    :param corr_method: methods to use for correspondences building ["STAB_MARR", "SYMM_MATCH", "TOP_1"]
    :param score: SimMin/SimMax/SimAvg
    :param weights: weighted sum of score
    :return: match table global with each source
    """
    global_matching_table = pd.DataFrame(columns=['GAT', 'SOURCE', 'LAT', 'SLAT', 'Sim. Score'])

    for y in tqdm(local_schemas.keys()):
        sm = SchemaMatching(global_schema, local_schemas[y])
        sim_table = sm.ensemble_sim(methods=sim_methods, weights=weights)
        sim_table = sim_table.rename(columns={score: 'Sim. Score'})
        sim_table = sim_table[["A", "B", "Sim. Score"]]

        cb = CorrisBuilder(sim_table, top_k=1)
        if corr_method == "STAB_MARR":
            match_table = cb.stable_marriage_method()
        elif corr_method == "SYMM_MATCH":
            match_table = cb.symmetric_best_match_method()
        elif corr_method == "TOP_1":
            match_table = cb.top_k_method()
        else:
            raise Exception(f"Corresp. method '{corr_method}' not supported")
        match_table = match_table[["A", "B", "Sim. Score"]]
        # a -> attr. of global schema
        # b -> attr. of local schema
        match_table = CorrisBuilder.thresholding(match_table, 0.6)

        match_table.columns = ['GAT', 'LAT', 'Sim. Score']
        match_table['SOURCE'] = str(y)
        match_table['SLAT'] = match_table['SOURCE'] + '_' + match_table['LAT']
        global_matching_table = global_matching_table.append(match_table, sort=False)

    return global_matching_table