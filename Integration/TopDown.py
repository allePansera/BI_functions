import pandas as pd
from tqdm import tqdm
from SchemaMatching import SchemaMatching
from CorrispBuilder.CorrisBuilder import CorrisBuilder
from threading import Thread

global_matching_table = None

def match_table_thread(global_schema, local_schemas, y_index,  sim_methods, corr_method, score, weights):
    global global_matching_table
    sm = SchemaMatching(global_schema, local_schemas[y_index])
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
    elif "TOP_K" in corr_method:
        # Expecting values like TOP_K_2 in order to extract 2 as an integer
        k = int(corr_method.split("_")[-1])
        cb.top_k = k
        match_table = cb.top_k_method()
    else:
        raise Exception(f"Corresp. method '{corr_method}' not supported")
    match_table = match_table[["A", "B", "Sim. Score"]]
    # a -> attr. of global schema
    # b -> attr. of local schema
    match_table = CorrisBuilder.thresholding(match_table, 0.6)

    match_table.columns = ['GAT', 'LAT', 'Sim. Score']
    match_table['SOURCE'] = str(y_index)
    match_table['SLAT'] = match_table['SOURCE'] + '_' + match_table['LAT']
    global_matching_table = global_matching_table.append(match_table, sort=False)


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
    global global_matching_table
    global_matching_table = pd.DataFrame(columns=['GAT', 'SOURCE', 'LAT', 'SLAT', 'Sim. Score'])
    thread_pool = []

    # declare all thread
    for y in local_schemas.keys():
        t = Thread(target=match_table_thread, args=(global_schema, local_schemas, y, sim_methods, corr_method, score, weights))
        thread_pool.append(t)

    # run all thread
    for t in tqdm(thread_pool):
        t.start()

    #wait for all thread end
    for t in tqdm(thread_pool):
        t.join()

    return global_matching_table