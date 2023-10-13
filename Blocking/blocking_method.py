import py_entitymatching as em
import recordlinkage as rl
import pandas as pd
from SimilarityMeasure.SimilarityMeasure import levenshtein_similarity_function


def equivalence_blocker(A: pd.DataFrame, B: pd.DataFrame, blocking_key, l_attrs, r_attrs):
    """
    Equivalence blocker technique. Only one attr as blocking key
    :param A: first Dataset
    :param B: second Dataset
    :param blocking_key: attr. to use for blocking on both table
    :param l_attrs: attrs to compare from left dataset
    :param r_attrs: attrs to compare from right dataset
    :return:
    """


    # si settano le key dei due dataset
    em.set_key(A, 'l_id')
    em.set_key(B, 'r_id')

    # em.set_key(GoldStandard, 'id')
    # em.set_ltable(GoldStandard, A)
    # em.set_rtable(GoldStandard, B)
    # em.set_fk_ltable(GoldStandard, 'l_id')
    # em.set_fk_rtable(GoldStandard, 'r_id')
    ab = em.AttrEquivalenceBlocker()
    C = ab.block_tables(
        A, B,
        blocking_key, blocking_key,
        l_output_attrs=l_attrs,
        r_output_attrs=r_attrs
    ).rename(columns={'ltable_l_id': 'l_id', 'rtable_r_id': 'r_id'})

    return C


def join_blocker(A: pd.DataFrame, B: pd.DataFrame, blocking_keys: list):
    """
    equivalence blocker but with multiple blocking key
    :param A: first dataset
    :param B: second dataset
    :param blocking_keys:
    :return:
    """
    return A.merge(B, on=blocking_keys, how='inner')


def record_linkage_blocking(A: pd.DataFrame, B: pd.DataFrame, blocking_keys: list):
    indexer = rl.Index()
    for key in blocking_keys:
        indexer.block(key)
    candidate_links = indexer.index(A, B)
    df = pd.DataFrame(list(candidate_links), columns=["l_id", "r_id"])
    return df


def overlap_blocking(A: pd.DataFrame, B: pd.DataFrame, blocking_key, l_attrs, r_attrs, q_gram=3, overlap_size=2):
    """
    Overlap blocker technique. Only one attr as blocking key
    :param A: first Dataset
    :param B: second Dataset
    :param blocking_key: attr. to use for blocking on both table
    :param l_attrs: attrs to compare from left dataset
    :param r_attrs: attrs to compare from right dataset
    :return:
    """
    em.set_key(A, 'l_id')
    em.set_key(B, 'r_id')
    ob = em.OverlapBlocker()
    C_OB = ob.block_tables(A, B, blocking_key, blocking_key,
                           word_level=False,
                           q_val=q_gram,
                           overlap_size=overlap_size,
                           l_output_attrs=l_attrs,
                           r_output_attrs=r_attrs,
                           show_progress=False).rename(columns={'ltable_l_id': 'l_id', 'rtable_r_id': 'r_id'})
    return C_OB


def black_box_blocker_lev(A: pd.DataFrame, B: pd.DataFrame, blocking_key, l_attrs, r_attrs, sim_thresh=0.7):
    """
    black_box technique. Only one attr as blocking key and we think negative: we block where value is different
    :param A: first Dataset
    :param B: second Dataset
    :param blocking_key: attr. to use for blocking on both table
    :param l_attrs: attrs to compare from left dataset
    :param r_attrs: attrs to compare from right dataset
    :param sim_thresh: threshold for similarity function
    :return:
    """

    def custom_blocking_function(x, y):
        attr_a = f"{blocking_key}_a"
        attr_b = f"{blocking_key}_b"
        d = {attr_a: x[blocking_key], attr_b: y[blocking_key]}
        df = pd.DataFrame([d], columns=[attr_a, attr_b])
        if levenshtein_similarity_function(df.head(1), col_a=attr_a, col_b=attr_b) < sim_thresh:
            return True
        else:
            return False

    em.set_key(A, 'l_id')
    em.set_key(B, 'r_id')
    bb = em.BlackBoxBlocker()
    bb.set_black_box_function(custom_blocking_function)
    c_custom = bb.block_tables(A, B, l_output_attrs=l_attrs,
                            r_output_attrs=r_attrs,
                            show_progress=True).rename(columns={'ltable_l_id': 'l_id', 'rtable_r_id': 'r_id'})
    return c_custom


def rule_based_blocker(A: pd.DataFrame, B: pd.DataFrame, rules_list, l_attrs, r_attrs):
    """
    rules based blocker
    :param A: left dataset
    :param B: right dataset
    :param rules_list: list of list [[name_name_rule, name_name_rule], [name_name_rule]]
    inner list rules are under or constraint while first level constraints are under and condition.
    Weight can't be used.
    :return:
    """
    em.set_key(A, 'l_id')
    em.set_key(B, 'r_id')
    block_f = em.get_features_for_blocking(A, B, validate_inferred_attr_types=False)
    rb = em.RuleBasedBlocker()
    for and_rules in rules_list:
        or_rules = []
        or_rules.extend(and_rules)
        rb.add_rule(or_rules, block_f)

    C = rb.block_tables(A, B, l_output_attrs=l_attrs, r_output_attrs=r_attrs, show_progress=False).rename(columns={'ltable_l_id': 'l_id', 'rtable_r_id': 'r_id'})
    return C


def rule_based_blocker_diff_type(A: pd.DataFrame, B: pd.DataFrame, rules_list, l_attrs, r_attrs):
    """
    rules based blocker but we can have different features type
    :param A: left dataset
    :param B: right dataset
    :param rules_list: list of list [[name_name_rule, name_name_rule], [name_name_rule]]
    inner list rules are under or constraint while first level constraints are under and condition.
    Weight can't be used.
    :return:
    """
    em.set_key(A, 'l_id')
    em.set_key(B, 'r_id')
    block_f = em.get_features(A, B,
                             em.get_attr_types(A), em.get_attr_types(B),
                             em.get_attr_corres(A, B),
                             em.get_tokenizers_for_blocking(),
                             em.get_sim_funs_for_blocking())

    rb = em.RuleBasedBlocker()
    for and_rules in rules_list:
        or_rules = []
        or_rules.extend(and_rules)
        rb.add_rule(or_rules, block_f)

    C = rb.block_tables(A, B, l_output_attrs=l_attrs, r_output_attrs=r_attrs, show_progress=False).rename(columns={'ltable_l_id': 'l_id', 'rtable_r_id': 'r_id'})
    return C