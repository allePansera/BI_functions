import pandas as pd
import numpy as np
from flexmatcher import *
from SimilarityMeasure.PreProcessing import string_pre_process
from SimilarityMeasure.SimilarityMeasure import levenshtein_similarity_function
from SimilarityMeasure.SimilarityMeasure import jaro_similarity_function
from SimilarityMeasure.SimilarityMeasure import jaro_winkler_similarity_function
from SimilarityMeasure.SimilarityMeasure import jaccard_similarity_function
from SimilarityMeasure.SimilarityMeasure import generalized_jaccard_similarity_function
from SimilarityMeasure.SimilarityMeasure import jaccard_tokenize_similarity_function
from SimilarityMeasure.SimilarityMeasure import monge_elkann_similarity_function
from SimilarityMeasure.SimilarityMeasure import overlap_coefficient_similarity_function
from SimilarityMeasure.SimilarityMeasure import similarity_join_function


## TODO Consider adding factory method for each similarity method
# BUG INSIDE JACCARD: not finding cols

class SchemaMatching:
    """Class for executing basic schema matching operations"""
    SUPPORTED_METHOD = {"label": ["LEV", "JARO", "JARO_WINK", "JAC", "ME", "OC"],
                        "value_overlap": ["JAC", "GEN_JAC", "EXT_JAC_LEV", "EXT_JAC_JAC", "SJ"]}

    def __init__(self, dataset_a: pd.DataFrame, dataset_b: pd.DataFrame):
        """
        Constructor only initialize class attributes.

        Parameters
        ----------
        dataset_a: pd.DataFrame
            first dataset to compare

        dataset_b: pd.DataFrame
            second dataset to compare
        """
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        # label pre-processing
        # self.__cols_label_pre_process(self.dataset_a)
        # self.__cols_label_pre_process(self.dataset_b)
        # sim. table pre-processor
        self.sim_table_empty = self.__sim_table_builder()

    def __sim_table_builder(self):
        """
        Build empty similarity table
        :return: pd.DataFrame containing a simTable
        """
        schema_a = self.dataset_a.columns
        schema_b = self.dataset_b.columns
        cols_a_schema = pd.DataFrame({'A': schema_a})
        cols_b_schema = pd.DataFrame({'B': schema_b})
        sim_table = cols_a_schema.assign(key=1).merge(cols_b_schema.assign(key=1), on='key').drop(columns='key')
        sim_table.columns = ['A', 'B']
        sim_table['Sim. Score'] = np.NAN
        return sim_table

    def __cols_label_pre_process(self, dataset: pd.DataFrame):
        """
        This method can be used to pre-process label name inside each dataset
        :return: Nothing, just update received dataframe instance
        """
        cols_list = dataset.columns
        for index, cols_name in enumerate(cols_list):
            new_cols_name = string_pre_process(cols_name)
            dataset.rename(columns={cols_name: new_cols_name}, inplace=True)

    def label_based_sim(self, method="LEV"):
        """
        Evaluate similarity table between 2 schemas label.
        :param method: stands for the distance measure technique
        :return: pd.DataFrame instance comparing all similarities
        """
        if method not in SchemaMatching.SUPPORTED_METHOD["label"]:
            raise Exception(f"Similarity method '{method}' not supported")

        sim_table = self.sim_table_empty.copy()

        if method == "LEV":
            sim_table['Sim. Score'] = sim_table.apply(levenshtein_similarity_function, axis=1)

        if method == "JARO":
            sim_table['Sim. Score'] = sim_table.apply(jaro_similarity_function, axis=1)

        if method == "JAC":
            sim_table['Sim. Score'] = sim_table.apply(jaccard_tokenize_similarity_function, axis=1)

        if method == "JARO_WINK":
            sim_table['Sim. Score'] = sim_table.apply(jaro_winkler_similarity_function, axis=1)

        if method == "ME":
            sim_table['Sim. Score'] = sim_table.apply(monge_elkann_similarity_function, axis=1)

        if method == "OC":
            sim_table['Sim. Score'] = sim_table.apply(overlap_coefficient_similarity_function, axis=1)

        return sim_table

    def value_overlap_sim(self, method="JAC"):
        """
        Evaluate schema similarity based on attributes' values overlap
        :param method: stands for the distance measure technique
        :return: pd.DataFrame instance comparing all similarities
        """
        if method not in SchemaMatching.SUPPORTED_METHOD["value_overlap"]:
            raise Exception(f"Similarity method '{method}' not supported")

        sim_table = self.sim_table_empty.copy()

        if method == "JAC":
            for index, row in self.sim_table_empty.iterrows():
                attr_a = row['A']
                attr_b = row['B']
                sim_score = jaccard_similarity_function(self.dataset_a.copy(), self.dataset_b.copy(), attr_a, attr_b)
                sim_table.at[index, 'Sim. Score'] = sim_score

        if method == "GEN_JAC":
            for index, row in self.sim_table_empty.iterrows():
                attr_a = row['A']
                attr_b = row['B']
                sim_score = generalized_jaccard_similarity_function(self.dataset_a.copy(), self.dataset_b.copy(),
                                                                    attr_a, attr_b)
                sim_table.at[index, 'Sim. Score'] = sim_score

        if method == "SJ":
            for index, row in self.sim_table_empty.iterrows():
                attr_a = row['A']
                attr_b = row['B']
                sim_score = similarity_join_function(self.dataset_a.copy(), self.dataset_b.copy(),
                                                     attr_a, attr_b)
                sim_table.at[index, 'Sim. Score'] = sim_score

        if method == "EXT_JAC_LEV":
            sim_table = self.dataset_a.copy().drop_duplicates().merge(self.dataset_b.copy().drop_duplicates(),
                                                                      how='cross')
            sim_table["Sim. Score"] = sim_table.apply(levenshtein_similarity_function(), axis=1)

        if method == "EXT_JAC_JAC":
            sim_table = self.dataset_a.copy().drop_duplicates().merge(self.dataset_b.copy().drop_duplicates(),
                                                                      how='cross')
            sim_table["Sim. Score"] = sim_table.apply(jaccard_tokenize_similarity_function, axis=1)

        return sim_table

    def hybrid_sim(self, methods=[]):
        """
        This method build a similarity table considering more similarity measure
        :param methods: [{"value_overlap": "JAC"}, {"label": "LEV"}, ...]
        :return: pd.DataFrame instance comparing all similarities
        """
        # check if each method exists
        for method in methods:
            type = list(method.keys())[0]
            value = method[type]
            if value not in SchemaMatching.SUPPORTED_METHOD[type]:
                raise Exception(f"Similarity method '{value}' not supported for '{type}'")

        # build N (len(methods)) simTable
        sim_table_list = []
        for index, method in enumerate(methods):
            type = list(method.keys())[0]
            value = method[type]
            if type == "label":
                sim_table_list.append(self.label_based_sim(method=value))
            elif type == "value_overlap":
                sim_table_list.append(self.value_overlap_sim(method=value))

        # build unique simTable with all sim. score
        sim_table = self.sim_table_empty.copy()
        sim_table = sim_table.drop('Sim. Score', axis=1)
        for index, temp_sim_table in enumerate(sim_table_list):
            sim_table[f'Sim. {list(methods[index].values())[0]}'] = temp_sim_table['Sim. Score']

        return sim_table

    def ensemble_sim(self, methods=[]):
        """
        This method build a similarity table considering more similarity measure
        :param methods: [{"value_overlap": "JAC"}, {"label": "LEV"}, ...]
        :return: pd.DataFrame instance comparing all similarities
        """
        # check if each method exists
        for method in methods:
            type = list(method.keys())[0]
            value = method[type]
            if value not in SchemaMatching.SUPPORTED_METHOD[type]:
                raise Exception(f"Similarity method '{value}' not supported for '{type}'")

        # build N (len(methods)) simTable
        sim_table_list = []
        for index, method in enumerate(methods):
            type = list(method.keys())[0]
            value = method[type]
            if type == "label":
                sim_table_list.append(self.label_based_sim(method=value))
            elif type == "value_overlap":
                sim_table_list.append(self.value_overlap_sim(method=value))

        # build unique simTable with all sim. score
        sim_table = self.sim_table_empty.copy()
        sim_table = sim_table.drop('Sim. Score', axis=1)
        for index, temp_sim_table in enumerate(sim_table_list):
            sim_table[f'Sim. {list(methods[index].values())[0]}'] = temp_sim_table['Sim. Score']

        # evaluating avg, min and max
        sim_table_measure = sim_table.drop(["A", "B"], axis=1)
        sim_table_ensemble = pd.DataFrame({'A': sim_table['A'], 'B': sim_table['B']})

        sim_table_ensemble['SimMin'] = sim_table_measure[sim_table_measure.columns].min(axis=1)
        sim_table_ensemble['SimMax'] = sim_table_measure[sim_table_measure.columns].max(axis=1)
        sim_table_ensemble['SimAvg'] = sim_table_measure[sim_table_measure.columns].mean(axis=1)
        return sim_table_ensemble

    def flex_matcher(self, source_list: list, data_mapping_list: list, local_schema: pd.DataFrame):
        """
        Return flex matcher after.

        :param source_list: list of DataFrame to use for flex matcher classifier.
        Expected a list o dataframe without indexes
        :param data_mapping_list: list of mapping to consider for flex matcher attr similarities. Expected
        a list of dictionary.
        :param local_schema: final schema to compare. Expected a list o dataframe without indexes
        :return: classifier prediction on new schema
        """
        fm = FlexMatcher(source_list, data_mapping_list, sample_size=100)
        fm.train()
        return fm.make_prediction(local_schema)
