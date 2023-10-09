import pandas as pd


class CorrisBuilder:

    def __init__(self, sim_table: pd.DataFrame, threshold=0.49, top_k=1):
        """

        :param simTable: similarity table
        """
        self.sim_table = sim_table.copy()
        self.sim_table["Sim. Score"] = self.sim_table["Sim. Score"].astype(float)
        self.threshold = threshold
        self.top_k = top_k

    def threshold_method(self):
        """
        Using threshold returns only similarity above fixed values. Local compare method
        :return: pd.DataFrame instance filtered
        """
        return self.sim_table[self.sim_table["Sim. Score"] > self.threshold]

    def top_k_method(self, side="A"):
        """
        Return per each attribute tok k similarities. Local compare method
        :return: pd.Dataframe instance filtered
        """
        if side not in ["A", "B"]: raise Exception(f"Side '{side}' must be in ['A', 'B']")
        self.sim_table['Rank'] = self.sim_table.sort_values(['Sim. Score'], ascending=[False]).groupby(side).cumcount() + 1
        self.sim_table.sort_values([side, 'Rank'])
        return self.sim_table[self.sim_table["Rank"] <= self.top_k].sort_values([side, 'Rank'])

    def symmetric_best_match_method(self):
        """
        Global method, the idea is similar to top_1 but with 1:1 correspondence
        :return: pd.DataFrame instance filtered
        """
        self.sim_table['A_Rank'] = self.sim_table.sort_values(['Sim. Score'], ascending=[False]).groupby(['A']).cumcount() + 1
        self.sim_table['B_Rank'] = self.sim_table.sort_values(['Sim. Score'], ascending=[False]).groupby(['B']).cumcount() + 1
        return self.sim_table[(self.sim_table['A_Rank'] == 1) & (self.sim_table['B_Rank'] == 1)].drop(
                    columns=['A_Rank', 'B_Rank']).sort_values(['Sim. Score'], ascending=[False])

    def stable_marriage_method(self):
        """
        Global method, the idea is similar to top_1 but with 1:1 correspondence
        :return: pd.DataFrame instance filtered
        """
        match = pd.DataFrame(columns=['A', 'B', 'Sim. Score'])
        self.sim_table = self.sim_table.sort_values(['Sim. Score'], ascending=[False])
        while True:
            r = self.sim_table.loc[(~self.sim_table['A'].isin(match['A'])) & (~self.sim_table['B'].isin(match['B']))]
            if len(r) == 0:
                break
            x = r.iloc[0, :]
            match = match.append(x, ignore_index=True)
        return match

