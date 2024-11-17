import numpy as np
import pandas as pd
import csv
import os
from scipy import special
import math


# Variance algo as described/implemented in:
# https://www.overleaf.com/project/672207ce18cc0a8388a1a7f8
# https://colab.research.google.com/drive/1JbvnecwCdaKpmxhugiLszQ3UXiIQ-dQ0?usp=sharing#scrollTo=BbpIleePIUaN
class Variance_algo:
    def __init__(self, config_file, graph_size):
        self.graph_size = graph_size
        if config_file["p_correction_type"] == "p_reduce":
            self.p0 = 0.5
        # elif config_file["p_correction_type"] == "p_increase":
        # TODO: ADAPT VARIANCE TEST IN p_increase CASE
        #     # calculate p_0 and p_1?
        else:
            raise ValueError(
                "Invalid p_correction_type. Variance test only available for 'p_reduce'"
            )

    def calculate_fraction_correct(self, clique_size):
        q_val = clique_size / self.graph_size
        z_val = (q_val**2 / (1 - q_val**2)) * ((1 - self.p0) / self.p0)

        # Calculating fraction of correct responses (Equation 2):
        return 0.5 + 0.5 * (
            special.erf(np.sqrt(np.log(1 / (1 - z_val)) / z_val))
            - special.erf(np.sqrt(((1 - z_val) / z_val) * np.log((1 / (1 - z_val)))))
        )

    def find_k0(self):

        # Notify start of testing:
        print(
            f"| Finding K0 of variance algorithm for graphs of size {self.graph_size}..."
        )

        # Results show that K0 value is above 0.6*graph_size -> start from this value and increment by 1 until fraction_correct > 0.75
        clique_size = 0.6 * self.graph_size
        fraction_correct = self.calculate_fraction_correct(clique_size)
        while fraction_correct <= 0.75:
            clique_size += 1
            print(f"||| Testing clique size = {clique_size}")
            # Calculating fraction of correct responses (Equation 2):
            fraction_correct = self.calculate_fraction_correct(clique_size)
            print(f"||| Fraction correct = {fraction_correct}")
            if clique_size > self.graph_size:
                raise ValueError(
                    f"K0 value for the variance algorithm is above the graph size {self.graph_size}. Check code."
                )

        # At the end of the loop, the last clique size value is the one that satisfies the condition:
        k0 = clique_size

        # Making sure that the p_correction is compatible with the defined K0:
        if k0 > (
            (1 + math.sqrt(1 + 4 * self.p0 * self.graph_size * (self.graph_size - 1)))
            / 2
        ):
            clique_limit = int(
                (
                    1
                    + math.sqrt(
                        1 + 4 * self.p0 * self.graph_size * (self.graph_size - 1)
                    )
                )
                / 2
            )
            raise ValueError(
                f"The k0 value for the variance algorithm is {int(k0)} for graphs of size {self.graph_size}, which leads to a negative corrected probability of association between nodes. Clique size values have to be lower than {round(clique_limit)}"
            )
        # Notify discovery of K0:
        print(
            f"| K0 value of variance algorithm is {int(k0)} in graphs of size {self.graph_size}."
        )

        return int(k0)

    def save_k0(self, results_dir):
        # Calculating K0:
        k0 = self.find_k0()

        # Saving K0 in .csv file:
        # - defining file name and path:
        file_path = os.path.join(
            results_dir, f"Variance_test_N{self.graph_size}_K0.csv"
        )
        # - saving the dictionary as a .csv file:
        with open(file_path, "w") as file:
            writer = csv.writer(file)
            writer.writerow(["K0"])  # Add column labels
            writer.writerow([k0])

        print(f"- K0 value file saved successfully in {results_dir}.")

        return
