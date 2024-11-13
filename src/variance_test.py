import numpy as np
import pandas as pd
import csv
import os

# custom import
import src.graphs_generation as gen_graphs


class Variance_algo:
    def __init__(self, graph_size, config_file):
        self.graph_size = graph_size
        self.config_file = config_file
        if config_file["p_correction_type"] == "p_reduce":
            self.p0 = 0.5
            self.p1 = 0.5
        # elif config_file["p_correction_type"] == "p_increase":
        # TODO: ADAPT VARIANCE TEST IN p_increase CASE
        #     # calculate p_0 and p_1?
        else:
            raise ValueError(
                "Invalid p_correction_type. Variance test only available for 'p_reduce'"
            )

    def predict(self, adjacency_matrices, clique_size):
        """
        Defines the threshold for the current clique size and determines if each graph in the batch contain a clique.

        Parameters:
            adjacency_matrices (torch.Tensor): Batch of binary adjacency matrices of the graphs (batch_size, graph_size, graph_size).
            clique_size (int): size of the clique in the current batch of graphs


        Returns:
            torch.Tensor: A tensor with values between 0 and 1 indicating the probability of clique presence for each graph.
        """

        # - calculate threshold for current clique size
        # - calculate variance
        # - return hard output on provided test batch

        # # Convert input tensor to numpy array for variance calculation
        # adjacency_matrices = adjacency_matrices.cpu().numpy()

        # predictions = []
        # for matrix in adjacency_matrices:
        #     # Calculate edge variance for the current graph
        #     edge_variance = np.var(matrix)

        #     # Apply decision criterion to output values between 0 and 1
        #     if abs(edge_variance) < self.threshold_const:
        #         predictions.append(1.0)  # Indicates presence of a clique
        #     else:
        #         predictions.append(0.0)  # Indicates absence of a clique

        # # Convert predictions to a torch tensor
        # return torch.tensor(predictions, dtype=torch.float32)

    def test_and_save(self, results_dir):

        # Notify start of testing:
        print(f"| Started testing variance algorithm...")

        # Create empty dictionaries for storing testing results:
        fraction_correct_results = {}  # Fraction correct for each clique size
        metrics_results = {}  # Metrics dictionary

        # Calculate max clique size (proportion of graph size):
        max_clique_size = int(
            self.config_file["testing_parameters"]["max_clique_size_proportion_test"]
            * self.graph_size
        )

        # Calculate array of clique sizes for all test curriculum:
        if (
            max_clique_size
            < self.config_file["testing_parameters"]["clique_testing_levels"]
        ):
            # If max clique size is less than the the number of test levels, use max clique size as the number of test levels
            clique_sizes = np.linspace(max_clique_size, 1, num=max_clique_size).astype(
                int
            )
        else:
            # If max clique size is greater than the minimum clique size, use the default number of test levels
            clique_sizes = np.linspace(
                max_clique_size,
                1,
                num=self.config_file["testing_parameters"]["clique_testing_levels"],
            ).astype(int)

        # Initialize true positive, false positive, true negative, false negative
        TP, FP, TN, FN = 0, 0, 0, 0

        # Initialize arrays for AUC-ROC calculation
        y_scores = []
        y_true = []

        # Loop for decreasing clique sizes
        for current_clique_size in clique_sizes:

            # Initialize fraction correct list, updated at each test iteration
            fraction_correct_list = []

            # Loop for testing iterations:
            for test_iter in range(
                self.config_file["testing_parameters"]["test_iterations"]
            ):

                # Generate clique size value of each graph in the current batch
                clique_size_array_test = gen_graphs.generate_batch_clique_sizes(
                    np.array([current_clique_size]),
                    self.config_file["testing_parameters"]["num_test"],
                )

                # Generate validation graphs
                test = gen_graphs.generate_batch(
                    self.config_file["testing_parameters"]["num_test"],
                    self.graph_size,
                    clique_size_array_test,
                    self.config_file["p_correction_type"],
                    False,
                )

                # Perform prediction on test data
                hard_output = self.predict(test[0], current_clique_size)

                # Initialize tensor to store hard predictions
                hard_output = np.zeros(
                    [self.config_file["testing_parameters"]["num_test"]]
                )

                # Counting the number of correct classifications:
                for index in range(self.config_file["testing_parameters"]["num_test"]):
                    if hard_output[index] == 1.0:
                        if test[1][index] == 1.0:
                            TP += 1
                        else:
                            FP += 1
                    else:
                        if test[1][index] == 0.0:
                            TN += 1
                        else:
                            FN += 1

                # Updating fraction correct list with the accuracy of the current test iteration:
                fraction_correct_list.append(
                    (hard_output == test[1]).sum().item()
                    / self.config_file["testing_parameters"]["num_test"]
                )

                # Free up memory after this test iteration
                del (
                    soft_output,
                    hard_output,
                    test,
                )  # Delete unnecessary variables

            # Update dictionary after all test iterations for the current clique size:
            fraction_correct_results[current_clique_size] = round(
                sum(fraction_correct_list) / len(fraction_correct_list), 2
            )

            # Print test progress for the current clique size:
            print(
                f"||| Completed testing for clique = {current_clique_size}. "
                f"Average fraction correct on test set = {fraction_correct_results[current_clique_size]}"
            )
            print("|||===========================================================")

        # Calculate relevant metrics:
        epsilon = 1e-10  # To avoid division by zero
        precision = TP / (TP + FP + epsilon)
        recall = TP / (TP + FN + epsilon)
        F1 = 2 * (precision * recall) / (precision + recall + epsilon)

        # Store metrics in results dictionary:
        metrics_results["TP"] = TP
        metrics_results["FP"] = FP
        metrics_results["TN"] = TN
        metrics_results["FN"] = FN
        metrics_results["precision"] = precision
        metrics_results["recall"] = recall
        metrics_results["F1"] = F1

        # Notify completion of testing:
        print(f"| Finished testing variance algorithm.")

        # Saving accuracy results in .csv file:
        # - defining file name and path:
        file_path = os.path.join(
            results_dir, f"Variance_test_N{self.graph_size}_fraction_correct.csv"
        )
        # - saving the dictionary as a .csv file:
        with open(file_path, "w") as file:
            writer = csv.writer(file)
            writer.writerow(["clique size", "fraction correct"])  # Add column labels
            for key, value in fraction_correct_results.items():
                writer.writerow([key, value])

        # Saving metrics results in .csv file:
        # - defining file name and path:
        file_path = os.path.join(
            results_dir, f"Variance_test_N{self.graph_size}_metrics.csv"
        )
        # - saving the dictionary as a .csv file:
        pd.DataFrame([metrics_results]).to_csv(file_path, index=False)

        print(f"- Variance algorithm results saved successfully in {results_dir}.")

        return
