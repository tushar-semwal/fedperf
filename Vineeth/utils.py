import os
import glob
import pickle
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


methods = ["FedAvg", "FedMed", "FedProx", "qFedAvg"]
local_rounds = ["01", "05", "10", "25"]


def plot_accuracy(path, dataset):
    ROUNDS = 50
    exp_matches = [" CNN on IID", " CNN on Non IID", " MLP on IID", " MLP on Non IID", " LSTM on IID", " LSTM on Non IID"]

    for exp_match in exp_matches:
        for local_round in local_rounds:
            plt.figure()
            for method in methods:
                pickle_files = glob.glob(f"{path}/{dataset}/{method}/{local_round}/*.pkl")
                print(pickle_files)

                if not pickle_files:
                    continue

                with open(pickle_files[0], "rb") as file:
                    log_dict = pickle.load(file)

                for experiment in log_dict.keys():
                    if experiment.endswith(exp_match):
                        print(experiment)

                        if 'Non IID' in experiment:
                            IS_IID = 'Non_IID'
                        else:
                            IS_IID = 'IID'

                        for accuracy_profile in log_dict[experiment]["test_accuracy"]:
                            if len(accuracy_profile) < ROUNDS:
                                accuracy_profile.extend([accuracy_profile[-1]] * (ROUNDS - len(accuracy_profile)))

                        accuracy_runs = np.array(log_dict[experiment]["test_accuracy"])

                        mean_accuracy_profile = np.mean(accuracy_runs, axis=0)
                        std_dev_accuracy_profile = np.std(accuracy_runs, axis=0)

                        plt.grid(False)
                        plt.plot(np.arange(ROUNDS), mean_accuracy_profile, label=f"{method}")
                        plt.fill_between(
                            np.arange(ROUNDS),
                            mean_accuracy_profile - std_dev_accuracy_profile,
                            mean_accuracy_profile + std_dev_accuracy_profile,
                            alpha=0.5,
                        )

                        # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True)
                        plt.title(f"Local Rounds {local_round}", pad=27.5)
                        plt.xlabel("Global Communication Rounds")
                        plt.ylabel("Test Accuracy")

                        plt.tight_layout()
                        plt.savefig(
                            f"plots/{dataset}/Local_Rounds/Accuracy_Profile/{IS_IID}/{experiment}_{local_round}.svg", format="svg", dpi=1000
                        )
            plt.show()


def plot_accuracy_stacked_error_bar_plot(path, dataset):
    ROUNDS = 50

    final_accuracy_mean_tracker = {}
    final_accuracy_std_tracker = {}
    final_accuracy_max_tracker = {}
    final_accuracy_min_tracker = {}

    for local_round in local_rounds:
        plt.figure()
        for method in methods:
            pickle_files = glob.glob(f"{path}/{dataset}/{method}/{local_round}/*.pkl")
            print(pickle_files)

            if not pickle_files:
                continue

            with open(pickle_files[0], "rb") as file:
                log_dict = pickle.load(file)

            for experiment in log_dict.keys():
                print(experiment)
                for accuracy_profile in log_dict[experiment]["test_accuracy"]:
                    if len(accuracy_profile) < ROUNDS:
                        accuracy_profile.extend([accuracy_profile[-1]] * (ROUNDS - len(accuracy_profile)))

                accuracy_runs = np.array(log_dict[experiment]["test_accuracy"])

                mean_accuracy_profile = np.mean(accuracy_runs, axis=0)
                std_dev_accuracy_profile = np.std(accuracy_runs, axis=0)
                max_accuracy_profile = np.max(accuracy_runs, axis=0)
                min_accuracy_profile = np.min(accuracy_runs, axis=0)

                final_accuracy_mean_tracker[f"{method}-{experiment}"] = mean_accuracy_profile[-1]
                final_accuracy_std_tracker[f"{method}-{experiment}"] = std_dev_accuracy_profile[-1]
                final_accuracy_max_tracker[f"{method}-{experiment}"] = max_accuracy_profile[-1]
                final_accuracy_min_tracker[f"{method}-{experiment}"] = min_accuracy_profile[-1]

            exp_matches = [" on IID", " on Non IID"]

            for exp_match in exp_matches:
                if 'Non IID' in exp_match:
                    IS_IID = 'Non_IID'
                else:
                    IS_IID = 'IID'

                final_acc_mean_list = []
                final_acc_std_list = []
                final_acc_max_list = []
                final_acc_min_list = []
                key_list = []
                for key in final_accuracy_mean_tracker:
                    if key.endswith(exp_match):
                        key_list.append(key.replace(f"-{dataset}", "").replace(exp_match, ""))
                        final_acc_mean_list.append(final_accuracy_mean_tracker[key])
                        final_acc_std_list.append(final_accuracy_std_tracker[key])
                        final_acc_max_list.append(final_accuracy_max_tracker[key])
                        final_acc_min_list.append(final_accuracy_min_tracker[key])

                plt.grid(True)
                plt.errorbar(
                    np.arange(len(key_list)),
                    np.array(final_acc_mean_list),
                    np.array(final_acc_std_list),
                    fmt="ok",
                    lw=3,
                )
                plt.errorbar(
                    np.arange(len(key_list)),
                    np.array(final_acc_mean_list),
                    [
                        np.array(final_acc_mean_list) - np.array(final_acc_min_list),
                        np.array(final_acc_max_list) - np.array(final_acc_mean_list),
                    ],
                    fmt=".k",
                    ecolor="black",
                    lw=1,
                )

                plt.xticks(np.arange(len(key_list)), key_list, rotation="vertical")
                plt.title(f"Local Rounds {local_round}")
                plt.xlabel("Algorithms")
                plt.ylabel("Test Accuracy")

                plt.tight_layout()
                plt.savefig(
                    f"plots/{dataset}/Local_Rounds/Stacked_Error_Bar/{IS_IID}/{dataset}{exp_match}_{local_round}.svg",
                    format="svg",
                    dpi=1000,
                )
                plt.show()


def plot_fairness(path, dataset, NUM_REPS):
    ROUNDS = 50
    NUM_CLIENTS = 100
    exp_matches = [" CNN on IID", " CNN on Non IID", " MLP on IID", " MLP on Non IID", " LSTM on IID", " LSTM on Non IID"]

    for exp_match in exp_matches:
        plt.figure()
        for method in methods:
            pickle_files = glob.glob(f"{path}/{dataset}/{method}/*.pkl")
            print(pickle_files)

            if not pickle_files:
                continue

            for pickle_file in pickle_files:
                with open(pickle_file, "rb") as file:
                    log_dict = pickle.load(file)

                for experiment in log_dict.keys():
                    if experiment.endswith(exp_match):
                        print(experiment)

                        if 'Non IID' in experiment:
                            IS_IID = 'Non_IID'
                        else:
                            IS_IID = 'IID'

                        final_accuracy = [0] * NUM_CLIENTS
                        final_accuracy_count = [0] * NUM_CLIENTS

                        for rep in range(NUM_REPS):
                            for client in range(NUM_CLIENTS):
                                if not log_dict[experiment]["test_accuracy_clients"][rep][client]:
                                    continue

                                final_accuracy[client] += log_dict[experiment]["test_accuracy_clients"][rep][client][
                                    -1
                                ][1]
                                final_accuracy_count[client] += 1

                        final_accuracy = [
                            accuracy / count if count else 0 for accuracy, count in zip(final_accuracy, final_accuracy_count)
                        ]
                        method_name = pickle_file.split("/")[-1].split(".")[0].replace("Fairness_", "")

                        plt.grid(True)
                        plt.hist(final_accuracy, bins=20)

                        # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                        plt.title(f"Fairness")
                        plt.xlabel("Test accuracy")
                        plt.ylabel("Number of clients")

                        plt.tight_layout()
                        plt.savefig(f"plots/{dataset}/Fairness/Histograms/{IS_IID}/{experiment}_{method_name}.svg", format="svg", dpi=1000)
                plt.show()


def plot_fairness_stacked_error_bar_plot(path, dataset, NUM_REPS):
    ROUNDS = 50
    NUM_CLIENTS = 100

    final_entropy_mean_tracker = {}
    final_entropy_std_tracker = {}
    final_entropy_max_tracker = {}
    final_entropy_min_tracker = {}

    plt.figure()
    for method in methods:
        pickle_files = glob.glob(f"{path}/{dataset}/{method}/*.pkl")
        print(pickle_files)

        if not pickle_files:
            continue

        for pickle_file in pickle_files:
            with open(pickle_file, "rb") as file:
                log_dict = pickle.load(file)

            for experiment in log_dict.keys():
                print(experiment)
                final_accuracy = [0] * NUM_CLIENTS
                entropy_runs = [None] * NUM_REPS

                for rep in range(NUM_REPS):
                    for client in range(NUM_CLIENTS):
                        if not log_dict[experiment]["test_accuracy_clients"][rep][client]:
                            continue

                        final_accuracy[client] = log_dict[experiment]["test_accuracy_clients"][rep][client][-1][1]

                    # print(method, experiment, len(final_accuracy), final_accuracy)

                    # histogram = np.histogram(final_accuracy, bins=1000, range=(0, 100), density=True)
                    # data_distribution = histogram[0]
                    # entropy = -(data_distribution * np.ma.log(np.abs(data_distribution))).sum()

                    histogram = np.asarray(np.histogram(final_accuracy, bins=1000, range=(0, 100), density=False)[0])
                    data_distribution = histogram / histogram.sum()
                    entropy = -(data_distribution * np.ma.log2(np.abs(data_distribution))).sum()

                    entropy_runs[rep] = entropy

                method_name = pickle_file.split("/")[-1].split(".")[0].replace("Fairness_", "")

                histogram = np.asarray([1 / histogram.shape[0] for _ in range(histogram.shape[0])])
                data_distribution = histogram / histogram.sum()
                max_entropy = -(data_distribution * np.ma.log2(np.abs(data_distribution))).sum()

                entropy_runs = np.array(entropy_runs)
                final_entropy_mean_tracker[f"{method_name}-{experiment}"] = np.mean(entropy_runs, axis=0)
                final_entropy_std_tracker[f"{method_name}-{experiment}"] = np.std(entropy_runs, axis=0)
                final_entropy_max_tracker[f"{method_name}-{experiment}"] = np.max(entropy_runs, axis=0)
                final_entropy_min_tracker[f"{method_name}-{experiment}"] = np.min(entropy_runs, axis=0)

    exp_matches = [" on IID", " on Non IID"]

    for exp_match in exp_matches:
        if 'Non IID' in exp_match:
            IS_IID = 'Non_IID'
        else:
            IS_IID = 'IID'

        final_acc_mean_list = []
        final_acc_std_list = []
        final_acc_max_list = []
        final_acc_min_list = []
        key_list = []
        for key in final_entropy_mean_tracker:
            if key.endswith(exp_match):
                key_list.append(key.replace(f"-{dataset}", "").replace(exp_match, ""))
                final_acc_mean_list.append(final_entropy_mean_tracker[key])
                final_acc_std_list.append(final_entropy_std_tracker[key])
                final_acc_max_list.append(final_entropy_max_tracker[key])
                final_acc_min_list.append(final_entropy_min_tracker[key])

        plt.grid(True)
        plt.errorbar(
            np.arange(len(key_list)), np.array(final_acc_mean_list), np.array(final_acc_std_list), fmt="ok", lw=3
        )
        plt.errorbar(
            np.arange(len(key_list)),
            np.array(final_acc_mean_list),
            [
                np.array(final_acc_mean_list) - np.array(final_acc_min_list),
                np.array(final_acc_max_list) - np.array(final_acc_mean_list),
            ],
            fmt=".k",
            ecolor="black",
            lw=1,
        )

        plt.errorbar(
            np.arange(len(key_list)),
            np.array([max_entropy for _ in range(len(key_list))]),
            [
                np.array([0 for _ in range(len(key_list))]),
                np.array([0 for _ in range(len(key_list))]),
            ],
            fmt='D',
            color="red",
            lw=1,
        )

        plt.xticks(np.arange(len(key_list)), key_list, rotation="vertical")
        plt.title(f"Fairness")
        plt.xlabel("Algorithms")
        plt.ylabel("Entropy")

        plt.tight_layout()
        plt.savefig(f"plots/{dataset}/Fairness/Stacked_Error_Bar/{IS_IID}/{dataset}{exp_match}.svg", format="svg", dpi=1000)
        plt.show()


if __name__ == "__main__":
    plot_accuracy("./Local_Rounds/", "MNIST")
    plot_accuracy("./Local_Rounds/", "CIFAR")
    plot_accuracy("./Local_Rounds/", "Shakespeare")

    plot_accuracy_stacked_error_bar_plot("./Local_Rounds/", "MNIST")
    plot_accuracy_stacked_error_bar_plot("./Local_Rounds/", "CIFAR")
    plot_accuracy_stacked_error_bar_plot("./Local_Rounds/", "Shakespeare")

    plot_fairness("./Fairness/", "MNIST", 5)
    plot_fairness("./Fairness/", "CIFAR", 5)
    plot_fairness("./Fairness/", "Shakespeare", 2)

    plot_fairness_stacked_error_bar_plot("./Fairness/", "MNIST", 5)
    plot_fairness_stacked_error_bar_plot("./Fairness/", "CIFAR", 5)
    plot_fairness_stacked_error_bar_plot("./Fairness/", "Shakespeare", 2)
