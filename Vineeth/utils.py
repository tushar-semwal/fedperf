import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt


methods = ["FedAvg", "FedMed", "FedProx", "qFedAvg"]
local_rounds = ["01", "05", "10", "25"]


def plot_accuracy(path, dataset):
    ROUNDS = 50

    for local_round in local_rounds:
        plt.figure()
        plt.grid()

        for method in methods:
            pickle_file = glob.glob(f"{path}/{dataset}/{method}/{local_round}/*.pkl")
            print(pickle_file)
            with open(pickle_file[0], "rb") as file:
                log_dict = pickle.load(file)

            for experiment in log_dict.keys():
                exp_match = " CNN on IID"
                # exp_match = ' CNN on Non IID'
                # exp_match = ' MLP on IID'
                # exp_match = ' MLP on Non IID'

                if experiment.endswith(exp_match):
                    print(experiment)
                    for accuracy_profile in log_dict[experiment]["test_accuracy"]:
                        if len(accuracy_profile) < ROUNDS:
                            accuracy_profile.extend([accuracy_profile[-1]] * (ROUNDS - len(accuracy_profile)))

                    accuracy_runs = np.array(log_dict[experiment]["test_accuracy"])

                    mean_accuracy_profile = np.mean(accuracy_runs, axis=0)
                    std_dev_accuracy_profile = np.std(accuracy_runs, axis=0)

                    plt.plot(np.arange(ROUNDS), mean_accuracy_profile, label=f"{method}")
                    plt.fill_between(
                        np.arange(ROUNDS),
                        mean_accuracy_profile - std_dev_accuracy_profile,
                        mean_accuracy_profile + std_dev_accuracy_profile,
                        alpha=0.5,
                    )

                    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                    plt.title(f"{dataset} Local Rounds {local_round}")
                    plt.xlabel("Global Communication Rounds")
                    plt.ylabel("Test Accuracy")

                    plt.tight_layout()
                    plt.savefig(f"plots/local_rounds_accuracy_{experiment}_{local_round}.svg", format="svg", dpi=1000)
        plt.show()


def plot_accuracry_stacked_errobar_plot(path, dataset):
    ROUNDS = 50

    final_accuracy_mean_tracker = {}
    final_accuracy_std_tracker = {}
    final_accuracy_max_tracker = {}
    final_accuracy_min_tracker = {}

    for local_round in local_rounds:
        plt.figure()
        plt.grid()

        for method in methods:
            pickle_file = glob.glob(f"{path}/{dataset}/{method}/{local_round}/*.pkl")
            print(pickle_file)
            with open(pickle_file[0], "rb") as file:
                log_dict = pickle.load(file)

            for experiment in log_dict.keys():
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

        final_acc_mean_list = []
        final_acc_std_list = []
        final_acc_max_list = []
        final_acc_min_list = []
        key_list = []
        for key in final_accuracy_mean_tracker:
            if key.endswith(" Non IID"):
                key_list.append(key.replace(f"-{dataset}", "").replace(" on Non IID", ""))
                final_acc_mean_list.append(final_accuracy_mean_tracker[key])
                final_acc_std_list.append(final_accuracy_std_tracker[key])
                final_acc_max_list.append(final_accuracy_max_tracker[key])
                final_acc_min_list.append(final_accuracy_min_tracker[key])

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

        plt.xticks(np.arange(len(key_list)), key_list, rotation="vertical")
        plt.title(f"{dataset} Local Rounds {local_round}")
        plt.xlabel("Algorithms")
        plt.ylabel("Test Accuracy")

        plt.tight_layout()
        plt.savefig(f"plots/local_rounds_accuracy_stacked_{dataset}_{local_round}.svg", format="svg", dpi=1000)
        plt.show()


if __name__ == "__main__":
    plot_accuracy("./Local_Rounds/", "MNIST")
    plot_accuracy("./Local_Rounds/", "CIFAR")
    # plot_accuracry_stacked_errobar_plot('./Local_Rounds/', 'MNIST')
    # plot_accuracry_stacked_errobar_plot('./Local_Rounds/', 'CIFAR')
