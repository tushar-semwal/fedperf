import os
import glob
import pickle
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})


methods = ["FedAvg", "FedMed", "FedProx", "qFedAvg"]
local_rounds = ["01", "05", "10", "25"]


def plot_accuracy(path, dataset):
    ROUNDS = 50
    exp_matches = [" CNN on IID", " CNN on Non IID", " MLP on IID", " MLP on Non IID", " LSTM on IID", " LSTM on Non IID"]

    for exp_match in exp_matches:
        for local_round in local_rounds:
            plt.figure(figsize=[8, 6])
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

                        if dataset == "Shakespeare":
                            accuracy_runs = accuracy_runs * 100

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

                        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True)
                        # plt.title(f"Local Rounds {local_round}", pad=27.5)
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
        plt.figure(figsize=[8, 6])
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

                if dataset == "Shakespeare":
                    accuracy_runs = accuracy_runs * 100

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

                plt.xticks(np.arange(len(key_list)), key_list, rotation="45", ha='right')
                # plt.title(f"Local Rounds {local_round}")
                plt.xlabel("Algorithms")
                plt.ylabel("Test Accuracy")

                plt.margins(0.1)
                plt.tight_layout()
                plt.savefig(
                    f"plots/{dataset}/Local_Rounds/Stacked_Error_Bar/{IS_IID}/{dataset}{exp_match}_{local_round}.svg",
                    format="svg",
                    dpi=1000,
                )
                plt.show()



def plot_accuracy_stacked_error_bar_multiple_plot(path, dataset):
    ROUNDS = 50

    final_accuracy_mean_tracker = {}
    final_accuracy_std_tracker = {}
    final_accuracy_max_tracker = {}
    final_accuracy_min_tracker = {}

    fmt_dict = {'25': "ok", '10': "ob", '05': "og", '01': "or"}
    efmt_dict = {'25': ".k", '10': ".b", '05': ".g", '01': ".r"}
    ecolor_dict = {'25': "black", '10': "blue", '05': "green", '01': "red"}
    fig_dict = {" on IID": 1, " on Non IID": 2}

    for local_round in local_rounds:
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

                if dataset == "Shakespeare":
                    accuracy_runs = accuracy_runs * 100

                mean_accuracy_profile = np.mean(accuracy_runs, axis=0)
                std_dev_accuracy_profile = np.std(accuracy_runs, axis=0)
                max_accuracy_profile = np.max(accuracy_runs, axis=0)
                min_accuracy_profile = np.min(accuracy_runs, axis=0)

                final_accuracy_mean_tracker[f"{method}-{experiment}_{local_round}"] = mean_accuracy_profile[-1]
                final_accuracy_std_tracker[f"{method}-{experiment}_{local_round}"] = std_dev_accuracy_profile[-1]
                final_accuracy_max_tracker[f"{method}-{experiment}_{local_round}"] = max_accuracy_profile[-1]
                final_accuracy_min_tracker[f"{method}-{experiment}_{local_round}"] = min_accuracy_profile[-1]

    exp_matches = [" on IID", " on Non IID"]

    for exp_match in exp_matches:
        plt.figure(figsize=[8, 6])
        for local_round in local_rounds:
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
                if exp_match in key and key.endswith(local_round):
                    key_list.append(key.split('_')[0].replace(f"-{dataset}", "").replace(exp_match, ""))
                    final_acc_mean_list.append(final_accuracy_mean_tracker[key])
                    final_acc_std_list.append(final_accuracy_std_tracker[key])
                    final_acc_max_list.append(final_accuracy_max_tracker[key])
                    final_acc_min_list.append(final_accuracy_min_tracker[key])

                    plt.grid(True)

                    if dataset == "Shakespeare":
                        plt.errorbar(
                            np.arange(len(key_list)),
                            np.array(final_acc_mean_list),
                            np.array(final_acc_std_list),
                            fmt=fmt_dict[local_round],
                            label=int(local_round) if 'qFedAvg' in key and exp_match in key else None,
                            lw=3,
                        )
                    else:
                        plt.errorbar(
                            np.arange(len(key_list)),
                            np.array(final_acc_mean_list),
                            np.array(final_acc_std_list),
                            fmt=fmt_dict[local_round],
                            label=int(local_round) if f'qFedAvg-{dataset} CNN' in key and exp_match in key else None,
                            lw=3,
                        )
                    plt.errorbar(
                        np.arange(len(key_list)),
                        np.array(final_acc_mean_list),
                        [
                            np.array(final_acc_mean_list) - np.array(final_acc_min_list),
                            np.array(final_acc_max_list) - np.array(final_acc_mean_list),
                        ],
                        fmt=efmt_dict[local_round],
                        ecolor=ecolor_dict[local_round],
                        lw=1,
                    )

                    plt.xticks(np.arange(len(key_list)), key_list, rotation="45", ha='right')
                    # plt.title(f"Local Rounds {local_round}")
                    plt.xlabel("Algorithms")
                    plt.ylabel("Test Accuracy")

        plt.margins(0.1)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True)
        plt.tight_layout()
        plt.savefig(
            f"plots/{dataset}/Local_Rounds/Stacked_Error_Bar_Multiple/{IS_IID}/{dataset}{exp_match.split('-')[0]}.svg",
            format="svg",
            dpi=1000,
        )
        plt.show()


def plot_fairness(path, dataset, NUM_REPS):
    ROUNDS = 50
    NUM_CLIENTS = 100
    exp_matches = [" CNN on IID", " CNN on Non IID", " MLP on IID", " MLP on Non IID", " LSTM on IID", " LSTM on Non IID"]

    for exp_match in exp_matches:
        plt.figure(figsize=[8, 6])
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

                        if dataset == "Shakespeare":
                            final_accuracy = [accuracy * 100 for accuracy in final_accuracy]

                        method_name = pickle_file.split("/")[-1].split(".")[0].replace("Fairness_", "")

                        plt.grid(True)
                        plt.hist(final_accuracy, bins=20)

                        # plt.title(f"Fairness")
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

                    if dataset == "Shakespeare":
                        final_accuracy = [accuracy * 100 for accuracy in final_accuracy]

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
        plt.figure(figsize=[8, 6])
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
            if "MLP" in key:
                continue

            if key.endswith(exp_match):
                key_list.append(key.replace(f"-{dataset}", "").replace(exp_match, "").replace("uniform", "u").replace("weighted", "w"))
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

        plt.xticks(np.arange(len(key_list)), key_list, rotation="45", ha='right')
        # plt.title(f"Fairness")
        plt.xlabel("Algorithms")
        plt.ylabel("Entropy")

        plt.tight_layout()
        plt.savefig(f"plots/{dataset}/Fairness/Stacked_Error_Bar/{IS_IID}/{dataset}{exp_match}.svg", format="svg", dpi=1000)
        plt.show()


def get_fairness_histogram(path, dataset, NUM_REPS, exp_match):
    ROUNDS = 50
    NUM_CLIENTS = 100

    method_match = exp_match.split('-')[0].split('_')[0]
    method_exp_match = exp_match.split('-')[0]

    for method in methods:
        if method != method_match:
            continue

        pickle_files = glob.glob(f"{path}/{dataset}/{method}/*.pkl")
        if not pickle_files:
            continue

        for pickle_file in pickle_files:
            if method_exp_match not in pickle_file:
                continue

            with open(pickle_file, "rb") as file:
                log_dict = pickle.load(file)

            for experiment in log_dict.keys():
                if exp_match.endswith(experiment):
                    # if 'Non IID' in experiment:
                    #     IS_IID = 'Non_IID'
                    # else:
                    #     IS_IID = 'IID'
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

                    if dataset == "Shakespeare":
                        final_accuracy = [accuracy * 100 for accuracy in final_accuracy]

                    return final_accuracy


def plot_fairness_stacked_error_bar_plot_with_distribution(path, dataset, NUM_REPS):
    ROUNDS = 50
    NUM_CLIENTS = 100

    final_entropy_mean_tracker = {}
    final_entropy_std_tracker = {}
    final_entropy_max_tracker = {}
    final_entropy_min_tracker = {}
    histogram_tracker = {}

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

                    if dataset == "Shakespeare":
                        final_accuracy = [accuracy * 100 for accuracy in final_accuracy]

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

                method_name_shorted = method_name.replace("uniform", "u").replace("weighted", "w")
                histogram_tracker[f"{method_name_shorted}-{experiment}"] = get_fairness_histogram(path, dataset, NUM_REPS, f"{method_name}-{experiment}")
                final_entropy_mean_tracker[f"{method_name_shorted}-{experiment}"] = np.mean(entropy_runs, axis=0)
                final_entropy_std_tracker[f"{method_name_shorted}-{experiment}"] = np.std(entropy_runs, axis=0)
                final_entropy_max_tracker[f"{method_name_shorted}-{experiment}"] = np.max(entropy_runs, axis=0)
                final_entropy_min_tracker[f"{method_name_shorted}-{experiment}"] = np.min(entropy_runs, axis=0)

    exp_matches = [" on IID", " on Non IID"]

    for exp_match in exp_matches:
        plt.figure(figsize=[8, 6])
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
            if 'MLP' in key:
                continue

            if key.endswith(exp_match):
                key_list.append(
                    key.replace(f"-{dataset}", "").replace(exp_match, "").replace("uniform", "u").replace("weighted",
                                                                                                          "w"))
                final_acc_mean_list.append(final_entropy_mean_tracker[key])
                final_acc_std_list.append(final_entropy_std_tracker[key])
                final_acc_max_list.append(final_entropy_max_tracker[key])
                final_acc_min_list.append(final_entropy_min_tracker[key])

        plt.grid(True)
        plt.margins(0.1)

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

        plt.xticks(np.arange(len(key_list)), key_list, rotation="45", ha='right')

        ax = plt.gca()
        plt.gcf().canvas.draw()
        ticks = [tick for tick in plt.gca().get_xticklabels()]

        for i, t in enumerate(ticks):
            method = t.get_text().split()[0]
            model = t.get_text().split()[1]

            # bbox = t.get_window_extent().transformed(plt.gca().transData.inverted())

            # if dataset == "MNIST":
            #     ax_ins  = ax.inset_axes([i * 0.064 + 0.06, 0.58, 0.05, 0.25])
            # else:
            #     ax_ins  = ax.inset_axes([i * 0.1395 + 0.03, 0.55, 0.1, 0.25])

            # print(histogram_tracker[f"{method}-{dataset} {model}{exp_match}"])
            ax_ins  = ax.inset_axes([i * 0.1395 + 0.03, 0.55, 0.1, 0.25])
            ax_ins.hist(histogram_tracker[f"{method}-{dataset} {model}{exp_match}"], bins=25)
            # ax_ins.set_xticks([0, 100])
            ax_ins.xaxis.set_visible(False)
            ax_ins.yaxis.set_visible(False)

        # plt.title(f"Fairness")
        plt.xlabel("Algorithms")
        plt.ylabel("Entropy")

        plt.tight_layout()
        plt.savefig(f"plots/{dataset}/Fairness/Stacked_Error_Bar_With_Dist/{IS_IID}/{dataset}{exp_match}.svg", format="svg", dpi=1000)
        plt.show()


if __name__ == "__main__":
    plot_accuracy("./Local_Rounds/", "MNIST")
    plot_accuracy("./Local_Rounds/", "CIFAR")
    plot_accuracy("./Local_Rounds/", "Shakespeare")

    plot_accuracy_stacked_error_bar_plot("./Local_Rounds/", "MNIST")
    plot_accuracy_stacked_error_bar_plot("./Local_Rounds/", "CIFAR")
    plot_accuracy_stacked_error_bar_plot("./Local_Rounds/", "Shakespeare")

    plot_accuracy_stacked_error_bar_multiple_plot("./Local_Rounds/", "MNIST")
    plot_accuracy_stacked_error_bar_multiple_plot("./Local_Rounds/", "CIFAR")
    plot_accuracy_stacked_error_bar_multiple_plot("./Local_Rounds/", "Shakespeare")

    plot_fairness("./Fairness/", "MNIST", 5)
    plot_fairness("./Fairness/", "CIFAR", 5)
    plot_fairness("./Fairness/", "Shakespeare", 2)

    plot_fairness_stacked_error_bar_plot("./Fairness/", "MNIST", 5)
    plot_fairness_stacked_error_bar_plot("./Fairness/", "CIFAR", 5)
    plot_fairness_stacked_error_bar_plot("./Fairness/", "Shakespeare", 2)

    plot_fairness_stacked_error_bar_plot_with_distribution("./Fairness/", "MNIST", 5)
    plot_fairness_stacked_error_bar_plot_with_distribution("./Fairness/", "CIFAR", 5)
    plot_fairness_stacked_error_bar_plot_with_distribution("./Fairness/", "Shakespeare", 2)
