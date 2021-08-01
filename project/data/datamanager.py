import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Data_manager:
    ## DataManager would either get the extact data (array/tensors) or it'll have a df of filenames

    def __init__(self, base_data, base_labels, OOD_data, OOD_labels):
        self.base_data = base_data
        self.base_labels = base_labels
        self.OOD_data = OOD_data
        self.OOD_labels = OOD_labels
        self.log = {}
        self.iter = None
        self.config = {}

    def create_merged_data(self, test_size, pool_size, labelled_size, OOD_ratio):

        print("Creating New Dataset")

        assert 0 <= OOD_ratio < 1, "Invalid OOD_ratio : {OOD_ratio}"

        base_pool_size = pool_size

        assert (
            test_size + base_pool_size + labelled_size <= self.base_data.shape[0]
        ), f"Insufficient Samples in Base Dataset: test_size + unlabelled_pool_size + labelled_size > base_data_size : {test_size} + {base_pool_size} + {labelled_size} > {self.base_data.shape[0]}"

        train_data, self.test_data, train_labels, self.test_labels = train_test_split(
            self.base_data,
            self.base_labels,
            test_size=test_size,
            stratify=self.base_labels,
        )

        if base_pool_size > 0:

            (
                labelled_data,
                unlabelled_data,
                labelled_labels,
                unlabelled_labels,
            ) = train_test_split(
                train_data,
                train_labels,
                train_size=labelled_size,
                test_size=base_pool_size,
                stratify=train_labels,
            )

            data_list = [labelled_data, unlabelled_data]
            label_list = [labelled_labels, unlabelled_labels]
            status_list = [
                np.ones_like(labelled_labels),
                np.zeros_like(unlabelled_labels),
            ]
            source_list = [
                np.ones_like(labelled_labels),
                np.ones_like(unlabelled_labels),
            ]

        else:
            print("Running Experiment without Pool")
            if labelled_size <= len(train_data) - len(np.unique(train_labels)):
                (
                    labelled_data,
                    unlabelled_data,
                    labelled_labels,
                    unlabelled_labels,
                ) = train_test_split(
                    train_data,
                    train_labels,
                    train_size=labelled_size,
                    test_size=len(np.unique(train_labels)),
                    stratify=train_labels,
                )
            else:
                labelled_data = train_data[:labelled_size]
                labelled_labels = train_labels[:labelled_size]

            data_list = [labelled_data]
            label_list = [labelled_labels]
            status_list = [np.ones_like(labelled_labels)]
            source_list = [np.ones_like(labelled_labels)]

        if OOD_ratio > 0:
            OOD_size = int(pool_size * (1 / ((1 / OOD_ratio) - 1)))
            assert OOD_size < len(
                self.OOD_data
            ), f"Insufficient Samples in OOD Dataset : OOD_size > OOD_Dataset : {OOD_size} > {len(self.OOD_data)}"

            OOD_data, _, OOD_labels, _ = train_test_split(
                self.OOD_data,
                self.OOD_labels,
                train_size=OOD_size,
                stratify=self.OOD_labels,
            )
            data_list.append(OOD_data)
            label_list.append(-np.ones_like(OOD_labels))

            status_list.append(np.zeros_like(OOD_labels))
            source_list.append(-np.ones_like(OOD_labels))
        else:
            pass
        
        self.pool_data = np.concatenate(data_list)
        pool_labels = np.concatenate(label_list)
        pool_status = np.concatenate(status_list)
        pool_source = np.concatenate(source_list)

        self.status_manager = pd.DataFrame(
            np.concatenate(
                [
                    pool_labels[..., np.newaxis],
                    pool_source[..., np.newaxis],
                    pool_status[..., np.newaxis],
                ],
                axis=1,
            ),
            columns=["target", "source", "status"],
        )

        self.iter = 0

        self.config = {
            "Total_overall_examples": len(self.status_manager),
            "Total_base_examples": len(
                self.status_manager[self.status_manager["source"] > 0]
            ),
            "Total_OOD_examples": len(
                self.status_manager[self.status_manager["source"] < 0]
            ),
            "Initail_examples_labelled": len(
                self.status_manager[self.status_manager["status"] == 1]
            ),
        }

        self.log = {}

        return None

    def get_train_data(self):
        ## Returns all data that has been labelled so far

        assert (
            self.iter is not None
        ), "Dataset not initialized. Call create_merged_data()"

        labelled_mask = self.status_manager[self.status_manager["status"] > 0].index
        train_data = self.pool_data[labelled_mask]
        train_labels = self.status_manager.iloc[labelled_mask]["target"].values

        return train_data, train_labels

    def get_test_data(self):
        ## Returns all test data with labels

        assert (
            self.iter is not None
        ), "Dataset not initialized. Call create_merged_data()"

        return self.test_data, self.test_labels

    def get_unlabelled_pool_data(self):
        ## Returns all data in pool, None is returned instead of labels.

        assert (
            self.iter is not None
        ), "Dataset not initialized. Call create_merged_data()"

        unlabelled_mask = self.status_manager[self.status_manager["status"] == 0].index

        return (
            self.pool_data[unlabelled_mask],
            self.status_manager.iloc[unlabelled_mask]["target"].values,
        )

    def add_log(self, writer, oracle, dataset, metric, log_dict=None):
        self.iter += 1

        current_iter_log = {
            "Iteration": self.iter,
            "Base_examples_labelled": len(
                self.status_manager[self.status_manager["status"] > 1]
            ),
            "OOD_examples_labelled": len(
                self.status_manager[self.status_manager["status"] < 0]
            ),
            "Remaining_pool_samples": len(
                self.status_manager[self.status_manager["status"] == 0]
            ),
        }

        writer.add_scalars(f"{metric}/{dataset}/{oracle}/examples_labelled", current_iter_log, self.iter)
        
        if log_dict is not None:
            writer.add_scalars(f"{metric}/{dataset}/{oracle}/{metric}", log_dict, self.iter)
            current_iter_log.update(log_dict)

        self.log[self.iter] = current_iter_log

    def get_logs(self):
        log_df = pd.DataFrame.from_dict(self.log, orient="index").set_index(
            "Iteration"
        )
        for key in self.config.keys():
            log_df[key] = self.config[key]
        return log_df

    def reset_pool(self):
        self.log = {}
        self.iter = 0
        self.status_manager.loc[self.status_manager["status"] != 1, "status"] = 0
