import pickle
import copy
import numpy as np
# from scipy import signal
import pandas as pd
# import matplotlib.pyplot as plt

from sklearn import utils
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

import chainer
import chainer.functions as F
from chainer import Variable
from chainer import optimizers
from chainer import serializers
from chainer.backends import cuda

from wavenet.net import WaveNet


class Iterator(object):

    def __init__(self, base_model, optimizer):
        self.model = base_model
        self.optimizer = optimizer

    def train(self, inputs, targets, batch_size=24):
        n_inputs = inputs.shape[0]
        total_loss = 0
        total_accuracy = 0
        pred_y = None
        inputs_, targets_ = utils.shuffle(inputs, targets)
        for i in range(0, n_inputs, batch_size):
            x = Variable(cuda.to_gpu(inputs_[i:i+batch_size]))
            y = Variable(cuda.to_gpu(targets_[i:i+batch_size]))

            pred, loss = self.model(x, y)
            self.model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            self.optimizer.update()
            total_loss += loss.data * x.shape[0]
            accuracy = F.accuracy(pred[..., -1], y.data[..., -1]) * x.shape[0]
            total_accuracy += accuracy.data
            if i == 0:
                pred_y = cuda.to_cpu(pred.data[..., -1]).argmax(axis=1)
            else:
                pred_y = np.hstack((pred_y, cuda.to_cpu(pred.data[..., -1]).argmax(axis=1)))

        total_loss /= n_inputs
        total_accuracy /= n_inputs

        return cuda.to_cpu(total_accuracy), cuda.to_cpu(total_loss), pred_y, targets_[..., 0]

    def test(self, inputs, targets, batch_size=24):
        n_inputs = inputs.shape[0]
        total_loss = 0
        total_accuracy = 0
        total_any_time_accuracy = 0
        pred_y = None
        with chainer.using_config('train', False):
            for i in range(0, n_inputs, batch_size):
                x = Variable(cuda.to_gpu(inputs[i:i + batch_size]))
                y = Variable(cuda.to_gpu(targets[i:i + batch_size]))
                pred, loss = self.model(x, y)
                total_loss += loss.data * x.shape[0]
                accuracy = F.accuracy(pred[..., -1], y.data[..., -1]) * x.shape[0]
                total_accuracy += accuracy.data
                any_time_accuracy = []
                for k in range(pred.shape[2]):
                    k_accuracy = F.accuracy(pred[..., k], y.data[..., k]) * x.shape[0]
                    any_time_accuracy.append(cuda.to_cpu(k_accuracy.data))
                if i == 0:
                    total_any_time_accuracy = np.array(any_time_accuracy)
                    pred_y = cuda.to_cpu(pred.data[..., -1]).argmax(axis=1)
                else:
                    total_any_time_accuracy += np.array(any_time_accuracy)
                    pred_y = np.hstack((pred_y, cuda.to_cpu(pred.data[..., -1]).argmax(axis=1)))

            total_loss /= n_inputs
            total_accuracy /= n_inputs
            total_any_time_accuracy /= n_inputs

        return cuda.to_cpu(total_accuracy), cuda.to_cpu(total_loss), pred_y, total_any_time_accuracy


class CrossValidation(object):

    def __init__(self, model, optimizer, inputs, targets, cv=10, n_epoch=30):
        self.cv_models = [model.copy() for _ in range(cv)]
        self.cv_optimizers = [copy.copy(optimizer) for _ in range(cv)]
        self.inputs = inputs
        self.targets = targets
        self.cv = cv
        self.n_epoch = n_epoch

    def run(self, dir_name=None):
        skf = StratifiedKFold(self.cv, shuffle=True, random_state=1)
        skf.get_n_splits(self.inputs, self.targets)
        count = 1
        scores = []

        for (train_index, test_index), cv_model, cv_optimizer in zip(skf.split(self.inputs, self.targets[..., -1]),
                                                                     self.cv_models, self.cv_optimizers):
            x_train, x_test = self.inputs[train_index], self.inputs[test_index]
            y_train, y_test = self.targets[train_index], self.targets[test_index]

            cv_model = cv_model.to_gpu(0)
            score = {"train_accuracy": [], "train_loss": [], "test_accuracy": [], "test_loss": []}
            classification_reports = {"train_confusion_matrix": [], "train_classification_report": [],
                                      "test_confusion_matrix": [], "test_classification_report": []}

            any_time_accuracies = []
            max_accuracy = 0
            min_loss = 1000
            optimizer = cv_optimizer.setup(cv_model)
            # optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(1.0e-4))
            # optimizer.add_hook(chainer.optimizer_hooks.Lasso(1.0e-4))
            itr = Iterator(cv_model, optimizer)
            for i in range(self.n_epoch):
                train_accuracy, train_loss, train_pred, train_true = itr.train(x_train, y_train)
                test_accuracy, test_loss, test_pred, any_time_accuracy = itr.test(x_test, y_test)

                score["train_accuracy"].append(train_accuracy)
                score["train_loss"].append(train_loss)
                score["test_accuracy"].append(test_accuracy)
                score["test_loss"].append(test_loss)

                train_confusion_matrix = confusion_matrix(train_true, train_pred)
                test_confusion_matrix = confusion_matrix(y_test[..., -1], test_pred)
                train_classification_report = classification_report(train_true, train_pred)
                test_classification_report = classification_report(y_test[..., -1], test_pred)
                classification_reports["train_confusion_matrix"].append(train_confusion_matrix)
                classification_reports["train_classification_report"].append(train_classification_report)
                classification_reports["test_confusion_matrix"].append(test_confusion_matrix)
                classification_reports["test_classification_report"].append(test_classification_report)

                if max_accuracy <= test_accuracy:
                    if max_accuracy == test_accuracy:
                        if min_loss > test_loss:
                            serializers.save_hdf5("{}/max_acc_model_cv{}.h5".format(dir_name, count),
                                                  cv_model.to_cpu())
                            cv_model.to_gpu()
                    else:
                        serializers.save_hdf5("{}/max_accuracy_model_cv{}.h5".format(dir_name, count),
                                              cv_model.to_cpu())
                        cv_model.to_gpu()
                    max_accuracy = test_accuracy

                if min_loss > test_loss:
                    min_loss = test_loss
                    serializers.save_hdf5("{}/min_loss_model_cv{}.h5".format(dir_name, count),
                                          cv_model.to_cpu())
                    cv_model.to_gpu()

                any_time_accuracies.append(any_time_accuracy)
                print("cv: {} | epoch: {}|".format(count, i))
                print("accuracy/ train/ test/  | loss/ train/ test/")
                print("          {:4.3} {:4.3}   |       {:4.3} {:4.4}".format(train_accuracy, test_accuracy,
                                                                               train_loss, test_loss))
                h_split = np.array([[0] for _ in range(len(test_confusion_matrix))])
                print(np.hstack((train_confusion_matrix, h_split, test_confusion_matrix)))

            scores.append(score)
            df_score = pd.DataFrame(score)
            df_score.to_csv("{}/score_cv{}.csv".format(dir_name, count))
            with open("{}/classification_reports_cv{}.pickle".format(dir_name, count), "wb") as f:
                pickle.dump(classification_reports, f)
            np.save("{}/any_time_accuracy_cv{}".format(dir_name, count), np.array(any_time_accuracies))
            count += 1


def main():
    from datasets.eeg.load_eeg import LoadEEG
    eeg = LoadEEG()
    X, t = eeg.load("../datasets/eeg/eeg_data/assr1000_8000")
    X = X[..., 1000:6000].astype(np.float32)
    model = WaveNet(15, 15, 15, 15, 2)
    optimizer = optimizers.Adam(amsgrad=True)
    # optimizer = optimizers.NesterovAG()
    t2 = np.array([t for _ in range(X.shape[2])]).T
    crv = CrossValidation(model, optimizer, X, t2, n_epoch=50)
    crv.run("results")


if __name__ == "__main__":
    main()
