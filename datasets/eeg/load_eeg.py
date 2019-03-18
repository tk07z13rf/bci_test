import os
import numpy as np
import pandas as pd


class LoadEEG(object):

    def __init__(self, ch=None, header=None):

        self.analysis_ch = ch
        self.header = header
        self.data = None
        self.targets = None

        if self.analysis_ch is None:
            self.analysis_ch = ['Fp2', 'F4', 'C4', 'P4', 'O2', 'Fp1', 'F3', 'C3', 'P3', 'O1',
                                'Fz', 'Cz', 'Pz', 'T4', 'T3']

        if self.header is None:
            self.header = ['Fp2', 'F4', 'C4', 'P4', 'O2', 'Fp1', 'F3', 'C3', 'P3', 'O1',
                           'Fz', 'Cz', 'Pz', 'T4', 'T3', 'A']

    def open_with_pandas(self, filename):

        df = pd.read_csv(filename, delim_whitespace=True, header=None, names=self.header)
        data = df.ix[:, self.analysis_ch]
        data = data.values
        return data

    def make_eeg_dataset(self, paths, tags, t1=1000, t2=8000):
        files = []
        for path in paths:
            filename = os.listdir("eeg_data/" + path)
            for f in filename:
                files.append("eeg_data/" + path + f)

        x, target = [], []
        for f in files:
            for i, _ in enumerate(tags):
                if f.count(_ + ".") > 0 or f.count(_ + "n.") > 0 or f.count("_" + _) > 0:
                    target.append(i)
                    data = self.open_with_pandas(f).T
                    if t1 == 0:
                        data = data[:, :t2]
                    else:
                        data = data[:, t1-1000:t2-1000] if data.shape[1] < 8000 else data[:, t1:t2]
                    x.append(data)

            self.data = np.array(x, dtype=np.float32)
            self.targets = np.array(target)

    def save(self, dir_name):
        os.makedirs(dir_name)
        np.save("{}/data.npy".format(dir_name), self.data)
        np.save("{}/target.npy".format(dir_name), self.targets)

    def load(self, dir_name):
        self.data = np.load("{}/data.npy".format(dir_name))
        self.targets = np.load("{}/target.npy".format(dir_name))
        return self.data, self.targets


if __name__ == "__main__":
    eeg = LoadEEG()
    X, y = eeg.load("eeg_data/assr1000_8000")
    print(X.shape)
    print(y.shape)
