# python调用并回显linux命令
import os

import numpy as np


def linux_command(command):
    try:
        with os.popen(command, "r") as p:
            command_return=p.readlines()
    except:
        print(command)
    return command_return


def hostname():
    return linux_command("hostname")[0].replace("\n", "")

def poly_fitting(x, y, fitting_order):
    _x = np.array(x).copy()
    _y = np.array(y).copy()

    # clf = LOF(n_neighbors=2)
    # predict = clf.fit_predict(np.concatenate((_x[np.newaxis, ...], _y[np.newaxis, ...]), axis=0).T)
    # outer = np.where(predict == -1)
    # coef, cov = np.polyfit(np.delete(_x, outer), np.delete(_y, outer), fitting_order, cov=True)

    # skip LOF
    coef, cov = np.polyfit(_x, _y, fitting_order, cov=True)
    outer = [np.array([])]

    return [coef, np.sqrt(cov.diagonal()), outer]