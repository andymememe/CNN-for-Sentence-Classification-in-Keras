import argparse
import os
import matplotlib.pyplot as plt
import numpy as np


def getResult(f, opt, model_name):
    """
    Extracted result in log file.

    Parameters:
    f : file
        Log file
    model_name : string
        Model's name

    Returns:
    tacc : array
        Accuracy of training
    vacc : array
        Accuracy of validation
    """
    tacc = []
    vacc = []
    opt.write('======= ' + model_name + ' =======\n')
    for l in f.readlines():
        if l.startswith('#'):
            opt.write(l[2:] + '\n')
        elif l.startswith('Epoch'):
            continue
        else:
            data = l.split(' - ')
            tacc.append(float(data[2].split(': ')[1]))
            vacc.append(float(data[4].split(': ')[1]))
    return np.array(tacc), np.array(vacc)

parser = argparse.ArgumentParser(description="Plot result.")
parser.add_argument('result_folder', metavar='Result-Folder', nargs=1,
                    help='The folder you want to plot.')
args = parser.parse_args()
folder_name = args.result_folder[0]
opt = open('Result/' + folder_name + '-result.txt', 'w')

if os.path.isdir(folder_name) and os.path.exists(folder_name):
    for filename in os.listdir(folder_name):
        f = open(folder_name + '/' + filename, 'r')
        model_name = os.path.basename(f.name).split('.')[0].split('-')[1]
        tacc, vacc = getResult(f, opt, model_name)

        t_max_i = tacc.argmax()
        t_max = tacc.max()
        opt.write('Maximum accuracy in training => Epoch ' + str(t_max_i + 1) +
                  ' : ' + str(t_max) + '\n')

        v_max_i = vacc.argmax()
        v_max = vacc.max()
        opt.write('Maximum accuracy in validation => Epoch ' +
                  str(v_max_i + 1) + ' : ' + str(v_max) + '\n\n\n')

        plt.subplot(211)
        plt.plot(tacc, label=model_name)
        plt.subplot(212)
        plt.plot(vacc, label=model_name)
        f.close()
    plt.subplot(211)
    plt.legend(loc='lower right')
    plt.subplot(212)
    plt.legend(loc='lower right')
    fig = plt.gcf()
    fig.set_size_inches((10, 10))
    plt.savefig('Result/' + folder_name + '-accuracy.png')
    opt.close()
else:
    raise FileNotFoundError('No such folder : ' + folder_name)
