# -*- coding: utf-8 -*-
#
#  main.py
#  
#  Copyright 2022 Antoine Passemiers <antoine.passemiers@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.

import os
import sys
import math

import torch
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
import captum.attr

import nn_wrapper


SA_METHODS = {
    'Saliency': 'Saliency',
    'InputXGradient': 'Input x Gradient',
    'IG_noMul': 'Integrated gradient',
    'SmoothGrad': 'SmoothGrad',
    'GuidedBackprop': 'Guided backpropagation',
    'DeepLift': 'DeepLift',
    'Deconvolution': 'Deconvolution',
    'FeatureAblation': 'Feature ablation',
    'FeaturePermutation': 'Feature permutation',
    'ShapleyValueSampling': 'Shapley value sampling'
}
SA_METHOD_NAMES = list(SA_METHODS.keys())


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')
OUTPUT_FOLDER = os.path.join(ROOT, 'results')
if not os.path.isdir(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


def read_indices(dirName):
    files = os.listdir(dirName)
    folds = []
    for i, f in enumerate(files):
        ifp = open(dirName+f)
        lines = ifp.readlines()
        ifp.close()
        tmp = []
        for line in lines:
            tmp.append(int(float(line.strip())))
        folds.append(tmp)
    return folds


def parse_features(filepath):
    x = []
    y = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        tmp = line.strip().split(',')
        x.append([float(element) for element in tmp[:-1]])
        y.append(int(float(tmp[-1])))
    return x, y


def build_vectors(fold, xdb, ydb):
    x = []
    y = []
    for f in fold:
        x.append(xdb[f])
        y.append(ydb[f])
    x = 2. * (np.asarray(x) - 0.5)
    y = np.asarray(y)
    assert np.all(x >= -1)
    assert np.all(x <= 1)
    return x, y
    

def main_pred(input_filename, realFeatPos):
    print("#################################################", input_filename)
    folds = read_indices("indices/")
    n_folds = len(folds)
    xdb, ydb = parse_features(os.path.join(DATA_FOLDER, input_filename))
    i = 0
    auc_scores = []
    auprcs = []
    attrib_db = {}

    sms = {method_name: [[], []] for method_name in SA_METHOD_NAMES}
    while i < n_folds:
        test = folds.pop(0)
        train = [x for f in folds for x in f]
        X, Y = build_vectors(train, xdb, ydb)
        model = nn_wrapper.Model(len(X[0]))
        wrapper = nn_wrapper.NNwrapper(model)
        wrapper.fit(X, Y, "cpu", epochs=200)
        y_hat = wrapper.predict(X, "cpu")
        compute_scores(y_hat, Y, verbose=True)
        x, y = build_vectors(test, xdb, ydb)
        yp = wrapper.predict(x, "cpu")

        sen, spe, acc, bac, pre, mcc, auc_score, auprc = compute_scores(yp, y, verbose=True)
        auc_scores.append(auc_score)
        auprcs.append(auprc)
        folds.append(test)
        for method_name in SA_METHOD_NAMES:
            print("Working on ", method_name)
            if method_name not in attrib_db:
                attrib_db[method_name] = {"k": [], "2k": []}
            tmp_x = torch.FloatTensor(x)
            tmp_x.requires_grad_()
            baselines = torch.zeros((1, tmp_x.size()[-1]))
            if "IG_noMul" == method_name:
                ig = captum.attr.IntegratedGradients(model, multiply_by_inputs=False)
                attr = ig.attribute(tmp_x, target=0, return_convergence_delta=False, baselines=baselines)
            elif "Saliency" == method_name:
                ig = captum.attr.Saliency(model)
                attr = ig.attribute(tmp_x, target=0, abs=True)
            elif "DeepLift" == method_name:
                ig = captum.attr.DeepLift(model, multiply_by_inputs=False)
                attr = ig.attribute(tmp_x, target=0, return_convergence_delta=False, baselines=baselines)
            elif "InputXGradient" == method_name:
                ig = captum.attr.InputXGradient(model)
                attr = ig.attribute(tmp_x, target=0)
            elif "SmoothGrad" == method_name:
                ig = captum.attr.NoiseTunnel(captum.attr.Saliency(model))
                attr = ig.attribute(tmp_x, target=0, nt_samples=10)
            elif "GuidedBackprop" == method_name:
                ig = captum.attr.GuidedBackprop(model)
                attr = ig.attribute(tmp_x, target=0)
            elif "Deconvolution" == method_name:
                ig = captum.attr.Deconvolution(model)
                attr = ig.attribute(tmp_x, target=0)
            elif "FeatureAblation" == method_name:
                ig = captum.attr.FeatureAblation(model)
                attr = ig.attribute(tmp_x, target=0, baselines=baselines)
            elif "FeaturePermutation" == method_name:
                ig = captum.attr.FeaturePermutation(model)
                attr = ig.attribute(tmp_x, target=0)
            elif "ShapleyValueSampling" == method_name:
                ig = captum.attr.ShapleyValueSampling(model)
                attr = ig.attribute(tmp_x, target=0, baselines=baselines)
            else:
                raise NotImplementedError(method_name)

            attr = attr.detach().numpy().tolist()
            sms[method_name][0] += attr
            sms[method_name][1] += x.tolist()
            
            best_ks, best_2ks = compute_ks(attr, realFeatPos)
            attrib_db[method_name]['k'].append(np.mean(best_ks))
            attrib_db[method_name]['2k'].append(np.mean(best_2ks))
        i += 1
    return np.mean(auc_scores), np.mean(auprcs), attrib_db, sms


def compute_ks(attr, real_feat_pos):
    k = []
    k2 = []
    i = 0
    while i < len(attr):
        s = 0
        tmp = []
        while s < len(attr[i]):
            tmp.append((s, attr[i][s]))
            s += 1
        tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
        k.append(find_correct_feats(real_feat_pos, tmp, 2))
        k2.append(find_correct_feats(real_feat_pos, tmp, 4))
        i += 1
    return k, k2


def find_correct_feats(correct_indices, attr, pos):
    i = 0
    k = 0
    while i < min(pos, len(attr)):
        if attr[i][0] in correct_indices:
            k += 1
        i += 1
    return k / float(len(correct_indices))


def compute_scores(pred, real, threshold=None, verbose=False, curves=False, savefig=None):
    if type(pred[0]) == list or type(pred[0]) == np.ndarray:
        tmp = []
        for i in pred:
            if type(i) == np.ndarray:
                i = i.flatten().tolist()
            tmp += i
        pred = tmp
        tmp = []
        for i in real:
            tmp += i
        real = tmp
    if len(pred) != len(real):
        raise Exception("ERROR: input vectors have different lengths!")
    if verbose:
        print("Computing scores for %d predictions" % len(pred))
    
    if curves or (savefig is not None):
        precision, recall, thresholds = precision_recall_curve(real, pred)
        fpr, tpr, _ = roc_curve(real, pred)     
        fig, (ax1, ax2) = plt.subplots(figsize=[10.0, 5], ncols=2)
        ax1.set_ylabel("Precision")
        ax1.set_xlabel("Recall")
        ax1.set_title("PR curve")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.plot(recall, precision)
        ax1.grid()
        ax2.plot(fpr, tpr)
        ax2.set_ylim(0, 1)
        ax2.set_xlim(0, 1)
        ax2.plot([0, 1], [0, 1], '--', c='grey')
        ax2.set_xlabel('FPR')
        ax2.set_ylabel('TPR')
        ax2.set_title('ROC curve')
        ax2.grid()
        if savefig is not None:
            plt.savefig(savefig, dpi=400)
        plt.show()
        plt.clf()
        
    fpr, tpr, thresholds = roc_curve(real, pred)
    auprc = average_precision_score(real, pred)
    auc_score = auc(fpr, tpr)
    i = 0
    r = []
    while i < len(fpr):
        r.append((fpr[i], tpr[i], thresholds[i]))
        i += 1
    ts = sorted(r, key=lambda x: (1.0-x[0]+x[1]), reverse=True)[:3]
    if threshold is None:
        if verbose:
            print(f' > Best threshold: {str(ts[0][2])}')
        threshold = ts[0][2]
    confusion_matrix = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
    for i in range(len(real)):
        if float(pred[i]) <= threshold and (real[i] == 0):
            confusion_matrix['TN'] += 1
        if float(pred[i]) <= threshold and real[i] == 1:
            confusion_matrix['FN'] += 1
        if float(pred[i]) >= threshold and real[i] == 1:
            confusion_matrix['TP'] += 1
        if float(pred[i]) >= threshold and real[i] == 0:
            confusion_matrix['FP'] += 1
    if verbose:
        print("      | DEL         | NEUT             |")
        print("DEL   | TP: %d   | FP: %d  |" % (confusion_matrix["TP"], confusion_matrix["FP"] ))
        print("NEUT  | FN: %d   | TN: %d  |" % (confusion_matrix["FN"], confusion_matrix["TN"]))
    
    sen = (confusion_matrix["TP"]/max(0.00001, float((confusion_matrix["TP"] + confusion_matrix["FN"]))))
    spe = (confusion_matrix["TN"]/max(0.00001, float((confusion_matrix["TN"] + confusion_matrix["FP"]))))
    acc = (confusion_matrix["TP"] + confusion_matrix["TN"])/max(0.00001, float((sum(confusion_matrix.values()))))
    bac = (0.5*(confusion_matrix["TP"]/max(0.00001, float((confusion_matrix["TP"] + confusion_matrix["FN"])))+(confusion_matrix["TN"]/max(0.00001, float((confusion_matrix["TN"] + confusion_matrix["FP"]))))))
    pre = (confusion_matrix["TP"]/max(0.00001, float((confusion_matrix["TP"] + confusion_matrix["FP"]))))
    mcc = (((confusion_matrix["TP"] * confusion_matrix["TN"])-(confusion_matrix["FN"] * confusion_matrix["FP"])) / max(0.00001, math.sqrt((confusion_matrix["TP"]+confusion_matrix["FP"])*(confusion_matrix["TP"]+confusion_matrix["FN"])*(confusion_matrix["TN"]+confusion_matrix["FP"])*(confusion_matrix["TN"]+confusion_matrix["FN"]))) )
    
    if verbose:
        print("\nSen = %3.3f" % sen)
        print("Spe = %3.3f" % spe)
        print("Acc = %3.3f " % acc)
        print("Bac = %3.3f" % bac)
        print("Pre = %3.3f" % pre)
        print("MCC = %3.3f" % mcc)
        print("#AUC = %3.3f" % auc_score)
        print("#AUPRC= %3.3f" % auprc)
        print("--------------------------------------------")
    
    return sen, spe, acc, bac, pre, mcc, auc_score, auprc


def plot_score_interpretation(sms, boundaries='quadrants'):
    for i, method_name in enumerate(SA_METHOD_NAMES):
        if i >= 8:
            plt.subplot(3, 4, i + 2)
        else:
            plt.subplot(3, 4, i + 1)
        x = np.asarray(sms[method_name][1])
        data = np.abs(np.asarray(sms[method_name][0]))
        score = data[:, 0] / (np.sum(data, axis=1) + 1e-50)
        plt.scatter(x[:, 0], x[:, 1], s=15, c=score, alpha=0.5)
        plt.title(SA_METHODS[method_name], fontname='Century gothic')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if boundaries == 'quadrants':
            plt.axvline(x=0, linestyle='--', linewidth=0.5, color='black')
            plt.axhline(y=0, linestyle='--', linewidth=0.5, color='black')
        else:
            pass
        plt.set_cmap('Spectral')
    plt.subplot(3, 4, 12)
    plt.colorbar()
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.axis('off')
    plt.tight_layout()


def main(args):

    dataset_name = 'ring+xor+sum'
    if dataset_name == 'ring':
        filenames = [f'ring_1000samples-{size}feat.csv' for size in [2, 4, 8, 16, 32, 64, 128, 256, 512]]
        output_filename = 'resultsRING.txt'
    elif dataset_name == 'xor':
        output_filename = 'resultsXOR.txt'
        filenames = [f'xor_1000samples-{size}feat.csv' for size in [2, 4, 8, 16, 32, 64, 128, 256, 512]]
    elif dataset_name == 'ring+xor':
        output_filename = 'resultsRING+XOR.txt'
        filenames = [f'ring+xor_1000samples-{size}feat.csv' for size in [2, 4, 8, 16, 32, 64, 128, 256, 512]]
    elif dataset_name == 'ring+xor+sum':
        output_filename = 'resultsRING+XOR+SUM.txt'
        filenames = [f'ring-xor-sum_1000samples-{size}feat.csv' for size in [6, 8, 16, 32, 64, 128, 256, 512]]
    else:
        raise NotImplementedError(f'Unknown dataset "{dataset_name}"')
    
    filepath = os.path.join(OUTPUT_FOLDER, output_filename)
    with open(filepath, 'w') as ofp:
        ofp.write("Dataset\tAUC\tAUPRC\t")
        for method in SA_METHOD_NAMES:
            ofp.write(f'{method}_bestK\t{method}_best2K\t')
        ofp.write("\n")
        for filename in filenames:
            auc, auprc, attrDB, sms = main_pred(filename, [0, 1])
            ofp.write("%s\t%.3f\t%.3f\t" % (filename, auc, auprc))
            for n in SA_METHOD_NAMES:
                ofp.write("%.3f\t%.3f\t" % (np.mean(attrDB[n]["k"]), np.mean(attrDB[n]["2k"])))
            ofp.write("\n")

            if filename == 'xor_1000samples-2feat.csv':
                plot_score_interpretation(sms, boundaries='quadrants')
                plt.show()
            if filename == 'ring_1000samples-2feat.csv':
                plot_score_interpretation(sms, boundaries='quadrants')
                plt.show()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
