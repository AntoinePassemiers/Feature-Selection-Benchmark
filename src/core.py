# -*- coding: utf-8 -*-
#
#  main.py
#
#  Copyright 2023 Antoine Passemiers <antoine.passemiers@gmail.com>
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

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import torch
import captum.attr

from src.fsnet import FSNet
from src.nn_wrapper import NNwrapper, Model
from src.relief import relief


SEED = 0xCAFE
np.random.seed(SEED)


def run_fa_method(wrapper, X_test: np.ndarray, method_name: str) -> np.ndarray:
    tmp_x = torch.FloatTensor(X_test)
    tmp_x.requires_grad_()
    baselines = torch.zeros((1, tmp_x.size()[-1]))
    if 'IG_noMul' == method_name:
        ig = captum.attr.IntegratedGradients(wrapper.model, multiply_by_inputs=False)
        attr = ig.attribute(tmp_x, target=0, return_convergence_delta=False, baselines=baselines)
    elif 'Saliency' == method_name:
        ig = captum.attr.Saliency(wrapper.model)
        attr = ig.attribute(tmp_x, target=0, abs=True)
    elif 'DeepLift' == method_name:
        ig = captum.attr.DeepLift(wrapper.model, multiply_by_inputs=False)
        attr = ig.attribute(tmp_x, target=0, return_convergence_delta=False, baselines=baselines)
    elif 'InputXGradient' == method_name:
        ig = captum.attr.InputXGradient(wrapper.model)
        attr = ig.attribute(tmp_x, target=0)
    elif 'SmoothGrad' == method_name:
        ig = captum.attr.NoiseTunnel(captum.attr.Saliency(wrapper.model))
        attr = ig.attribute(tmp_x, target=0, nt_samples=50, stdevs=0.1)
    elif 'GuidedBackprop' == method_name:
        ig = captum.attr.GuidedBackprop(wrapper.model)
        attr = ig.attribute(tmp_x, target=0)
    elif 'Deconvolution' == method_name:
        ig = captum.attr.Deconvolution(wrapper.model)
        attr = ig.attribute(tmp_x, target=0)
    elif 'FeatureAblation' == method_name:
        ig = captum.attr.FeatureAblation(wrapper.model)
        attr = ig.attribute(tmp_x, target=0, baselines=baselines)
    elif 'FeaturePermutation' == method_name:
        ig = captum.attr.FeaturePermutation(wrapper.model)
        attr = ig.attribute(tmp_x, target=0)
    elif 'ShapleyValueSampling' == method_name:
        ig = captum.attr.ShapleyValueSampling(wrapper.model)
        attr = ig.attribute(tmp_x, target=0, baselines=baselines)
    else:
        raise NotImplementedError(method_name)

    attr = attr.detach().numpy()
    attr = np.mean(np.abs(attr), axis=0)
    return attr


def run_fs_method(
        dataset_name,
        method_name,
        X_train,
        X_tilde_train,
        y_train,
        X_test,
        X_tilde_test,
        k,
        _2k=True
):
    y_train_hat = None
    y_hat = None
    scores = None
    scores2 = None
    n_classes = len(set(y_train))
    n_features = X_train.shape[1]
    if method_name == 'random':
        scores = np.random.rand(n_features)
        scores2 = np.random.rand(n_features)
    elif method_name == 'nn':
        wrapper = NNwrapper.create(dataset_name, n_features, n_classes)
        wrapper.fit(X_train, y_train)
        y_train_hat = wrapper.predict_proba(X_train)
        y_hat = wrapper.predict_proba(X_test)
    elif method_name == 'rf':
        clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED)
        clf.fit(X_train, y_train)
        scores = clf.feature_importances_
        scores2 = scores
        y_train_hat = clf.predict_proba(X_train)
        y_hat = clf.predict_proba(X_test)
    elif method_name == 'mi':
        scores = mutual_info_classif(X_train, y_train)
        scores2 = scores
    elif method_name == 'relief':
        scores = relief(X_train, y_train)
        scores2 = scores
    elif method_name == 'fsnet':
        n_selected = min(2 * k, n_features)
        fsnet = FSNet(Model(n_selected, n_classes), n_features, 30, n_selected, n_classes)
        fsnet.fit(X_train, y_train)
        y_train_hat = fsnet.predict(X_train)
        y_hat = fsnet.predict(X_test)
        scores = fsnet.get_feature_importances()
        scores2 = scores
    elif method_name == 'lassonet':
        import lassonet
        model = lassonet.LassoNetClassifier(
            hidden_dims=(32, 32),
            n_iters=(30, 30),
            batch_size=64,
            dropout=0,
            patience=10,
            lambda_start=3.2768,
            tol=0.9999,
            verbose=False)
        path = model.path(X_train, y_train, return_state_dicts=True)
        scores = model.feature_importances_.numpy()
        scores2 = scores
        val_losses = [save.val_loss for save in path]
        state_dict = path[np.argmin(val_losses)].state_dict
        model.load(state_dict)
        y_train_hat = model.predict_proba(X_train)
        y_hat = model.predict_proba(X_test)
    elif method_name == 'mrmr':
        """
        import pymrmr
        n_bins = 20
        Xy = np.empty((len(X_train), n_features + 1), dtype=int)
        Xy[:, 0] = y_train
        Xy[:, 1:] = np.round(X_train * n_bins).astype(int)  # Discretization
        column_names = ['y'] + [f'x{i}' for i in range(n_features)]
        df = pd.DataFrame(data=Xy, index=None, columns=column_names)

        feature_names = set(pymrmr.mRMR(df, 'MIQ', k))
        idx = np.asarray([int(feature_name[1:]) for feature_name in feature_names])
        scores = np.zeros(n_features, dtype=float)
        scores[idx] = 1

        if _2k:
            feature_names = set(pymrmr.mRMR(df, 'MIQ', 2 * k))
            if 'y' in feature_names:
                feature_names.remove('y')
            idx = np.asarray([int(feature_name[1:]) for feature_name in feature_names])
            scores2 = np.zeros(n_features, dtype=float)
            scores2[idx] = 1
        """
        print('Running MRMR')
        from pymrmre import mrmr
        X_df = pd.DataFrame(X_train, columns=[f'x{i}' for i in range(X_train.shape[1])])
        y_df = pd.DataFrame(y_train, columns=['y'])
        solutions = mrmr.mrmr_ensemble(features=X_df, targets=y_df, solution_length=k, solution_count=1)
        feature_names = solutions.iloc[0][0]
        idx = np.asarray([int(feature_name[1:]) for feature_name in feature_names])
        scores = np.zeros(n_features, dtype=float)
        scores[idx] = np.random.rand(len(idx))
        if _2k:
            solutions = mrmr.mrmr_ensemble(features=X_df, targets=y_df, solution_length=2*k, solution_count=1)
            feature_names = solutions.iloc[0][0]
            idx = np.asarray([int(feature_name[1:]) for feature_name in feature_names])
            scores2 = np.zeros(n_features, dtype=float)
            scores2[idx] = np.random.rand(len(idx))
    elif method_name == 'cae':
        from concrete_autoencoder import ConcreteAutoencoderFeatureSelector
        import keras
        def nn(x):
            n_out = 1 if (n_classes <= 2) else n_classes
            x = keras.layers.GaussianNoise(0.1)(x)
            x = keras.layers.Dense(32)(x)
            x = keras.layers.Dropout(0.2)(x)
            x = keras.layers.LeakyReLU(alpha=0.2)(x)
            x = keras.layers.Dense(32)(x)
            x = keras.layers.Dropout(0.2)(x)
            x = keras.layers.LeakyReLU(alpha=0.2)(x)
            x = keras.layers.Dense(n_out, activation='sigmoid')(x)
            return x

        selector = ConcreteAutoencoderFeatureSelector(
            K=k, output_function=nn, start_temp=10, min_temp=0.01, num_epochs=300,
            learning_rate=0.0001, tryout_limit=1)
        selector.fit(X_train, y_train)
        indices = selector.get_support(indices=True).flatten()
        scores = np.zeros(n_features, dtype=float)
        scores[indices] = 1
        selector = ConcreteAutoencoderFeatureSelector(
            K=2 * k, output_function=nn, start_temp=10, min_temp=0.01, num_epochs=30,
            learning_rate=0.0001, tryout_limit=1)
        selector.fit(X_train, y_train)
        y_train_hat = selector.get_params().predict(X_train)
        y_hat = selector.get_params().predict(X_test)
        indices = selector.get_support(indices=True).flatten()
        scores2 = np.zeros(n_features, dtype=float)
        scores2[indices] = 1
    elif method_name == 'canceloutsigmoid':
        wrapper = NNwrapper.create(dataset_name, n_features, n_classes, arch='cancelout-sigmoid')
        wrapper.fit(X_train, y_train)
        y_train_hat = wrapper.predict_proba(X_train)
        y_hat = wrapper.predict_proba(X_test)
        scores = wrapper.model.cancel_out.get_weights().data.numpy()
        scores2 = scores
    elif method_name == 'canceloutsoftmax':
        wrapper = NNwrapper.create(dataset_name, n_features, n_classes, arch='cancelout-softmax')
        wrapper.fit(X_train, y_train)
        y_train_hat = wrapper.predict_proba(X_train)
        y_hat = wrapper.predict_proba(X_test)
        scores = wrapper.model.cancel_out.get_weights().data.numpy()
        scores2 = scores
    elif method_name == 'deeppink':
        X_augmented = np.empty((X_train.shape[0], X_train.shape[1], 2))
        X_augmented[:, :, 0] = X_train
        X_augmented[:, :, 1] = X_tilde_train
        wrapper = NNwrapper.create(dataset_name, len(X_train[0]), n_classes, arch='deeppink')
        wrapper.fit(X_augmented, y_train)
        y_train_hat = wrapper.predict_proba(X_augmented)
        X_augmented = np.empty((X_test.shape[0], X_test.shape[1], 2))
        X_augmented[:, :, 0] = X_test
        X_augmented[:, :, 1] = X_tilde_test
        y_hat = wrapper.predict_proba(X_augmented)
        scores = wrapper.model.get_weights()
        scores -= scores.min()
        scores2 = scores
    elif method_name == 'treeshap':
        import shap
        clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED)
        clf.fit(X_train, y_train)
        explainer = shap.TreeExplainer(clf)
        shap.initjs()
        X = np.concatenate((X_train, X_test), axis=0)
        scores = explainer.shap_values(X, approximate=False)[1]
        scores = np.mean(np.abs(scores), axis=0)
        scores2 = scores
        y_train_hat = clf.predict_proba(X_train)
        y_hat = clf.predict_proba(X_test)
    else:
        raise NotImplementedError(f'Unknown FS method "{method_name}"')

    if n_classes > 2:
        if (y_hat is not None) and (y_train_hat is not None):
            assert y_hat.shape[1] == n_classes
            assert y_train_hat.shape[1] == n_classes
    return y_train_hat, y_hat, scores, scores2
