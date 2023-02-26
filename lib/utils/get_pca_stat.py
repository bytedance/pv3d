# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

# Copyright (c) Facebook, Inc. and its affiliates.


import os
import numpy as np
import torch
import omegaconf
from tqdm import tqdm
from collections import defaultdict
from sklearn.decomposition import IncrementalPCA
from lib.common.build import build_model
from lib.utils.env import setup_imports
from lib.utils.options import options
from lib.utils.logger import setup_logger
from lib.utils.configuration import Configuration, get_global_config


class IPCAEstimator():
    def __init__(self, n_components):
        self.n_components = n_components
        self.whiten = False
        self.transformer = IncrementalPCA(n_components,
                                          whiten=self.whiten,
                                          batch_size=max(
                                              100, 2 * n_components))

    def fit(self, X):
        self.transformer.fit(X)

    def get_components(self):
        stdev = np.sqrt(self.transformer.explained_variance_)  # already sorted
        var_ratio = self.transformer.explained_variance_ratio_
        return self.transformer.components_, stdev, var_ratio  # PCA outputs are normalized


def style_code_sampler(shape, dist='gaussian'):
    if dist == 'gaussian':
        z = torch.randn(shape)
    elif dist == 'uniform':
        z = torch.rand(shape) * 2 - 1
    return z


def get_style_code(batch_size, nerf_z_dim, inr_z_dim):
    z_nerf = style_code_sampler(shape=(batch_size, nerf_z_dim))
    z_inr = style_code_sampler(shape=(batch_size, inr_z_dim))
    zs = {
        'z_nerf': z_nerf,
        'z_inr': z_inr,
    }
    return zs


def main():

    setup_imports()

    batchSize = 4000
    pca_iterations = 250
    dataset = "ffhq_256"
    save_pca_path = f"save/pca_stats/{dataset}"

    parser = options.get_parser()
    args = parser.parse_args()

    configuration = Configuration(args)
    configuration.args = args
    config = configuration.get_config()

    setup_logger(
        color=config.training.colored_logs, disable=config.training.should_not_log
    )

    print("Loading model")
    if config.model in config.model_config:
        attributes = config.model_config[config.model]
    else:
        raise RuntimeError(
            f"Model {config.model}'s config not present. "
            + "Continuing with empty config"
        )
    # Easy way to point to config for other model
    if isinstance(attributes, str):
        attributes = config.model_config[attributes]

    with omegaconf.open_dict(attributes):
        attributes.model = config.model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(attributes)
    model = model.generator.to(device)

    style_list = defaultdict(list)

    with torch.no_grad():
        for _ in tqdm(range(pca_iterations)):
            z = get_style_code(batchSize, 
                    config.model_config[config.model].generator.mapping_nerf_cfg.z_dim, 
                    config.model_config[config.model].generator.mapping_inr_cfg.z_dim ) 
            z = { k:v.to(device) for k,v in z.items() }
            styles = model.mapping_network(**z)
            for k, v in styles.items():
                style_list[k].append(v.cpu().numpy())

        comp, stdev, var_ratio, style_mean, style_var = {}, {}, {}, {}, {}

        for k in tqdm(style_list):
            styles_all = np.concatenate(style_list[k], axis=0)
            _style_mean = np.mean(styles_all, axis=0)
            _style_var = np.var(styles_all, axis=0)
            style_mean[k], style_var[k] = _style_mean, _style_var
            pca = IPCAEstimator(n_components=styles_all.shape[-1])
            pca.fit(styles_all)
            _comp, _stdev, _var_ratio = pca.get_components()
            comp[k], stdev[k], var_ratio[k] = _comp, _stdev, _var_ratio

        os.makedirs(save_pca_path, exist_ok=True)
        np.save(os.path.join(save_pca_path, f'pca_comp.npy'), comp)
        np.save(os.path.join(save_pca_path, f'pca_stdev.npy'), stdev)
        np.save(os.path.join(save_pca_path, f'pca_var_ratio.npy'), var_ratio)
        np.save(os.path.join(save_pca_path, f'style_mean.npy'), style_mean)
        np.save(os.path.join(save_pca_path, f'style_var.npy'), style_var)


if __name__ == "__main__":
    main()