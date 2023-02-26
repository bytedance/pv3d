# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

# Copyright (c) Facebook, Inc. and its affiliates.


import uuid
import logging
from lib.utils.logger import log_class_usage
from lib.common.build import build_dataloader_and_sampler


logger = logging.getLogger(__name__)


class BaseDatasetBuilder():
    """Base class for implementing dataset builders. See more information
    on top. Child class needs to implement ``build`` and ``load``.

    Args:
        dataset_name (str): Name of the dataset passed from child.
    """

    def __init__(self, dataset_name, *args, **kwargs):

        if dataset_name is None:
            # In case user doesn't pass it
            dataset_name = f"dataset_{uuid.uuid4().hex[:6]}"
        self.dataset_name = dataset_name
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

        log_class_usage("DatasetBuilder", self.__class__)

    @property
    def dataset_name(self):
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, dataset_name):
        self._dataset_name = dataset_name

    def build(self, config):
        self.config = config
        self.train_dataset = self.load_dataset(config, "train")
        self.val_dataset = self.load_dataset(config, "val")
        self.test_dataset = self.load_dataset(config, "test")

    @property
    def train_dataset(self):
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, dataset):
        self._train_dataset = dataset

    @property
    def val_dataset(self):
        return self._val_dataset

    @val_dataset.setter
    def val_dataset(self, dataset):
        self._val_dataset = dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    @test_dataset.setter
    def test_dataset(self, dataset):
        self._test_dataset = dataset

    def load_dataset(self, config, dataset_type="train", *args, **kwargs):
        """Main load function, this will internally call ``load``
        function. Calls ``init_processors`` and ``try_fast_read`` on the
        dataset returned from ``load``

        Args:
            config (DictConfig): Configuration of this dataset loaded from config.
            dataset_type (str): Type of dataset, train|val|test

        Returns:
            dataset (BaseDataset): Dataset containing data to be trained on

        .. warning::

            DO NOT OVERRIDE in child class. Instead override ``load``.
        """
        dataset = self.load(config, dataset_type, *args, **kwargs)
        if dataset is not None and hasattr(dataset, "init_processors"):
            # Checking for init_processors allows us to load some datasets
            # which don't have processors and don't inherit from BaseDataset
            dataset.init_processors()
        return dataset

    def load(self, config, dataset_type="train", *args, **kwargs):
        """
        This is used to prepare the dataset and load it from a path.
        Override this method in your child dataset builder class.

        Args:
            config (DictConfig): Configuration of this dataset loaded from config.
            dataset_type (str): Type of dataset, train|val|test

        Returns:
            dataset (BaseDataset): Dataset containing data to be trained on
        """
        raise NotImplementedError(
            "This dataset builder doesn't implement a load method"
        )

    @classmethod
    def config_path(cls):
        return None

    def build_dataloader(
        self, dataset_instance, dataset_type: str, *args, **kwargs
    ):
        if dataset_instance is None:
            logger.info(
                f"Dataset instance for {self.dataset_name} {dataset_type}set hasn't been set and is None"
            )
            return None
            
        dataloader = build_dataloader_and_sampler(dataset_instance, self.config)
        return dataloader

    def train_dataloader(self, *args, **kwargs):
        return self.build_dataloader(self.train_dataset, "train")

    def val_dataloader(self, *args, **kwargs):
        return self.build_dataloader(self.val_dataset, "val")

    def test_dataloader(self, *args, **kwargs):
        return self.build_dataloader(self.test_dataset, "test")