import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


from trieste.types import TensorType
from trieste.data import Dataset

from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition import ExpectedImprovement
from trieste.acquisition.interface import (
    AcquisitionFunction,
    AcquisitionFunctionClass,
    SingleModelAcquisitionBuilder
)
from trieste.models import ProbabilisticModel

from typing import cast

class ExplorationalVarianceMinimizer(SingleModelAcquisitionBuilder):
    """
        Returns Explorational Variance Minimizer acquisition fucntion value EVM(x) = max(\eta + threshold - f(x), 0)
    """
    def __init__(self, threshold):

        self._threshold = threshold
 
    def __repr__(self) -> str:
        """"""
        return "ExplorationalVarianceMinimizer()"
    
    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Dataset = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :return: The explorational_variance_minimizer. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        :raise tf.errors.InvalidArgumentError: If ``dataset`` is empty.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)
        return explorational_variance_minimizer(model, eta, dataset, self._threshold)
    
    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Dataset = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer.  Must be populated.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(isinstance(function, explorational_variance_minimizer), [])
        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)
        function.update(eta, dataset, self._threshold)  # type: ignore
        return function
    
    
class explorational_variance_minimizer(AcquisitionFunctionClass):
    def __init__(self, 
                 model: ProbabilisticModel, 
                 eta: TensorType, 
                 dataset : Dataset, 
                 threshold : float):
        """"""
        self._model = model
        self._eta = tf.Variable(eta)
        self._dataset = dataset
        self._threshold = threshold

    def update(self, 
               eta: TensorType, 
               dataset : Dataset, 
               threshold : float) -> None:
        """Update the acquisition function with a new eta value, dataset, threshold"""
        self._eta.assign(eta)
        self._dataset = dataset
        self._threshold = threshold

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        mean, variance = self._model.predict(tf.squeeze(x, -2))
        normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
        tau = self._eta + self._threshold
        return normal.cdf(tau) * (((tau - mean)**2) * (1 - normal.cdf(tau)) + variance) + tf.sqrt(variance) * normal.prob(tau) *\
                (tau - mean) * (1 - 2*normal.cdf(tau)) - variance * (normal.prob(tau)**2)
