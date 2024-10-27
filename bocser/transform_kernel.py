import gpflow
import tensorflow as tf

from typing import Callable, Union

class TransformKernel(gpflow.kernels.Kernel):
    """
        The transform kernel. Requieres transform function and base kernel

            k(x, y) = base_kernel(f(x), f(y))

        where base_kernel - another kernel function.
    """
    def __init__(
        self,
        f : Callable[[tf.Tensor], tf.Tensor],
        base_kernel : gpflow.kernels.Kernel,
    ) -> None:
        super().__init__()
        
        self.f = f
        self.base_kernel = base_kernel
    
    def K(
        self,
        X1 : tf.Tensor,
        X2 : Union[tf.Tensor, None] = None,
    ) -> tf.Tensor:
        
        if X2 is None:
            X2 = X1
        
        return self.base_kernel.K(
            self.f(X1), 
            self.f(X2)
        )
    
    def K_diag(
        self, 
        X : tf.Tensor
    ) -> tf.Tensor:
        return self.base_kernel.K_diag(
            self.f(X)
        )
