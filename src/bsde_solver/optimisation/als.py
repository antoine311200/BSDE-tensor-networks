

class ALS:
    """Alternting Least Scheme for Tensor train format.

    This implementation follows the work of Thorsten Rohwedder and Reinhold Schneider
    in their paper "The Alternating Linear Scheme for Tensor Optimization in the Tensor Train Format", 2011.
    """

    def __init__(self, tensor_train):
        """Initialize the ALS algorithm with a given tensor train.

        Args:
            tensor_train (TensorTrain): Tensor train to optimize.
        """
        self.tensor_train = tensor_train
        self.tensor_train.orthonormalize(orthogonality="right", start=1)