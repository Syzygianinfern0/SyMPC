# third party
import torch

from sympc.utils import parallel_execution

from .mpc_tensor import *  # why doesn't from .mpc_tensor import MPCTensor work?


def argmax(self, dim=None, keepdim=False, one_hot=False) -> "MPCTensor":
    """
    Compute argmax using pairwise comparisons. Makes the number of rounds fixed, here it is 2.
    This is inspired from CrypTen.
    Args:
        dim: compute argmax over a specific dimension
        keepdim: when one_hot is true, keep all the dimensions of the tensor
        one_hot: return the argmax as a one hot vector
    """
    # implementation pseudo code

    x = self.flatten() if dim is None and len(self.shape) > 1 else self
    # for each share in MPCTensor
    #   do the algorithm protrayed in paper (helper_argmax_pairwise)
    #   results in creating two matrices and substracting them
    args = [[share_ptr_tensor, dim] for share_ptr_tensor in self.share_ptrs]
    shares = parallel_execution(helper_argmax_pairwise, self.session.parties)(args)
    # then create an MPCTensor tensor based on this results per share
    # (we can do that bc substraction can be done in mpc fashion out of the box)
    x_pairwise = MPCTensor(shares=shares, session=self.session)
    # with the MPCTensor tensor we check what entries are postive
    # then we check what columns of M matrix have m-1 non-zero entries after comparison
    # (by summing over cols)
    pairwise_comparisons = x_pairwise >= 0
    # re-compute row_length
    _dim = -1 if dim is None else dim
    row_length = x.shape[_dim] if x.shape[_dim] > 1 else 2

    result = pairwise_comparisons.sum(0)
    result = result >= (row_length - 1)

    result = (
        result.reshape(self.shape) if dim is None and len(self.shape) > 1 else result
    )

    if not one_hot:
        result = result._one_hot_to_index(dim, keepdim)

    # we return a boolean vector with the same shape as the input.
    return result


# from syft < 0.3.0
def helper_argmax_pairwise(self, dim=None):
    dim = -1 if dim is None else dim
    row_length = self.size(dim) if self.size(dim) > 1 else 2

    # Copy each row (length - 1) times to compare to each other row
    a = self.expand(row_length - 1, *self.size())

    # Generate cyclic permutations for each row
    b = torch.stack([self.roll(i + 1, dims=dim) for i in range(row_length - 1)])

    return a - b
