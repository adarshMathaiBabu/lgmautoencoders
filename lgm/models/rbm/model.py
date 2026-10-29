"""This module implements the RBM model core."""
from typing import override

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Distribution

from lgm.types import DataBatchFloat, ScalarFloat, TabularBatchFloat


class RBM(nn.Module):
    """Binary Restricted Boltzmann Machine."""
    @override
    def __init__(self,
                 original_input_shape: tuple[int, ...],
                 n_hidden: int,
                 chain_length: int,
                 start_sampler: Distribution | None = None):
        """Configure and initialize the module.
        
        Parameters:
            original_input_shape: RBMs work on vector data. This means images are flattened. This
                should give the actual shape of the images (or whatever structured data you are
                using), so we can reshape our flat generations to the appropriate shape.
            n_hidden: Number of hidden units to use.
            chain_length: Default length of Markov chains to sample with.
            start_sampler: A distribution we can sample from to start the Markov chains. This should
                be a sampler over `n_visible` values, i.e. taking a single `sample()` should return
                a vector of length `n_visible`.
        """
        super().__init__()
        self.original_input_shape = original_input_shape
        n_visible = np.prod(original_input_shape)
        self.chain_length = chain_length

        self.w_v_to_h = nn.Parameter(torch.zeros(n_visible, n_hidden))
        self.bias_v = nn.Parameter(torch.zeros(n_visible))
        self.bias_h = nn.Parameter(torch.zeros(n_hidden))
        if start_sampler is None:
            start_sampler = Bernoulli(probs=torch.full((n_visible,), fill_value=0.5))
        self.start_sampler = start_sampler
    
    def to_hidden_p(self,
                    visible: TabularBatchFloat) -> TabularBatchFloat:
        """Get conditional probabilities `p(h|v)`.
        
        Parameters:
            visible: `(B x V)` matrix of visible units.

        Returns:
            `(B x H)` matrix of probabilities for each hidden unit.
        """
        return nn.functional.sigmoid(visible @ self.w_v_to_h + self.bias_h)

    def to_visible_p(self,
                     hidden: TabularBatchFloat) -> TabularBatchFloat:
        """Get conditional probabilities `p(v|h)`.
        
        Parameters:
            hidden: `(B x H)` matrix of hidden units.
        
            Returns:
                `(B x V)` matrix of probabilities for each visible unit.
        """
        return nn.functional.sigmoid(hidden @ self.w_v_to_h.T + self.bias_v)

    def generate(self,
                 n_samples: int,
                 chain_length: int | None = None,
                 return_probs: bool = True,
                 return_chain: bool = False) -> DataBatchFloat | tuple[DataBatchFloat,
                                                                       list[TabularBatchFloat]]:
        """Create samples from the RBM distribution via Markov chains.
        
        Parameters:
            n_samples: How many independent chains to run (i.e. how many images do you want).
            chain_length: Markov chain length to override the object default. If not given, use the
                default.
            return_probs: If `True`, we return the visible probabilities for the last `p(v|h)`
                instead of sampling. This gives much smoother, less noisy samples.
            return_chain: If `True`, we also return the entire chain of visible samples as a list.
                Note that even if `return_probs` is `True`, this will contain the *binary samples*
                for the last step.

        Returns:
            Either visible samples, or tuple of visible samples and list of all visible samples in
            the Markov chain. Whether the first return value is binary samples or probabilities
            depends on `return_probs`, but the chain (for `return_chain=True`) always contains
            binary samples. Also, visible samples are returned in the correct data shape, whereas
            the chain return is left as vectors.
        """
        if chain_length is None:
            chain_length = self.chain_length

        initial_v = self.start_sampler.sample((n_samples,)).to(self.w_v_to_h.device)
        chain_outputs = self.gibbs_chain(initial_v,
                                         chain_length,
                                         return_v_probs=return_probs,
                                         return_chain=return_chain)
        
        final_v = chain_outputs[0].view(-1, *self.original_input_shape)
        if return_chain:
            return final_v, chain_outputs[2]
        else:
            return final_v

    def gibbs_chain(self,
                    initial_visible: TabularBatchFloat,
                    n_steps: int,
                    return_v_probs: bool = False,
                    return_chain: bool = False)\
                        -> tuple[TabularBatchFloat, TabularBatchFloat] | tuple[TabularBatchFloat,
                                                                               TabularBatchFloat,
                                                                               list[TabularBatchFloat]]:
        """Run Markov Chains using Gibbs Sampling.
        
        Parameters:
            initial_visible: `(B x D)` batch of chain starting points. We run `B` chains in
                parallel, returning one output per chain.
            n_steps: How many Gibbs Sampling steps to run. Note that one step means sampling from
                `p(h|v)` *and* then `p(v|h)`.
            return_v_probs: See `generate()`.
            return_chain: See `generate()`.

        Returns:
            Tuple of visible and hidden samples, and optionally a list of all visible samples as a
            third element. See notes regarding the return of the `generate()` function for some
            details.
        """
        new_v = initial_visible.clone()
        if return_chain:
            full_chain = [new_v]
        for _ in range(n_steps):
            new_v, new_h = self.gibbs_update(new_v)
            if return_chain:
                full_chain.append(new_v)

        if return_v_probs:
            new_v = self.to_visible_p(new_h)
        if return_chain:
            return new_v, new_h, full_chain
        else:
            return new_v, new_h

    def gibbs_update(self,
                     visible: TabularBatchFloat) -> tuple[TabularBatchFloat, TabularBatchFloat]:
        """A single Gibbs update step, returning both new `h` and `v` values.
        
        Parameters:
            visible: Previously sampled visible variables.

        Returns:
            Sample form `p(v|h_new)`, sample of `p(h|v_old)`.
        """
        raise NotImplementedError("Implement one step of block Gibbs sampling! This means: 1. Get "
            "p(h|v) conditioned on the visible input. This class already has a function for that! "
            "2. Sample random values from a Bernoulli distribution  using those probabilities. "
            "There is torch.distributions.Bernoulli. Check online docs for usage. 3./4. Repeat the "
            "same for p(v|h) (USE THE NEW H YOU JUST SAMPLED!) and sample new values for v.")
        p_h = ...
        new_h = ...
        p_v = ...
        new_v = ...
        return new_v, new_h

    def energy(self,
               visible: TabularBatchFloat,
               hidden: TabularBatchFloat) -> ScalarFloat:
        """RBM energy function, implemented for *batches* of `v` and `h`.
        
        Parameters:
            visible: Visible unit samples.
            hidden: Hidden unit samples.

        Returns:
            Mean energy over batch.
        """
        return -(visible @ self.bias_v
                 + hidden @ self.bias_h
                 + ((visible @ self.w_v_to_h) * hidden).sum(dim=1)).mean()
