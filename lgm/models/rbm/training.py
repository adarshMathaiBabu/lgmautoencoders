"""This module implements a lightning module for training a simple RBM."""
from typing import Literal, override

import torch
from torch.distributions import Bernoulli

from lgm.models import TrainingModuleBase
from lgm.optimizers import OptimizationConfig
from lgm.types import DataBatchFloat, LabelBatch, ScalarFloat, TabularBatchFloat
from .model import RBM


class RBMTrainingModule(TrainingModuleBase[RBM]):
    """Lightning module for a binary Restricted Boltzmann Machine.
    
    This follows the training structure using the positive & negative phase we learned about. We
    combine both into a "loss function" that can be minimized, although the actual goal is *not* to
    minimize it arbitrarily; while it can technically got to `-inf`, the goal is to reach 0, where
    positive and negative phase are perfectly balanced.  
    It is recommended to add a `GenerationLogging` callback for subjective evaluation of generation
    quality.
    """
    @override
    def __init__(self,
                 model: RBM,
                 optimization_config: OptimizationConfig,
                 chain_mode: Literal["simple", "cd", "pcd"],
                 chain_length: int | None = None,
                 marginal_chain_init: bool = False):
        """Configure the module.

        Parameters:
            model: `RBM` instance.
            optimization_config: See `TrainingModuleBase`.
            chain_mode: One of `'simple'`, `'cd'`, or `'pcd'`.  
                `'simple'` starts a new chain from random samples at each training step.  
                `'cd'` starts from the data samples.  
                `'pcd'` starts a batch of chains at the start of training, and keeps running those.
            chain_length: How many steps to sample for each chain (training step negative phase). It
                can be useful to have a different number here than for the model. To be concrete,
                the point of CD and PCD sampling is to get away with shorter chains. However, once
                you actually want to sample from the model, you need to start fresh chains. Thus,
                you'll probably want longer chains for sampling from the model than for training. If
                `None` is passed here, training will use the model's `chain_length`.
            marginal_chain_init: If `True`, random Markov chain initialization is done using dataset
                statistics -- for each dimension (e.g. pixel position), the probability to sample a
                1 is set to the average value for this pixel over the training dataset. If `False`,
                each dimension is sampled as 1 with probability 0.5. NOTE that this affects
                generations, and when later reloading the model in a different application, you
                likely want to make sure to use the same kind of initialization!
        """
        super().__init__(model=model,
                         optimization_config=optimization_config)
        self.chain_mode = chain_mode
        self.chain_length = chain_length if chain_length is not None else model.chain_length
        self.marginal_chain_init = marginal_chain_init

        self.previous_v = None  # used for PCD only

    @override
    def forward(self,
                inputs: TabularBatchFloat) -> TabularBatchFloat:
        raise NotImplementedError("forward() is not well-defined for RBMs!")

    @override
    def SOMETHING(self,
                  tag: Literal["train", "val", "test"],
                  batch: tuple[DataBatchFloat, LabelBatch],
                  batch_index: int) -> ScalarFloat:
        """Compute RBM "loss" via positive phase (data samples) and negative phase (model samples).
        
        By analyzing the RBM objective, you can see that it comes down to maximizing (positive 
        - negative) phases, which in turn is equivalent to minimizing the negative of that.

        Parameters:
            See: `TrainingModuleBase`.

        Returns:
            Scalar "Loss" for the batch.
        """
        raise NotImplementedError("Oh no! This function is not named correctly. Which function do "
            "we need to implement to get a functioning training loop? Check the base class and/or "
            "the autoencoder implementation for clues!")
        input_batch, _ = batch
        v_data = torch.flatten(input_batch, start_dim=1)

        # this part gets a batch of model samples
        with torch.no_grad():
            h_probs = self.model.to_hidden_p(v_data)
            h_data = Bernoulli(probs=h_probs).sample().to(v_data.device)
            
            # gibbs sampling -- naive/cd/pcd only differ in how the chains are initialized
            v_old = self.start_chain(v_data)
            v_new, h_new = self.model.gibbs_chain(v_old, self.chain_length)
            if self.chain_mode == "pcd":
                self.previous_v = v_new
    
        # although this loss can take any value (real number), it should converge to around 0
        # if it goes to -infinity there is something wrong!
        # most likely, your sampling is bad (e.g. not enough steps -> use longer chains)
        logits_positive = -self.model.energy(v_data, h_data)
        logits_negative = -self.model.energy(v_new, h_new)
        raise NotImplementedError("Given the positive/negative phase computed above, what is the "
            "overall 'loss' to minimize? Recall the derivation from the exercise, where we combined "
            "both parts into one term (also shown in the associated notebook)!")
        loss = ...

        raise NotImplementedError("We are missing logging of the metrics as well as the appropriate "
            "return value. How does logging work in our base class and what should be returned? You "
            "can also check the autoencoder.training implementation to see how it's done there. Be "
            "sure to use a descriptive name for the loss metric so you can find it in Tensorboard!")
        return None  # TODO

    @override
    def predict_step(self,
                     batch: tuple[DataBatchFloat, LabelBatch],
                     batch_index: int) -> DataBatchFloat:
        raise NotImplementedError("No predict_step since forward() is not well-defined for RBMs!")

    def start_chain(self,
                    v_data: TabularBatchFloat) -> TabularBatchFloat:
        """Produce a starting point for a Markov chain, depending on the training algorithm used.

        Parameters:
            v_data: Visible data used as a starting point for CD, or to get the shape for random
                samples.

        Returns:
            Batch of visible data samples to start the chain with.
        """
        if self.chain_mode == "cd":
            raise NotImplementedError("How are Markov chains started in Contrastive Divergence?")
            return None
        elif self.chain_mode == "pcd" and self.previous_v is not None:
            raise NotImplementedError("How are Markov chains started in *Persistent* Contrastive "
                "Divergence?")
            return None
        else:  # simple OR first PCD step
            return self.model.start_sampler.sample((v_data.shape[0],)).to(v_data.device)

    @override
    def on_fit_start(self):
        if self.marginal_chain_init:
            pixel_averages = self.trainer.datamodule.get_statistics(mode="dim")[0]
            flat_probs = pixel_averages.view(-1)
            self.model.start_sampler = Bernoulli(probs=flat_probs)
