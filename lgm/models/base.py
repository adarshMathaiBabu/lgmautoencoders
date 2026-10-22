"""Training code that is not specific to any model type.

This base module will cover almost all use cases in this class. You essentially only need to
implement `core_step()`, which should compute and log all component losses, and return the overall
loss used for training. This, in turn, usually requires applying the model. The only exception to
this are GANs and models using GAN components -- their slightly more complex training procedure
requires using "manual" optimization, which means overriding `training_step()`.
"""
from abc import ABC, abstractmethod
from typing import Generic, Literal, override

from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from lgm.optimizers import OptimizationConfig
from lgm.types import DataBatchFloat, LabelBatch, Model, ScalarFloat


class TrainingModuleBase(LightningModule, Generic[Model], ABC):
    """Base lightning module class bundling a bunch of boilerplate.
    
    Almost all models we build in this class will sit on top of this structure. As such, make sure
    you can understand what is going on here!
    """
    @override
    def __init__(self,
                 model: Model,
                 optimization_config: OptimizationConfig):
        """Bare minimum module configuration.

        Parameters:
            model: The actual model (`nn.Module`) we want to train.
            optimization_config: `OptimizationConfig` instance to configure optimizer and learning
                rate schedule.
        """
        super().__init__()
        self.model = model
        self.optimization_config = optimization_config
        self.save_hyperparameters()

    @override
    def forward(self,
                inputs: DataBatchFloat) -> DataBatchFloat:
        """Bare minimum forward method. Simply apply the model to the inputs."""
        return self.model(inputs)

    @abstractmethod
    def core_step(self,
                  tag: Literal["train", "val", "test"],
                  batch: tuple[DataBatchFloat, LabelBatch],
                  batch_index: int) -> ScalarFloat:
        """Compute model outputs and corresponding losses.

        Parameters:
            tag: One of `'train'`, `'val'`, or `'test'`. Tells us what we are currently doing in
                this step. Mainly used for logging correctly.
            batch: Expected to be a *tuple* of `inputs, labels`. The labels are ignored for most
                generative models, but we keep them since we later want to implement conditional
                generative models, and this way we can keep using the same datasets. Still, this
                means your datasets should *always* include labels! These could be dummy entries in
                the unconditional case.
            batch_index: Ignored.

        Returns:
            Scalar loss for the batch.
        """
    
    @override
    def training_step(self,
                      batch: tuple[DataBatchFloat, LabelBatch],
                      batch_index: int) -> ScalarFloat:
        return self.core_step("train", batch, batch_index)

    @override
    def validation_step(self,
                        batch: tuple[DataBatchFloat, LabelBatch],
                        batch_index: int) -> ScalarFloat:
        return self.core_step("val", batch, batch_index)

    @override
    def test_step(self,
                  batch: tuple[DataBatchFloat, LabelBatch],
                  batch_index: int) -> ScalarFloat:
        return self.core_step("test", batch, batch_index)
    
    @override
    def predict_step(self,
                     batch: tuple[DataBatchFloat, LabelBatch],
                     batch_index: int) -> DataBatchFloat:
        """Bare minimum predict_step.
        
        Take the input out of the batch, thus discarding labels, then call the module. This may NOT
        always be correct, so you may need to overwrite this. However, we usually get away without
        ever using this method, so whatever."""
        return self(batch[0])

    @override
    def configure_optimizers(self):
        """Create optimizer and learning rate schedule.
        
        In principle, it's possible to provide optimizer and schedule through the config directly.
        However, our decay has to know the number of training steps, which cannot be directly
        inferred from the config. As such, we instead bundle related parameters in one place in the
        config, and handle the rest in here.
        """
        optimizer = self.optimization_config.optimizer.build(self.parameters())
        scheduler = self.optimization_config.scheduler.build(
            optimizer,
            self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "interval": "step",
                                 "frequency": 1}}

    def generate(self,
                 n_samples: int,
                 **kwargs) -> DataBatchFloat:
        """Convenience wrapper for generating samples, simply calling `self.model.generate()`.
        
        Parameters:
            n_samples: How many samples to generate.
            **kwargs: Keyword arguments to the module's `generate()` function.

        Returns:
            Batch of generated samples.
        """
        return self.model.generate(n_samples, **kwargs)

    def log_metrics(self,
                    tag: str,
                    metrics: dict[str, ScalarFloat]) -> OptimizerLRScheduler:
        """Log all given metrics under a certain tag.
        
        Parameters:
            tag: String prepended to all metrics. If `"train"`, will log metrics on step, else only
                on epoch.
        """
        on_step_log = tag == "train"
        for name, value in metrics.items():
            self.log(f"{tag}_{name}", value,
                     on_epoch=True, on_step=on_step_log, prog_bar=True, logger=True)
