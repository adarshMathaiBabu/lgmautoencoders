# Config Checklist

Since Lightning CLI configs are rather verbose, you will likely want to copy & paste configs from
previous models and only change the necessary parts.
You may also want to create slightly different configs for various experiments.
Below, you can find reminders for which parts of the config to change in certain scenarios.
It's very easy to forget this stuff!

We put all our example configs in the `config` folder, but you can
put them wherever you want.


## Same Model Type & Dataset, Different Settings

"Settings" here refers to things like architecture hyperparameters, optimizer etc.

- Change all the things you want to change (duh!).
- Change `trainer.logger.init_args.name`!
This is easy to forget.
While not strictly necessary, you should use different names that describe what the respective
experiment is about.
If you were to rely on version numbers only, you will soon forget which version is which.


## Same Model Type, Different Dataset

- Change `data.init_args.dataset` (duh).
    - Also adapt things like batch size, and include necessary transforms.
    For example, if you swap to something like `flickr`, you will most likely want to include an
    appropriate `resize` value.
- In the model architecture, change `input_shape` to the correct value.
Pretty much all model types we use will have this somewhere, often as part of some Config object.
It may be possible to "auto-fill" this from the data module, but this is not something we support at
this point.
- Also in the model architecture, change things like `blocks_per_level` or `level_filters` to have
an appropriate number of levels for the dataset (mainly dependent on resolution).
- Change `trainer.logger.init_args.save_dir`.
Assuming you are using the naming scheme we propose, this argument contains the name of the dataset.
If you forget to change this, your runs will end up in the wrong directory (i.e. for a different
dataset).
    - Also be sure to adapt `trainer.logger.init_args.name` to something appropriate for your
    experiment.
- If using a `GenerativeMetrics` callback, change the `extractor_path` to make sure you are using
the correct feature extractor. Note, this is only available after we have treated how to evaluate
generative models in class.


## Different Model Type

- Change `save_dir` and `name` for `trainer.logger`.
- Set the correct callbacks.
    - Are things like `ReconstructionLogging` (for autoencoders) or `GenerationLogging` (for
    anything that can generate) appropriate?
    - Do you want to use `GenerativeMetrics`, and if so, how often/how many samples?
    - Is `BestCheckpoint` appropriate, i.e. is there a sensible metric to use as a measure?
    - A `ModelSummary` callback is included by default, but you might want to include one with a
    different `max_depth`, as this can give a better overview over the components of your model.
    - Some models profit from using exponential moving averages of the weights. We have an
    `EMAModel` callback for this purpose.
    In principle, there is no issue with always having this included.
    However, this callback can incur a suprisingly large overhead, so if you don't plan on using it,
    you might want to exclude it to speed up training.
- Change all the `model` stuff for the new model type.
- Include any necessary transforms for the `data`, e.g. likelihood-based models for images will
usually want `UniformQuantize`; other models may expect data to be scaled to some specific range.


## Profiling

Somewhat unrelated to the above points:
You can add a _profiler_ to check your training runs for bottlenecks.
Look for the `profiler: null` line in the config, and replace it by something like

```
profiler:
  class_path: lightning.pytorch.profilers.SimpleProfiler
  init_args:
    filename: <desired_filename>
```

This will write a text file with profiling information into the log directory with the given
`filename`.
There is also `AdvancedProfiler`, which will give much more detailed output.
Note that the file is only written after training is finished.
Also, using the profiler incurs some overhead in itself, so you might want to use it only if you
really want to check for bottlenecks.
