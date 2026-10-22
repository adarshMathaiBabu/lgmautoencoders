# Lightning & CLI

Pytorch Lightning is well documented ([LINK](https://lightning.ai/docs/pytorch/stable/)).
There are tutorials as well as an API reference.

The CLI has its own documentation:
[LINK](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli).
At the end of the day, you just need to understand what the config files look like and how to start
training runs using these files.
It's really just a command like this:

`python3 train.py fit --config config/<model>/<name>.yaml`

Here, the path is just an example.
You could adapt it to anything you want.
That's it!
As long as the actual code is complete and the config is set up correctly, this can be used to train
any model we will treat in this class.
We recommend starting runs from inside a `tmux` session, so that they aren't cancelled if you
disconnect (see [Tips & Tricks](tips.md)).


## Debugging

While the CLI is extremely convenient, it has one major weakness that is especially problematic when
you are just learning the concepts:
Error messages are often misleading or indecipherable, especially when they occur when trying to
build the objects (i.e. during model construction, before training has started).
This can make it very difficult to understand what exactly is going wrong, and how one might fix it.

Unfortunately, we have not found a good way around this.
The error handling seems to be done by the `jsonargparse` module, which is responsible for turning
the `yaml` configs into executable code.
This means we would have to circumvent that module to get better error messages, but that would mean
abandoning the convenience of config files, or somehow reimplementing the entire functionality
ourselves (not happening).

Thus, you might want to debug your model and training implementation in small notebooks, and only
switch to the CLI once training works.
Unfortunately, this means you would need to build everything "by hand", as we cannot parse the
config files and maintain clear error messages.
We will not supply such notebooks for every model type to avoid bloating the repository.
However, it should be relatively easy to adapt the `01_showcase.ipynb` notebook each week, and try
to get the training to run in there.
You can of course still use the configs as guidance as to what the various arguments to the model
might look like.
This way, you should get more understandable error tracebacks, which should help with the initial
development.
