"""This module implements all the models we treat in this class.

They all build on top of `TrainingModuleBase`.

Unfortunately, we cannot import all the different `TrainingModule` variants here, as that leads to
some kind of circular import. You will need to import from the respective submodules. Sorry!
"""
from .base import TrainingModuleBase
