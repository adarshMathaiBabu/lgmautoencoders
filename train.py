import os
if os.path.isdir("/var/tmp"):
    os.environ["TMPDIR"] = "/var/tmp"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/var/tmp"
    print("\ntmpdir set to /var/tmp!\n")

import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.utilities import rank_zero_info


class CustomCLI(LightningCLI):
    """CLI adding options for high matmul precision (for Tensor cores) and model compilation.
    
    May be expanded by further arguments.
    """
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--compile", default=False)
        parser.add_argument("--matmul_high", default=False)  # TODO change to string accepting highest/high/medium

    def before_fit(self):
        rank_zero_info("\n===================================================================")
        rank_zero_info("CustomCLI output here:\n")

        # TODO get around the .fit thing
        if self.config.fit.get("matmul_high", False):
            torch.set_float32_matmul_precision("high")
            rank_zero_info("Using 'high' matmul precision.")

        if self.config.fit.get("compile", False):       
            try:
                self.model.compile()
                rank_zero_info("Model was successfully compiled.")
            except AttributeError as e:
                rank_zero_info(f"Caught attribute error {e}")
                rank_zero_info("WARNING attempt to compile model failed. It seems there is no "
                               "`model` attribute!")
        else:
            rank_zero_info("Model compile was not requested.")

        rank_zero_info("===================================================================\n")


if __name__ == "__main__":
    cli = CustomCLI()
