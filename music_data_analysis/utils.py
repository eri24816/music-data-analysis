import subprocess
import matplotlib.pyplot as plt

HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass


def run_command(command: str):
    subprocess.run(
        command,
        shell=True,
        check=True,
    )

def show_pianoroll(tensor: torch.Tensor):
    pianoroll = tensor.detach().cpu().t().flip(0)
    plt.imshow(pianoroll, cmap="gray", vmin=0, vmax=1)
    plt.show()
