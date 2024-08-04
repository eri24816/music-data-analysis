import subprocess


def run_command(command: str):
    subprocess.run(
        command,
        shell=True,
        check=True,
    )
