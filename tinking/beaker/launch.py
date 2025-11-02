import os
import re
import secrets
import string
import sys
from typing import List, Optional

import beaker as bk

from gantry.api import launch_experiment

import chz
from rich.console import Console

from tinking.beaker.constants import WEKA_CLUSTERS
from tinking.beaker.defaults import get_env_vars, get_mounts

console = Console()


@chz.chz
class BeakerConfig:
    workspace: str = "ai2/davidh"
    cluster: list[str] = chz.field(default_factory=lambda: WEKA_CLUSTERS)
    budget: str = "ai2/oe-base"

    # Optional args
    hostname: Optional[list[str]] = None  # specific nodes to run a job
    max_retries: int = 0
    gpus: int = 0
    num_nodes: int = 1
    image: str = "ai2/cuda12.8-dev-ubuntu22.04-torch2.7.0"
    description: str = "tinker 🤍 papergym conductor"
    task_name: str = "papergym-tinker"
    priority: str = "normal"
    preemptible: bool = True
    env: list[dict[str, str]] = chz.field(default_factory=list)
    secret: list[dict[str, str]] = chz.field(default_factory=list)
    no_host_networking: bool = False
    pure_docker_mode: bool = False

    # beaker_datasets: List[Dict[str, str]] = field(
    #     default_factory=list
    # )  # TODO: Add parser from mason.py

    follow: bool = False
    allow_dirty: bool = False
    dry_run: bool = False


def gen_uuid(length: int = 8) -> str:
    """Random base-36 string of `length` digits."""
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def make_command(cmd: List[str], config: BeakerConfig) -> str:
    # escape the command (e.g., --stop_strings "</answer>")
    for i in range(len(cmd)):
        if "</" in cmd[i]:
            cmd[i] = f"'{cmd[i]}'"

    # special logic to deal with escape like
    # python mason.py ... -- python x.py --dataset_mixer '{"trl-internal-testing/sentiment-trl-style": 1.0}'
    # we need to wrap the json string with single quote
    for idx in range(len(cmd)):
        if "{" in cmd[idx]:
            cmd[idx] = "'" + cmd[idx] + "'"

    setup_cmd = ""
    if not config.pure_docker_mode:
        setup_cmd = f"cd {os.getcwd()} && "

    # override accelerate call
    join_cmd = " ".join(cmd)
    if config.num_nodes > 1:
        if "--num_processes" not in join_cmd and "accelerate" in join_cmd:
            raise ValueError(
                "num_processes must be specified in the command for accelerate-based multi-node jobs."
            )
        join_cmd = re.sub(
            r"--num_processes (\d+)",
            lambda m: (
                f"--num_processes {int(m.group(1)) * config.num_nodes} "
                f"--num_machines {config.num_nodes} "
                "--machine_rank $BEAKER_REPLICA_RANK "
                "--main_process_ip $BEAKER_LEADER_REPLICA_HOSTNAME "
                "--main_process_port 29400 "
            ),
            join_cmd,
        )

    cmd = setup_cmd + join_cmd

    return cmd


_REMOTE_COMMAND = None  # Global to store the remote command


def parse_remote_command_from_argv():
    """
    Parse commands separated by '--' into list of command lists.

    E.g.:    launch.py [options] -- cmd1 arg1 -- cmd2 arg2
    Returns: [["cmd1", "arg1"], ["cmd2", "arg2"]]
    """
    global _REMOTE_COMMAND
    
    if "--" not in sys.argv:
        raise ValueError("No command separator '--' found. Usage: launch.py [options] -- command")
    
    first_cmd_idx = sys.argv.index("--")
    
    # Get everything after first --
    remote_args = sys.argv[first_cmd_idx + 1 :]
    
    if not remote_args:
        raise ValueError("No command provided after '--'")
    
    # Store the remote command
    _REMOTE_COMMAND = remote_args
    
    # Remove the remote command from sys.argv so chz only sees the config args
    sys.argv = sys.argv[:first_cmd_idx]


def get_remote_command() -> List[str]:
    """Get the remote command that was extracted from argv."""
    if _REMOTE_COMMAND is None:
        raise ValueError("Remote command not parsed yet")
    return _REMOTE_COMMAND


def launch_beaker(config: BeakerConfig):
    global_wandb_id = gen_uuid()

    beaker_client = bk.Beaker.from_env(default_workspace=config.workspace)

    beaker_secrets = [
        secret.name for secret in beaker_client.secret.list(workspace=config.workspace)
    ]
    whoami = beaker_client.user_name

    # Get the remote command that was parsed from argv
    remote_command = get_remote_command()
    full_commands = make_command(remote_command, config)

    env_vars, env_secrets = get_env_vars(
        config.cluster,
        beaker_secrets,
        whoami,
        global_wandb_id,
        config.pure_docker_mode,
        config.num_nodes,
        config.env,
        config.secret,
        config.preemptible,
    )
    env_vars = [f"{var.name}={var.value}" for var in env_vars]
    env_secrets = [f"{var.name}={var.secret}" for var in env_secrets]

    # TODO: Move this to constants
    weka = [
        "oe-adapt-default:/oe-adapt-default",
        "oe-training-default:/oe-training-default",
        "oe-eval-default:/oe-eval-default",
    ]

    # mounts = None # TODO: fix mounts
    # mounts = get_mounts(config.beaker_datasets, config.cluster)

    # Launch the experiment
    launch_experiment(
        args=full_commands.split(" "),
        workspace=config.workspace,
        clusters=config.cluster,
        budget=config.budget,
        # datasets= # TODO: add ability to add this
        name=config.task_name,
        description=config.description,
        hostnames=config.hostname,
        beaker_image=config.image,
        gpus=config.gpus,
        preemptible=config.preemptible,
        retries=config.max_retries,
        # mounts=mounts, # need to fix
        replicas=config.num_nodes,
        host_networking=not config.no_host_networking,
        env_vars=env_vars,
        env_secrets=env_secrets,
        yes=True,
        priority=config.priority,
        weka=weka,
        timeout=(
            99999999 if config.follow else 0
        ),  # only way to follow the experiment without canceling

        # install="uv sync",        
        uv_venv="/app/.venv", # pre-installed uv directory

        allow_dirty=config.allow_dirty,
        dry_run=config.dry_run,
    )


def main(config: BeakerConfig):
    launch_beaker(config)


if __name__ == "__main__":
    # Parse the remote command (after '--') before chz processes config args
    parse_remote_command_from_argv()

    chz.nested_entrypoint(main, allow_hyphens=True)
