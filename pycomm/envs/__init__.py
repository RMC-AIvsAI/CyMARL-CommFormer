from functools import partial

from .multiagentenv import MultiAgentEnv

from .cyborg import CyborgEnv
from .switch import SwitchGame


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}

REGISTRY["cyborg"] = partial(env_fn, env=CyborgEnv)
REGISTRY["switch"] = partial(env_fn, env=SwitchGame)


