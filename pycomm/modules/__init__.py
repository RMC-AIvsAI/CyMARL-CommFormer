REGISTRY = {}

from .cyborg import CybORGCNet
from .switch import SwitchCNet

REGISTRY["cyborg"] = CybORGCNet
REGISTRY["switch"] = SwitchCNet