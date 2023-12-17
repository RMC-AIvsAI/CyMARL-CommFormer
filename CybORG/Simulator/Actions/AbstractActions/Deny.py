## The following code contains work of the United States Government and is not subject to domestic copyright protection under 17 USC § 105.
## Additionally, we waive copyright and related rights in the utilized code worldwide through the CC0 1.0 Universal public domain dedication.

"""
Crappy rework of decoy code for making Red processes
"""
# pylint: disable=invalid-name
from ipaddress import IPv4Address
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

from CybORG.Shared import Observation
from CybORG.Simulator.Actions import Action
from CybORG.Shared.Enums import DecoyType
from CybORG.Simulator.Host import Host
from CybORG.Simulator.Session import Session
from CybORG.Simulator.State import State

from CybORG.Simulator.Session import RedAbstractSession

@dataclass
class DosProc:
    """
    Contains information necessary to create a misinform process on a host
    """
    service_name: str
    name: str
    open_ports: List[dict]
    process_type: str
    process_path: Optional[str] = None
    version: Optional[str] = None
    properties: Optional[List[str]] = None

def _is_host_using_port(host: Host, port: int):
    """
    Convenience method for checking if a host is using a port
    """
    if host.processes is not None:
        for proc in host.processes:
            for proc_state in proc.get_state():
                if proc_state.get('local_port', None) == port:
                    return True
    return False

class ProcFactory(ABC):
    """
    Assembles process informationt to appear as a vulnerable process
    """
    @abstractmethod
    def make_proc(self, host: Host) -> DosProc:
        """
        Creates a DosProc instance that contains the necessary information
        to put a decoy on a given host.

        :param host: Host that this decoy will be placed on
        """

    @abstractmethod
    def is_host_compatible(self, host: Host) -> bool:
        """
        Determines whether an instance of this decoy can be placed
        successfully on the given host

        :param host: Host to examine for compatibility with this decoy.
        """

class DosProcFactory(ProcFactory):
    """
    Assembles process information
    """
    def make_proc(self, host: Host) -> DosProc:
        del host
        return DosProc(service_name="DoS", name="DoS.exe",
                open_ports=[{'local_port':5555, 'local_address':'0.0.0.0'}],
                process_type="DoS")

    def is_host_compatible(self, host: Host) -> bool:
        return not _is_host_using_port(host, 5555)
dos_proc_factory = DosProcFactory()

class Deny(Action):
    """
    Creates a misleading process on the designated host depending on
    available and compatible options.
    """
    def __init__(self, *, session: int, agent: str, hostname: str):
        super().__init__()
        self.agent = agent
        self.session = session
        self.hostname = hostname
        self.decoy_type = DecoyType.NONE

    def execute(self, state: State) -> Observation:
        # find session inside or close to the target subnet
        if type(state.sessions[self.agent][self.session]) is not RedAbstractSession:
            return Observation(success=False)

        obs_fail = Observation(False)
        obs_succeed = Observation(True)\

        session = state.sessions[self.agent][self.session]
        host = state.hosts[self.hostname]

        try:
            dos_proc = dos_proc_factory.make_proc(host)
            self.__create_process(obs_succeed, session, host, dos_proc)

            return obs_succeed

        except RuntimeError:
            #print ("Deny FAILURE")
            return obs_fail

    def __create_process(self, obs: Observation, sess: Session, host: Host,
            dos_proc: DosProc) -> None:
        """
        Creates a process & service from DosProc on current host, adds it
        to the observation.
        """

        parent_pid = 1

        process_name = dos_proc.name
        username = sess.username
        version = dos_proc.version
        open_ports = dos_proc.open_ports
        process_type = dos_proc.process_type
        process_props = dos_proc.properties

        service_name = dos_proc.service_name

        new_proc = host.add_process(name=process_name, ppid=parent_pid,
                user=username, version=version, process_type=process_type,
                open_ports=open_ports, decoy_type=self.decoy_type,
                properties=process_props)
        
        host.events['ProcessCreation'].append(
            {'process_name': process_name,
             'parent_pid': parent_pid,
             'pid': new_proc.pid,
            }
        )

        host.add_service(service_name=service_name, process=new_proc.pid,
                session=sess)

        obs.add_process(hostid=self.hostname, pid=new_proc.pid,
                parent_pid=parent_pid, name=process_name,
                username=username, service_name=service_name,
                properties=process_props)

    def __str__(self):
        return f"{self.__class__.__name__} {self.hostname}"
