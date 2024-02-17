from ipaddress import IPv4Address

from CybORG.Shared import Observation
from CybORG.Simulator.Actions.Action import RemoteAction
from CybORG.Simulator.Actions.ConcreteActions.LocalAction import LocalAction
from CybORG.Simulator.Actions.Action import lo
from CybORG.Simulator.Host import Host
from CybORG.Simulator.State import State


class Portscan(RemoteAction):
    def __init__(self, session: int, agent: str, ip_address: IPv4Address):
        super().__init__(session, agent)
        self.ip_address = ip_address

    def get_used_route(self, state: State) -> list:
        """finds the route used by the action and returns the hostnames along that route"""
        return self.get_route(state, state.ip_addresses[self.ip_address], state.sessions[self.agent][self.session].hostname)

    def execute(self, state: State) -> Observation:
        self.state = state
        obs = Observation()
        if self.session not in state.sessions[self.agent]:
            obs.set_success(False)
            obs.add_session_info(hostid=self.ip_address, agent=self.agent)
            return obs
        from_host = state.hosts[state.sessions['Red'][self.session].hostname]
        session = state.sessions['Red'][self.session]

        # Check if the target subnet is blocking traffic from the current sessions subnet
        from_subnet = state.subnets_cidr_to_name[from_host.interfaces[0].subnet]
        to_subnet = state.hostname_subnet_map[state.ip_addresses[self.ip_address]]
        if to_subnet in state.blocks:
            if state.blocks[to_subnet] == from_subnet:
                obs.set_success(False)
                return obs

        if not session.active:
            obs.set_success(False)
            return obs

        originating_ip_address = self._get_originating_ip(state, from_host, self.ip_address)
        if originating_ip_address is None:
            obs.set_success(False)
            return obs

        target_host = state.hosts[state.ip_addresses[self.ip_address]]

        obs.set_success(True)

        for process in target_host.processes:
            for conn in process.connections:
                if 'local_port' in conn and 'remote_port' not in conn:
                    obs.add_process(hostid=str(self.ip_address), local_port=conn["local_port"], local_address=self.ip_address)
                    target_host.events['NetworkConnections'].append({'local_address': self.ip_address,
                                                                     'local_port': conn["local_port"],
                                                                     'remote_address': originating_ip_address,
                                                                     'remote_port': target_host.get_ephemeral_port()})
        return obs
