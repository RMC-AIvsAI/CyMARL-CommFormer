import random

from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Shared import Results
from CybORG.Simulator.Actions import PrivilegeEscalate, ExploitRemoteService, DiscoverRemoteSystems, Impact, \
    DiscoverNetworkServices, Sleep


class RedDialAgent(BaseAgent):
    # a red agent that meanders through scenario 1b
    def __init__(self):
        self.scanned_subnets = set() # Tracks which subnets have been scanned
        self.scanned_ips = {} # Tracks which IP addresses have been scanned and open ports associated with it
        self.discovered_ips = [] # Tracks IP addresses that have been discovered
        self.exploited_ips = {} # Tracks which IP addresses have been successfully exploited
        self.escalated_hosts = set() # Tracks hosts that have been escalated (admin access achieved)
        self.host_ip_map = {} # Maps each host to its corresponding IP address
        self.ip_host_map = {} # Maps each IP to its corresponding Host
        self.ip_subnet_map = {} # Maps each IP address to its corresponding subnet
        self.step_count = 0 # Tracks the current step count of the agent
        self.sessions = {} # Stores session IDs obtained by exploiting hosts
        self.discovered_subnet_host = {} # Keeps track of the host that discovered a specific subnet
        self.action = None # Keeps track of the last action taken by the agent for logic decisions

    def train(self, results: Results):
        """allows an agent to learn a policy"""
        pass

    def get_action(self, observation, action_space):
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        # At step0 add the already compromised User0 information to Class Variables
        if 'User0' not in self.escalated_hosts:
            self._initialize_with_observation(observation)

        # Check the last action, if it was not successful then remove the Ips and/or hosts from the compromised lists
        self._process_success(observation)

        self.step_count += 1

        # Always impact if able
        if 'Op_Server0' in self.escalated_hosts:
            self.action = Impact(agent='Red', hostname='Op_Server0', session=self.sessions['Op_Server0'])
            return self.action
        
        # Discover other IP addresses in the current or other subnets
        for subnet, host in self.discovered_subnet_host.items():
            # Determine the session to use based on the host that discovered the subnet
            session = self.sessions.get(host)
            if session is not None and host in self.escalated_hosts:
                # Check if the subnet has already been scanned
                if subnet not in self.scanned_subnets:
                    self.action =  DiscoverRemoteSystems(subnet=subnet, agent='Red', session=session)
                    return self.action
        
        # Discover network services on newly discovered IPs
        random.shuffle(self.discovered_ips)
        for ip_address in self.discovered_ips:
            if ip_address not in self.exploited_ips and ip_address not in self.scanned_ips:  # Only scan unexploited IPs and unscanned IPs
                subnet = self.ip_subnet_map[ip_address]
                # Retrieve the session associated with the subnet where the IP was discovered
                discovering_host = self.discovered_subnet_host.get(subnet)
                session = self.sessions.get(discovering_host)

                if session is not None and discovering_host in self.escalated_hosts :
                    # Use the session that discovered the IP to scan for open ports
                    self.action =  DiscoverNetworkServices(ip_address=ip_address, agent='Red', session=session)
                    return self.action
        
        # Check if any exploited host is ready for privilege escalation
        keys = list(self.exploited_ips.keys())
        random.shuffle(keys)
        exploited_ips = {key: self.exploited_ips[key] for key in keys}
        for ip_address, exploit_info in exploited_ips.items():
            os_type = exploit_info.get('os')
            exploit_time = exploit_info.get('time')
            delay = random.randint(2, 3)  # Random delay between 2 to 3 timesteps
            hostname = self.ip_host_map.get(ip_address)
            if self.step_count - exploit_time >= delay and hostname not in self.escalated_hosts:
                if hostname and self.sessions.get(hostname):
                    session_id = self.sessions[hostname]
                    self.action = PrivilegeEscalate(hostname=hostname, agent='Red', session=session_id, os_type=os_type)
                    return self.action
                
        # Attempt to exploit a host that has open ports discovered
        keys = list(self.scanned_ips.keys())
        random.shuffle(keys)
        scanned_ips = {key: self.scanned_ips[key] for key in keys}
        for ip_address, ports in scanned_ips.items():
            if ip_address in self.exploited_ips:  # Skip already exploited IPs
                continue

            # Select the session based on the host that discovered the IP
            # This logic assumes exploitation uses the session that discovered the open ports
            subnet = self.ip_subnet_map[ip_address]
            from_host = self.discovered_subnet_host.get(subnet)
            if from_host is None:
                for i, sub in self.ip_subnet_map.items():
                    if i != ip_address:
                        if sub == subnet:
                            from_host = self.ip_host_map.get(i)
            session = self.sessions.get(from_host)

            if session is not None and ports and from_host in self.escalated_hosts:
                self.action = ExploitRemoteService(ip_address=ip_address, agent='Red', session=session, ports=ports)
                return self.action
        # Default action if no other actions are applicable
        return Sleep() 

    def _process_success(self, observation):
        last_action = self.action
        if last_action is not None:
            name = last_action.__class__.__name__
            # Check for successful DiscoverRemoteSystems action by examining 'success' and interface details in observation
            if name == 'DiscoverRemoteSystems':
                if observation['success'] == True:
                    for ip_address, info in observation.items():
                        if ip_address == 'success':
                            continue  

                        if 'Interface' in info:
                            interface_details = info['Interface'][0]
                            discovered_ip = interface_details['IP Address']
                            subnet = interface_details['Subnet']

                            # Update tracking variables with the new IP and its subnet
                            if discovered_ip not in self.discovered_ips:
                                self.discovered_ips.append(discovered_ip)
                                self.ip_subnet_map[discovered_ip] = subnet
                            
                            # Update discovered subnet variable
                            if subnet not in self.scanned_subnets:
                                self.scanned_subnets.add(subnet)
                else:
                    for subnet, info in observation.items():
                        if subnet == 'success':
                            continue

                        if 'Sessions' in info:
                            host = self.discovered_subnet_host[subnet]
                            host_ip = self.host_ip_map[host]
                            self.discovered_subnet_host.pop(subnet)
                            self.escalated_hosts.remove(host)
                            self.exploited_ips.pop(host_ip)
                            self.sessions.pop(host)
                            self.host_ip_map.pop(host)
                            self.ip_host_map.pop(host_ip)
            elif name == 'DiscoverNetworkServices':
                if observation['success'] == True:
                    for ip_address, info in observation.items():
                        if ip_address == 'success':
                            continue  

                        if 'Processes' in info:
                            for process_details in info['Processes']:
                                if 'Connections' in process_details:
                                    connections = process_details['Connections'][0]
                                    ip = connections['local_address']
                                    port = connections['local_port']
                                    if ip not in self.scanned_ips:
                                        self.scanned_ips[ip] = [port]
                                    if port not in self.scanned_ips[ip]:
                                        self.scanned_ips[ip].append(port)
                else:
                    for ip, info in observation.items():
                        if ip == 'success':
                            continue  

                        if 'Sessions' in info:
                            subnet = self.ip_subnet_map[ip]
                            host = self.discovered_subnet_host[subnet]
                            host_ip = self.host_ip_map[host]
                            self.discovered_subnet_host.pop(subnet)
                            self.escalated_hosts.remove(host)
                            self.exploited_ips.pop(host_ip)
                            self.sessions.pop(host)
                            self.host_ip_map.pop(host)
                            self.ip_host_map.pop(host_ip)
            elif name == 'ExploitRemoteService':
                if observation['success'] == True:
                    for ip_address, info in observation.items():
                        if ip_address == 'success':
                            continue
                        hostname = None
                        os = None
                        ip = None
                        session_id = None
                        if 'System info' in info and 'Sessions' in info and 'Interface' in info:
                            system_info = info['System info']
                            session_info = info['Sessions'][0]
                            interface_info = info['Interface'][0]
                            hostname = system_info.get('Hostname')
                            os = system_info.get('OSType')
                            ip = interface_info.get('IP Address')
                            session_id = session_info.get('ID')
                            if ip is not None and os is not None:
                                self.exploited_ips[ip] = {'os': os, 'time': self.step_count}
                            if session_id is not None:
                                self.sessions[hostname] = session_id
                            if hostname is not None and ip is not None:
                                self.host_ip_map[hostname] = ip
                                self.ip_host_map[ip] = hostname
                else:
                    session = last_action.get_params()['session']
                    for ip, info in observation.items():
                        if ip == 'success':
                            continue  

                        if 'Sessions' in info:
                            subnet = self.ip_subnet_map[ip]
                            host = [key for key, value in self.sessions.items() if value == session]
                            #host = self.discovered_subnet_host[subnet]
                            host_ip = self.host_ip_map[host[0]]
                            if subnet in self.discovered_subnet_host and host[0] == self.discovered_subnet_host[subnet]:
                                self.discovered_subnet_host.pop(subnet)
                            self.escalated_hosts.remove(host[0])
                            self.exploited_ips.pop(host_ip)
                            self.sessions.pop(host[0])
                            self.host_ip_map.pop(host[0])
                            self.ip_host_map.pop(host_ip)
            elif name == 'PrivilegeEscalate':
                hostname = last_action.get_params()['hostname']
                if observation['success'] == True:
                    for host, info in observation.items():
                        if host == 'success':
                            continue
                        
                        if hostname == host:
                            if 'Sessions' in info:
                                session_info = info['Sessions'][0]
                                if 'Username' in session_info:
                                    if session_info['Username'] == 'root' or session_info['Username'] == 'SYSTEM':
                                        self.escalated_hosts.add(hostname)
                        else:
                            if 'Interface' in info:
                                interface_info = info['Interface'][0]
                                if 'Subnet' in interface_info:
                                    self.discovered_subnet_host[interface_info['Subnet']] = hostname
                else:
                    for host, info in observation.items():
                        if host == hostname and 'Sessions' in info:
                            ip = self.host_ip_map[host]
                            self.exploited_ips.pop(ip)
                            self.sessions.pop(host)
                            self.host_ip_map.pop(hostname)
                            self.ip_host_map.pop(ip)
            elif name == 'Impact':
                hostname = last_action.get_params()['hostname']
                if observation['success'] == False:
                    for host, info in observation.items():
                        if host == hostname and 'Sessions' in info:
                            ip = self.host_ip_map[host]
                            self.escalated_hosts.remove(host)
                            self.exploited_ips.pop(ip)
                            self.sessions.pop(host)
                            self.host_ip_map.pop(hostname)
                            self.ip_host_map.pop(ip)

    def _initialize_with_observation(self, observation):
        user0_data = observation.get('User0', {})
        
        if 'System info' in user0_data:
            system_info = user0_data['System info']
            hostname = system_info.get('Hostname')
            os = system_info.get('OSType')
            if hostname:
                self.escalated_hosts.add(hostname)
                
        if 'Interface' in user0_data:
            interface_info = user0_data['Interface'][0]  # Assuming there's at least one interface
            ip_address = interface_info.get('IP Address')
            subnet = interface_info.get('Subnet')
            if ip_address and subnet:
                self.discovered_ips.append(ip_address)
                self.exploited_ips[ip_address] = {'os': os, 'time': self.step_count}
                self.host_ip_map[hostname] = ip_address
                self.ip_host_map[ip_address] = hostname
                self.ip_subnet_map[ip_address] = subnet
                self.discovered_subnet_host[subnet] = hostname
                if ip_address not in self.scanned_ips:
                    self.scanned_ips[ip_address] = []
                
        if 'Sessions' in user0_data:
            session_info = user0_data['Sessions'][0]  # Assuming there's at least one session
            session_id = session_info.get('ID')
            if session_id is not None:
                self.sessions[hostname] = session_id

    def end_episode(self):
        self.scanned_subnets = set()
        self.scanned_ips = {}
        self.discovered_ips = []
        self.exploited_ips = {}
        self.escalated_hosts = set()
        self.host_ip_map = {}
        self.ip_host_map = {}
        self.ip_subnet_map = {}
        self.step_count = 0
        self.sessions = {}
        self.discovered_subnet_host = {}
        self.action = None

    def set_initial_values(self, action_space, observation):
        pass
