from copy import deepcopy
from prettytable import PrettyTable
import numpy as np

from CybORG.Shared.Results import Results
from CybORG.Wrappers.BaseWrapper import BaseWrapper
from CybORG.Wrappers.TrueTableWrapper import TrueTableWrapper

class BlueTableDIALWrapper(BaseWrapper):
    def __init__(self,env=None, output_mode='table'):
        super().__init__(env)
        self.env = TrueTableWrapper(env=env)

        self.baseline = None
        self.output_mode = output_mode
        self.blue_info = {}
        self.agent_hosts = {} # CyMARL
        self.agent_blocks = {} # DIAL
        # CyMARL - Set class vars
        self.reset() 

    def reset(self):        
        result = self.env.reset()
        obs = result.observation
        self._process_initial_obs(obs)
        #obs = self.observation_change(obs, baseline=True)
        #result.observation = obs
        #return result

    def get_table(self,output_mode='blue_table'):
        if output_mode == 'blue_table':
            return self._create_blue_table(success=None)
        elif output_mode == 'true_table':
            return self.env.get_table()

    def observation_change(self, observation, agent: str = None, baseline=False):
        obs = observation if type(observation) == dict else observation.data
        obs = deepcopy(observation)
        success = obs['success']

        if not self.agent_hosts.get(agent, None):
            self.agent_hosts[agent] = list(obs.keys())
        
        self.agent_blocks[agent] = []
        for host, info in self.blue_info.items():
            if host in self.agent_hosts[agent]:
                info[-2] = 'None'
        
        self._process_last_action(agent)
        anomaly_obs = self._detect_anomalies(obs) if not baseline else obs
        del obs['success']
        #info = self._process_anomalies(anomaly_obs)
        self._process_anomalies(anomaly_obs)
        """
        if baseline:
            for host in info:
                info[host][-2] = 'None'
                info[host][-1] = 'No'
                self.blue_info[host][-1] = 'No'
        self.info = info
        """
        if self.output_mode == 'table':
            return self._create_blue_table(success)
        elif self.output_mode == 'anomaly':
            anomaly_obs['success'] = success
            return anomaly_obs
        elif self.output_mode == 'raw':
            return observation
        elif self.output_mode == 'vector':
            return self._create_vector(agent, obs, success)
        else:
            raise NotImplementedError('Invalid output_mode for BlueTableWrapper')

    def _process_initial_obs(self, obs):
        obs = obs.copy()
        for agent in self.agents:
            self.agent_blocks[agent] = []
        self.baseline = obs
        del self.baseline['success']
        for hostid in obs:
            if hostid == 'success':
                continue
            host = obs[hostid]
            interface = host['Interface'][0]
            # CyMARL - bug with reading interface occurs 1/10,000 games when running pymarl
            subnet = 'NA'
            try:
                subnet = interface['Subnet']
            except:
                print(f"Warning: Subnet could not be read from host: {hostid}")
            ip = str(interface['IP Address'])
            hostname = host['System info']['Hostname']
            self.blue_info[hostname] = [str(subnet),str(ip),hostname, 'None','No']
        return self.blue_info

    def _process_last_action(self, agent):
        action = self.get_last_action(agent=agent)
        if action is not None:
            name = action.__class__.__name__
            hostname = action.get_params()['hostname'] if name in ('Analyse', 'Restore','Remove') else None
            subnet = action.get_params()['subnet'] if name in ('Block') else None
            if name == 'Analyse':
                self.blue_info[hostname][-1] = 'No'
            if name == 'Block':
                self.agent_blocks[agent].append(subnet)
            if name == 'Restore':
                self.blue_info[hostname][-1] = 'No'
            if name == 'Remove':
                compromised = self.blue_info[hostname][-1]
                if compromised != 'No':
                    if compromised != 'Privileged':
                        self.blue_info[hostname][-1] = 'Unknown'
        """
        action = self.get_last_action(agent=agent)
        if action is not None:
            name = action.__class__.__name__
            hostname = action.get_params()['hostname'] if name in ('Analyse', 'Restore','Remove') else None

            if name == 'Restore':
                self.blue_info[hostname][-1] = 'No'
            elif name == 'Remove':
                #compromised = self.blue_info[hostname][-1]
                #if compromised != 'No':
                self.blue_info[hostname][-1] = 'Unknown'
            
            elif name == 'Analyse':
                #Added this for DIAL implementation, previously after analyse action, with no malware state wasn't updating to reflect no exploits
                #In process_anomalies function added an extra layer to detect user privileges. So if malware is detected, this will get updated
                self.blue_info[hostname][-1] = 'No'
            """
    def _detect_anomalies(self,obs):
        if self.baseline is None:
            raise TypeError('BlueTableWrapper was unable to establish baseline. This usually means the environment was not reset before calling the step method.')

        anomaly_dict = {}

        for hostid,host in obs.items():
            if hostid == 'success':
                continue

            host_baseline = self.baseline[hostid]
            if host == host_baseline:
                continue

            host_anomalies = {}
            if 'Files' in host:
                baseline_files = host_baseline.get('Files',[])
                anomalous_files = []
                for f in host['Files']:
                    if f not in baseline_files:
                        anomalous_files.append(f)
                if anomalous_files:
                    host_anomalies['Files'] = anomalous_files

            if 'Processes' in host:
                baseline_processes = host_baseline.get('Processes',[])
                anomalous_processes = []
                for p in host['Processes']:
                    if p not in baseline_processes:
                        anomalous_processes.append(p)
                if anomalous_processes:
                    host_anomalies['Processes'] = anomalous_processes

            if host_anomalies:
                anomaly_dict[hostid] = host_anomalies

        return anomaly_dict

    def _process_anomalies(self,anomaly_dict):
        #info = deepcopy(self.blue_info)
        for hostid, host_anomalies in anomaly_dict.items():
            assert len(host_anomalies) > 0
            if 'Processes' in host_anomalies:
                connection_type = self._interpret_connections(host_anomalies['Processes'])
                #info[hostid][-2] = connection_type
                self.blue_info[hostid][-2] = connection_type
                if connection_type == 'Exploit':
                    #info[hostid][-1] = 'User'
                    self.blue_info[hostid][-1] = 'User'
            if 'Files' in host_anomalies:
                priv_malware = [f['Density'] >= 0.9 for f in host_anomalies['Files']]
                user_malware = [f['Density'] == 0.8 for f in host_anomalies['Files']]
                if any(user_malware):
                    #info[hostid][-1] = 'User'
                    self.blue_info[hostid][-1] = 'User'
                if any(priv_malware):
                    #info[hostid][-1] = 'Privileged'
                    self.blue_info[hostid][-1] = 'Privileged'

        #return info

    def _interpret_connections(self,activity:list):                
        num_connections = len(activity)
        ports = set([item['Connections'][0]['local_port'] \
            for item in activity if 'Connections' in item and 'local_port' in item['Connections'][0]])
        port_focus = len(ports)

        remote_ports = set([item['Connections'][0].get('remote_port') \
            for item in activity if 'Connections' in item])
        if None in remote_ports:
            remote_ports.remove(None)

        #if num_connections >= 3 and port_focus >=3:
        if num_connections == port_focus:
            anomaly = 'Scan'
        elif 4444 in remote_ports:
            anomaly = 'Exploit'
        elif num_connections >= 3 and port_focus == 1:
            anomaly = 'Exploit'
        else:
            anomaly = 'None'

        return anomaly

    def _create_blue_table(self, agent, obs, success):
        table = PrettyTable([
            'Hostname',
            'Activity',
            'Compromised'
            ])
        for hostid in [hostid for hostid in self.blue_info.keys() if hostid not in ['Defender', 'User0', 'Enterprise_router', 'Operational_router', 'User_router']]:
            if agent is not None:
                if hostid in self.agent_hosts[agent]:
                    table.add_row(self.blue_info[hostid][2:])
        
        table.sortby = 'Hostname'
        table.success = success
        return table

    def _create_vector(self, agent, obs, success):
        table = self._create_blue_table(agent, obs, success)._rows
        #print(table)
        # return empty if no obs provided
        if not len(table):
            return np.empty(shape=(0,))

        proto_vector = []
        # six flags for each host:
        # 0 - is this host being scanned
        # 1 - is this host compromised
        # 2 - was this host exploited last turn
        # 3 - does Red have at least User privleges (exploited, and never removed or restored)
        # 4 - is the status of this host unknown (has been exploited, and not restored or analyzed)
        # 5 - does Red have Root privleges (host was analyzed)
        for row in table:
            vector = []
            activity = row[1]
            if activity == 'None':
                value = [0,0]
            elif activity == 'Scan':
                value = [1,0]
            elif activity == "Exploit":
                value = [1,1]
            vector.extend(value)

            compromised = row[2]
            if compromised == "No":
                value = [0, 0]
            elif compromised == "User":
                value = [0, 1]
            elif compromised == "Privileged":
                value = [1, 1]
            else:
                value = [1, 0] # Unknown
            vector.extend(value)

            proto_vector.append(vector)

        if not self.agent_blocks[agent]:
            value = [0]
        else:
            value = [1]
        proto_vector.append(value)
        return proto_vector

    def get_attr(self,attribute:str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        if agent == 'Blue' and self.output_mode == 'table':
            output = self.get_table()
        else:
            output = self.get_attr('get_observation')(agent)
        
        # CyMARL - convert obs to machine readable
        output = self.observation_change(output, agent)
        return output

    def get_agent_state(self,agent:str):
        return self.get_attr('get_agent_state')(agent)

    def get_action_space(self,agent):
        return self.env.get_action_space(agent)

    def get_last_action(self,agent):
        return self.get_attr('get_last_action')(agent)

    def get_ip_map(self):
        return self.get_attr('get_ip_map')()

    def get_rewards(self):
        return self.get_attr('get_rewards')()
    
    def get_agent_hosts(self, agent):
        return self.agent_hosts[agent]

def create_table_from_vector(vector):
    table = PrettyTable()
    table.field_names = ['Host name', 'Activity', 'Compromised']
    for i in range(0, len(vector), 4):
        host = 'Host' + str(i/4)

        activity_vector = vector[i, i+2]
        if activity_vector == [0,0]:
            activity = 'No Activity'
        elif activity_vector == [1,0]:
            activity = 'Scan'
        elif activity_vector == [1,1]:
            activity = 'Exploit'
        else:
            raise ValueError(f'Input Vector has no valid activity component from \
                    index {i} to {i+1} inclusive.')

        compromised_vector = vector[i+2, i+4]
        if compromised_vector == [0,0]:
            compromised = 'Not Compromised'
        elif compromised == [1,0]:
            compromised = 'Unknown Compromised Level'
        elif compromised == [0,1]:
            compromised = 'User Access'
        elif compromised == [1,1]:
            compromised = 'Privileged Access'
        else:
            raise ValueError(f'Input Vector has no valid compromised component from \
                    index {i+2} to {i+3} inclusive.')

        table.add_row([host, activity, compromised])
