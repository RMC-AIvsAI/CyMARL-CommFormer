from copy import deepcopy
from prettytable import PrettyTable
import numpy as np

from CybORG.Shared.Results import Results
from CybORG.Wrappers.BaseWrapper import BaseWrapper
from CybORG.Wrappers.TrueTableWrapper import TrueTableWrapper

class BlueTableWrapper(BaseWrapper):
    def __init__(self,env=None, output_mode='table'):
        super().__init__(env)
        self.env = TrueTableWrapper(env=env)

        self.baseline = None
        self.output_mode = output_mode
        self.blue_info = {}
        self.agent_hosts = {} # CyMARL
        # CyMARL - Set class vars
        self.reset() 

    def reset(self):        
        result = self.env.reset()
        obs = result.observation
        self._process_initial_obs(obs)
        obs = self.observation_change(obs, baseline=True)
        result.observation = obs
        return result

    def get_table(self,output_mode='blue_table'):
        if output_mode == 'blue_table':
            return self._create_blue_table(success=None)
        elif output_mode == 'true_table':
            return self.env.get_table()

    def observation_change(self, observation, agent: str = None, baseline=False):
        obs = observation if type(observation) == dict else observation.data
        obs = deepcopy(observation)
        success = obs['success']

        self._process_last_action()

        anomaly_obs = self._detect_anomalies(obs) if not baseline else obs
        del obs['success']
        info = self._process_anomalies(anomaly_obs)
        if baseline:
            for host in info:
                info[host][-2] = 'None'
                info[host][-1] = 'No'
                self.blue_info[host][-1] = 'No'
        self.info = info

        if not self.agent_hosts.get(agent, None):
            self.agent_hosts[agent] = list(obs.keys())

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

    def _process_last_action(self):
        action = self.get_last_action(agent='Blue')
        if action is not None:
            name = action.__class__.__name__
            hostname = action.get_params()['hostname'] if name in ('Restore','Remove') else None

            if name == 'Restore':
                self.blue_info[hostname][-1] = 'No'
            elif name == 'Remove':
                compromised = self.blue_info[hostname][-1]
                if compromised != 'No':
                    self.blue_info[hostname][-1] = 'Unknown'

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
        info = deepcopy(self.blue_info)
        for hostid, host_anomalies in anomaly_dict.items():
            assert len(host_anomalies) > 0
            if 'Processes' in host_anomalies:
                connection_type = self._interpret_connections(host_anomalies['Processes'])
                info[hostid][-2] = connection_type
                if connection_type == 'Exploit':
                    info[hostid][-1] = 'User'
                    self.blue_info[hostid][-1] = 'User'
            if 'Files' in host_anomalies:
                malware = [f['Density'] >= 0.9 for f in host_anomalies['Files']]
                if any(malware):
                    info[hostid][-1] = 'Privileged'
                    self.blue_info[hostid][-1] = 'Privileged'

        return info

    def _interpret_connections(self,activity:list):                
        num_connections = len(activity)
        ports = set([item['Connections'][0]['local_port'] \
            for item in activity if 'Connections' in item and 'local_port' in item['Connections'][0]])
        port_focus = len(ports)

        remote_ports = set([item['Connections'][0].get('remote_port') \
            for item in activity if 'Connections' in item])
        if None in remote_ports:
            remote_ports.remove(None)

        if num_connections >= 3 and port_focus >=3:
            anomaly = 'Scan'
        elif 4444 in remote_ports:
            anomaly = 'Exploit'
        elif num_connections >= 3 and port_focus == 1:
            anomaly = 'Exploit'
        else:
            anomaly = 'Scan'

        return anomaly

    # def _malware_analysis(self,obs,hostname):
        # anomaly_dict = {hostname: {'Files': []}}
        # if hostname in obs:
            # if 'Files' in obs[hostname]:
                # files = obs[hostname]['Files']
            # else:
                # return anomaly_dict
        # else:
            # return anomaly_dict

        # for f in files:
            # if f['Density'] >= 0.9:
                # anomaly_dict[hostname]['Files'].append(f)

        # return anomaly_dict


    def _create_blue_table(self, agent, obs, success):
        table = PrettyTable([
            'Hostname',
            'Activity',
            'Compromised'
            ])
        for hostid in [hostid for hostid in self.info.keys() if hostid not in ['Defender', 'User0', 'Enterprise_router', 'Operational_router', 'User_router']]:
            if agent is not None:
                if hostid in self.agent_hosts[agent]:
                    table.add_row(self.info[hostid][2:])
        
        table.sortby = 'Hostname'
        table.success = success
        return table

    def _create_vector(self, agent, obs, success):
        table = self._create_blue_table(agent, obs, success)._rows
        # return empty if no obs provided
        if not len(table):
            return np.empty(shape=(0,))

        proto_vector = []
        # five flags for each host:
        # 1 - is this host uncompromised
        # 2 - was this host exploited last turn
        # 3 - is the status of this host unknown (has been exploited, and not restored or analyzed)
        # 4 - does Red have at least User privleges (exploited, and never removed or restored)
        # 5 - does Red have Root privleges (host was analyzed)
        for row in table:
            activity = row[1]
            compromised = row[2]
            # Flag 0
            if activity == "Scan":
                value = [1]
            else:
                value = [0]
            proto_vector.extend(value)
            # Flag 2
            activity = row[1]
            if activity == "Exploit":
                value = [1]
            else:
                value = [0]
            proto_vector.extend(value)
            # Flag 1
            if compromised != "No":
                value = [1]
            else:
                value = [0]
            proto_vector.extend(value)
            # Flag 3
            if ((compromised != "No") and (compromised != "Privileged")):
                value = [1]
            else:
                value = [0]
            proto_vector.extend(value)
            # Flag 4
            if compromised == "User":
                value = [1]
            else:
                value = [0]
            proto_vector.extend(value)
            # Flag 5
            if compromised == "Privileged":
                value = [1]
            else:
                value = [0]
            proto_vector.extend(value)

        return np.array(proto_vector)

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
