import torch
from utils.dotdic import DotDic

class Episode:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.steps = torch.zeros(opt.bs).int().to(device)
        self.ended = torch.zeros(opt.bs).int().to(device)
        self.r = torch.zeros(opt.bs, opt.game_nagents).float().to(device)
        self.step_records = []

    def reset(self):
        self.steps = torch.zeros(self.opt.bs).int().to(self.device)
        self.ended = torch.zeros(self.opt.bs).int().to(self.device)
        self.r = torch.zeros(self.opt.bs, self.opt.game_nagents).float().to(self.device)
        self.step_records = []

    def create_step_record(self):
        opt = self.opt
        record = DotDic({})
        record.s_t = None
        record.r_t = torch.zeros(opt.bs, opt.game_nagents).to(self.device)
        record.terminal = torch.zeros(opt.bs).to(self.device)

        record.agent_inputs = []

        # Track actions at time t per agent
        record.a_t = torch.zeros(opt.bs, opt.game_nagents, dtype=torch.long).to(self.device)
        if not opt.model_dial:
            record.a_comm_t = torch.zeros(opt.bs, opt.game_nagents, dtype=torch.long).to(self.device)

        # Track messages sent at time t per agent
        if opt.comm_enabled:
            comm_dtype = opt.model_dial and torch.float or torch.long
            comm_dtype = torch.float
            record.comm = torch.zeros(opt.bs, opt.game_nagents, opt.game_comm_bits, dtype=comm_dtype).to(self.device)
            if opt.model_dial and opt.model_target:
                record.comm_target = record.comm.clone()

        # Track hidden state per time t per agent
        record.hidden = torch.zeros(opt.game_nagents, opt.model_rnn_layers, opt.bs, opt.model_rnn_size).to(self.device)
        record.hidden_target = torch.zeros(opt.game_nagents, opt.model_rnn_layers, opt.bs, opt.model_rnn_size).to(self.device)

        # Track Q(a_t) and Q(a_max_t) per agent
        record.q_a_t = torch.zeros(opt.bs, opt.game_nagents).to(self.device)
        record.q_a_max_t = torch.zeros(opt.bs, opt.game_nagents).to(self.device)

        # Track Q(m_t) and Q(m_max_t) per agent
        if not opt.model_dial:
            record.q_comm_t = torch.zeros(opt.bs, opt.game_nagents).to(self.device)
            record.q_comm_max_t = torch.zeros(opt.bs, opt.game_nagents).to(self.device)

        return record