import torch

s_t = torch.tensor([[1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 5],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 5]])
for i in range(3):
    start_index = i * 5
    end_index = min(start_index + 5, s_t.size(1))
    if end_index - start_index == 5:
        module_input = s_t[:, start_index:end_index].matmul(2**torch.arange(end_index - start_index).flip(0))
    else:
        module_input = s_t[:, start_index:end_index].squeeze()

    print(start_index, end_index, module_input)