import torch


def weight_normal_(model, r, ltype=torch.nn.Linear):
    for m in model.modules():
        if isinstance(m, ltype):
            torch.nn.init.kaiming_normal_(m.weight)
            m.weight.data.mul_(r)


def weight_uniform_(model, r):
    for m in model.modules():
        if isinstance(m, torch.torch.nn.Linear):
            # torch.nn.init.kaiming_uniform_(m.weight, a=r*math.sqrt(5))
            fan = torch.nn.init._calculate_correct_fan(m.weight, "fan_in")
            std = r / torch.sqrt(fan)
            bound = (
                torch.sqrt(3.0) * std
            )  # Calculate uniform bounds from standard deviation

            with torch.no_grad():
                m.weight.uniform_(-bound, bound)