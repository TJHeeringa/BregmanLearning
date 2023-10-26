import torch
from typing import List, Union


def row_simplification(model: torch.nn.Module, typical_input: torch.Tensor, pinned_outputs: List[Union[str, torch.nn.Module]]) -> torch.nn.Module:
    """This method simplifies the inputted neural network based on zero rows in the models nn.Linear's weight
    matrices. Zero rows mean that certain outputs from the previos layers can be ignored. Hence, neurons with
    zero rows can be removed. It does this by creating hooks for the network. After running the network with a
    certain input, these hooks remove the unneeded neurons. Directly thereafter the hooks are removed from
    the network.

    Note:
    This is an in-place method. If you want to keep your original model for comparison, call this method with
    `copy.deepcopy(model)` to prevent this from changing the original.

    :param model:
    :param typical_input:
    :param pinned_outputs:
    :return:
    """

    # Define hooks that helps propagate which inputs are relevant to the next layer
    @torch.no_grad()
    def __remove_nan(_, input_):
        nan_idx = torch.isnan(input_[0])
        new_input = input_[0].clone()
        new_input[~nan_idx] = 0
        new_input[nan_idx] = 1
        return new_input, *input_[1:]

    @torch.no_grad()
    def __remove_nan2(module, input):
        """
        PyTorch hook that removes nans from input.
        """
        module.register_buffer("pruned_input", ~torch.isnan(input[0][0].view(input[0][0].shape[0], -1).sum(dim=1)))
        if torch.isnan(input[0]).sum() > 0:
            input[0][torch.isnan(input[0])] = 0
        return input

    # Define hook that pushes biases to node that aren't being removed
    @torch.no_grad()
    def __propagate_biases_hook(module, input_, output, name):
        bias_feature_maps = output[0].clone()

        ###########################################################################################
        #  STEP 1. Fuse biases of pruned channels in the previous module into the current module  #
        ###########################################################################################
        module.register_parameter('bias', torch.nn.Parameter(bias_feature_maps))

        #############################################
        #  STEP 2. Propagate output to next module  #
        #############################################
        shape = module.weight.shape  # Compute mask of zeroed (pruned) channels
        pruned_channels = module.weight.view(shape[0], -1).sum(dim=1) == 0

        if name in pinned_outputs:
            # No bias is propagated for pinned layers
            return output * float('nan')

        if isinstance(module, torch.nn.Linear):
            output[~pruned_channels[None, :].expand_as(output)] *= float('nan')
            if getattr(module, 'bias', None) is not None:
                module.bias.data.mul_(~pruned_channels)

        del module._buffers["pruned_input"]
        return output

    # Define hook that indicates which neurons need to go
    @torch.no_grad()
    def __remove_zero_rows_hook(module, input_, output, name):
        input_ = input_[0][0]

        ##########################################
        #     STEP 1 - REMOVE INPUT CHANNELS     #
        ##########################################

        # Compute non-zero input channels indices
        nonzero_idx = ~(input_.view(input_.shape[0], -1).sum(dim=1) == 0)

        module.weight = torch.nn.Parameter(module.weight[:, nonzero_idx])
        module.in_features = module.weight.shape[1]

        ###########################################
        #     STEP 2 - REMOVE OUTPUT CHANNELS     #
        ###########################################

        output = torch.ones_like(output) * float('nan')

        # Compute non-zero output channels indices
        shape = module.weight.shape
        nonzero_idx = ~(module.weight.view(shape[0], -1).sum(dim=1) == 0)

        if name in pinned_outputs:
            print("Will be ignored!")
        else:
            output = torch.zeros_like(output)
            output[:, nonzero_idx] = float('nan')

            module.bias = torch.nn.Parameter(module.bias[nonzero_idx])
            module.weight = torch.nn.Parameter(module.weight[nonzero_idx])

        module.out_features = module.weight.shape[0]

        return output

    # Add hook to every layer in the network
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        handle = module.register_forward_pre_hook(__remove_nan2)
        handles.append(handle)
        handle = module.register_forward_hook(
            lambda m, i, o, n=name: __propagate_biases_hook(m, i, o, n)
        )
        handles.append(handle)

    # Feed the network some input to run the hooks
    nan_input = torch.ones_like(typical_input) * float("nan")
    model(nan_input)

    # Remove the handles from the network to prevent side effects when using it
    for handle in handles:
        handle.remove()

    # Add hook to every layer in the network
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        handle = module.register_forward_pre_hook(__remove_nan)
        handles.append(handle)
        handle = module.register_forward_hook(
            lambda m, i, o, n=name: __remove_zero_rows_hook(m, i, o, n)
        )
        handles.append(handle)

    # Feed the network some input to run the hooks
    nan_input = torch.ones_like(typical_input) * float("nan")
    model(nan_input)

    # Remove the handles from the network to prevent side effects when using it
    for handle in handles:
        handle.remove()

    return model
