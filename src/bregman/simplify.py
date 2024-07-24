import copy
import typing

import torch

if typing.TYPE_CHECKING:
    from .models import AutoEncoder

__all__ = ["simplify"]


def simplify(model: "AutoEncoder", in_place: bool = False) -> "AutoEncoder":
    from simplify import simplify as EIDOSLAB_simplify

    def model_to_layers(model_) -> tuple[list[int], list[int]]:
        layers = []
        for weight_matrix in model_.parameters():
            layers.append(weight_matrix.shape)

        return (
            [size[-1] for (i, size) in enumerate(layers) if len(size) > 1 and i <= len(layers) / 2],
            [size[-1] for (i, size) in enumerate(layers) if len(size) > 1 and i >= len(layers) / 2] + [layers[0][-1]],
        )

    def from_linear_expand_to_linear(module):
        linear_layer = torch.nn.Linear(module.weight.shape[1], len(module.zeros))

        # clear values initialized by default
        linear_layer.bias.data = torch.zeros(len(module.zeros))
        linear_layer.weight.data = torch.zeros(len(module.zeros), module.weight.shape[1])

        # set values based on linear expand values
        linear_layer.bias.data = module.zeros.scatter(dim=0, index=module.idxs, src=module.bias.data) + module.bf
        for idx, row in zip(module.idxs, module.weight):
            linear_layer.weight.data[idx] = row

        return linear_layer

    dummy_input = torch.ones([1, model.fom_size])
    pinned_out = [name for name, module in model.named_modules()][-1:]
    if in_place:
        EIDOSLAB_simplify(model, dummy_input, pinned_out=pinned_out)
    else:
        model = EIDOSLAB_simplify(copy.deepcopy(model), dummy_input, pinned_out=pinned_out)
    model.decoder[-1] = from_linear_expand_to_linear(model.decoder[-1])

    encoder_layers, decoder_layers = model_to_layers(model)
    model.encoder_layers = encoder_layers
    model.decoder_layers = decoder_layers

    return model
