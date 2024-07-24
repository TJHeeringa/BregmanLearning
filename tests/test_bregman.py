import pytest
import torch

from bregman import sparsify, AutoEncoder, row_density, column_density, simplify


def test_row_density():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3)
    )
    model[0].weight.data = torch.tensor([
        [0., 0., 0.],
        [1, 4, 5],
        [2, 7, 8],
        [3, 6, 9]
    ])
    assert row_density(model=model, absolute=False) == 0.75
    assert row_density(model=model, absolute=True) == 3


def test_column_density():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4)
    )
    model[0].weight.data = torch.tensor([
        [0., 1, 2, 3],
        [0., 4, 5, 6],
        [0., 7, 8, 9],
    ])
    assert column_density(model=model, absolute=False) == 0.75
    assert column_density(model=model, absolute=True) == 3


@pytest.mark.parametrize('run_count', range(10))
def test_latent_pod(run_count):
    pass


@pytest.mark.parametrize('in_place', [True, False])
@pytest.mark.parametrize('run_count', range(10))
@pytest.mark.parametrize('latent', [10, 100])
def test_simplify(run_count, in_place, latent):
    model = AutoEncoder(
        encoder_layers=[11, 20, latent],
        decoder_layers=[latent, 20, 11]
    )
    sparsify(model, 0.2)

    fom_size = model.fom_size
    reduced_sizes = []
    for parm in model.parameters():
        if len(parm.data.shape) > 1:
            reduced_sizes.append(torch.max(parm.data, dim=1).values.count_nonzero())
    reduced_sizes = reduced_sizes[:-1]  # last layer shouldn't be reduced

    if in_place:
        simplify(model, in_place)
    else:
        model = simplify(model)

    assert model.layers[1:-1] == reduced_sizes
    assert model.layers[0] == fom_size
    assert model.layers[-1] == fom_size


@pytest.mark.parametrize('matrix_size', range(1, 10, 4))
@pytest.mark.parametrize('density_level', torch.arange(start=0, end=1+1e-8, step=0.1))
@pytest.mark.parametrize('run_count', range(10))
def test_sparsify(matrix_size, density_level, run_count):
    model = torch.nn.Sequential(
        torch.nn.Linear(matrix_size, matrix_size)
    )
    sparsify(model, density_level, spectral_sparsity=False)

    zero_rows = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            w = m.weight.data
            zero_rows += (w.norm(dim=1) == 0).sum().item()
    expected_zero_rows = torch.floor((1-density_level) * matrix_size)
    assert zero_rows == expected_zero_rows.int()
