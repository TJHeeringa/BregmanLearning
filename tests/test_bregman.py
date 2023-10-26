import torch

from bregman import sparsify, AutoEncoder, row_density, column_density


def test_row_density():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3)
    )
    model[0].weight.data = torch.tensor([
        [0, 0, 0],
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
        [0, 1, 2, 3],
        [0, 4, 5, 6],
        [0, 7, 8, 9],
    ])
    assert column_density(model=model, absolute=False) == 0.75
    assert column_density(model=model, absolute=True) == 3


@pytest.mark.parametrize('run_count', range(10))
def test_latent_pod(run_count):
    pass


@pytest.mark.parametrize('run_count', range(10))
def test_simplify(run_count):
    pass


@pytest.mark.parametrize('matrix_size', range(1, 10, 4))
@pytest.mark.parametrize('density_level', torch.arange(start=0, end=1+1e-8, step=0.1))
@pytest.mark.parametrize('run_count', range(10))
def test_sparsify(matrix_size, density_level, run_count):
    model = torch.nn.Sequential(
        torch.nn.Linear(matrix_size, matrix_size)
    )
    sparsify(model, density_level)

    zero_rows = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            w = m.weight.data
            zero_rows += (w.norm(dim=1) == 0).sum().item()
    expected_zero_rows = torch.ceil(density_level * matrix_size)
    assert zero_rows == expected_zero_rows
