import torch

from .models import AutoEncoder


@torch.no_grad()
def latent_pod(model: AutoEncoder, relative_error_tol=1e-6) -> AutoEncoder:
    """Applies POD on the latent space of the provided AutoEncoder.

    The function computes a singular value decomposition of the last weight matrix of the encoder. It will truncate the
    singular values. The last weight matrix of the encoder will be replaced by the truncated singular values and
    associated eigenvectors. The U matrix of the SVD will be passed along to the decoder.

    See <> for the paper introducing the operation.

    :param AutoEncoder model: AutoEncoder of which the latent space should be truncated.
    :param float relative_error_tol: Tolerance to use for truncating the singular values in the SVD.
    :return: AutoEncoder with truncated latent space
    :rtype: AutoEncoder
    """
    U, S, Vh = torch.linalg.svd(model.encoder[-1].weight, full_matrices=False)
    Sigma = S**2
    relative_error = (torch.sum(Sigma, dim=0) - torch.cumsum(Sigma, dim=0)) / torch.sum(Sigma, dim=0)
    ldim: int = torch.count_nonzero(relative_error >= relative_error_tol).item() + 1

    if ldim >= model.latent_size("full"):
        return model

    new_encoder_layers = [*model.encoder_layers]
    new_encoder_layers[-1] = ldim

    new_decoder_layers = [*model.decoder_layers]
    new_decoder_layers[0] = ldim

    pod_model = AutoEncoder(
        encoder_layers=new_encoder_layers,
        decoder_layers=new_decoder_layers,
    )

    for i in range(len(new_encoder_layers) - 2):
        k = 2 * i
        pod_model.encoder[k].weight.data = model.encoder[k].weight.data
        pod_model.encoder[k].bias.data = model.encoder[k].bias.data

    pod_model.encoder[-1].bias.data = torch.zeros((ldim,))
    pod_model.encoder[-1].weight.data = torch.diag_embed(S[:ldim]) @ Vh[:ldim, :]
    pod_model.decoder[0].weight.data = model.decoder[0].weight.data @ U[:, :ldim]
    pod_model.decoder[0].bias.data = model.decoder[0].bias.data + model.decoder[0].weight @ model.encoder[-1].bias.data

    for i in range(len(new_decoder_layers) - 2):
        k = 2 * i + 2
        pod_model.decoder[k].weight.data = model.decoder[k].weight.data
        pod_model.decoder[k].bias.data = model.decoder[k].bias.data

    return pod_model
