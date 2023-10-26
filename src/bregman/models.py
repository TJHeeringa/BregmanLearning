import igraph
import torch


class AutoEncoder(torch.nn.Module):
    def __init__(
        self,
        encoder_layers: list[int],
        decoder_layers: list[int]
    ):
        super().__init__()
        # latent dimension must match
        assert encoder_layers[-1] == decoder_layers[0]
        # full order dimension must match
        assert encoder_layers[0] == decoder_layers[-1]

        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        # ===========  Encoder  ===========
        num_of_relu_needed = len(encoder_layers) - 2
        shifted_encoder_layers = encoder_layers[1:] + encoder_layers[:1]
        layers = []
        for k, (current_width, next_width) in enumerate(
            zip(encoder_layers, shifted_encoder_layers)
        ):
            if k <= num_of_relu_needed - 1:
                layers.extend(
                    [torch.nn.Linear(current_width, next_width), torch.nn.ReLU()]
                )
            elif k <= num_of_relu_needed:
                layers.append(torch.nn.Linear(current_width, next_width))
        self.encoder = torch.nn.Sequential(*layers)

        # ===========  Decoder  ===========
        num_of_relu_needed = len(decoder_layers) - 2
        shifted_decoder_layers = decoder_layers[1:] + decoder_layers[:1]
        layers = []
        for k, (current_width, next_width) in enumerate(
            zip(decoder_layers, shifted_decoder_layers)
        ):
            if k <= num_of_relu_needed - 1:
                layers.extend(
                    [torch.nn.Linear(current_width, next_width), torch.nn.ReLU()]
                )
            elif k <= num_of_relu_needed:
                layers.append(torch.nn.Linear(current_width, next_width))
        self.decoder = torch.nn.Sequential(*layers)

    @property
    def fom_size(self):
        """Return the full order dimension. This is simply the input size of the network."""
        return self.encoder_layers[0]

    @property
    def layers(self):
        """Return all layers of the nn. This is the encoder and decoder layers stuck back to back, with the duplicate
        center removed."""
        return self.encoder_layers[:-1] + self.decoder_layers

    def latent_size(self, direction: str = "encoder spectral", tol=0.0) -> int:
        """Computes the latent dimension based on the `direction` option.

        :param direction: one of {`encoder spectral`, `decoder spectral`, `decoder row`, `encoder row`, `decoder column`,
        `encoder column`, `latent POD`, `full`}
        :param tol: relative error tolerance to use when direction == `latent POD`
        """
        match direction:
            case "encoder spectral":
                return torch.count_nonzero(
                    torch.linalg.svdvals(self.encoder[-1].weight)
                )
            case "decoder spectral":
                return torch.count_nonzero(torch.linalg.svdvals(self.decoder[0].weight))
            case "decoder row":
                return torch.count_nonzero(self.decoder[0].weight.abs().sum(dim=0))
            case "encoder column":
                return torch.count_nonzero(self.encoder[-1].weight.abs().sum(dim=1))
            case "encoder row":
                return torch.count_nonzero(self.encoder[-1].weight.abs().sum(dim=0))
            case "full":
                return self.encoder_layers[-1]
            case "latent POD":
                S = torch.linalg.svdvals(self.encoder[-1].weight)
                Sigma = S ** 2
                relative_error = (torch.sum(Sigma, dim=0) - torch.cumsum(Sigma, dim=0)) / torch.sum(Sigma, dim=0)
                return min(torch.count_nonzero(relative_error >= tol).item() + 1, self.encoder_layers[-1])
            case _:
                raise ValueError("Invalid direction chosen")

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x

    def _color_vertices(self, vertices: igraph.VertexSeq) -> list[str]:
        """
        Used by `self.plot` to produce the colours for the plot. It gives living/fed neurons a blue colour and
        dead/starved ones a red colour. Neurons are dead/starved once they get no longer get any input.

        Args:
            vertices: Sequence of igraph vertices representing the neurons of the autoencoder.

        Returns:
            List with colour, where the first element corresponds to the first vertex etc.
        """
        def color_vertex(index: int, vertex_: igraph.Vertex):
            if index < self.encoder_layers[0]:
                return "blue"
            if vertex_.indegree() > 0:
                return "blue"
            return "red"

        colors = []
        for i, vertex in enumerate(vertices):
            colors.append(color_vertex(i, vertex))
        return colors

    def plot(self, save_to: str | None = None) -> None:
        """
        Function for visualizing the autoencoder. Each neuron will become a vertex in a directed graph with a colour
        depending on whether it's living/fed (blue) or dead/starved (red). Light gray edges will be drawn between
        neurons with a non-zero weight between them. All neurons will be vertically centered.

        Args:
            save_to: File to save output plot in, if given. Format "path/to/folder/plot_name.extension". Given extension
             determines in the format in which the plot is stored.

        """
        num_vertices = 0
        edges = []
        layout = []

        max_matrix_index = len(self.encoder_layers) + len(self.decoder_layers) - 3
        max_width = max(max(self.encoder_layers), max(self.encoder_layers))
        for i, weight_matrix in enumerate(get_weights_linear(self)):
            in_start = num_vertices
            in_end = out_start = num_vertices + weight_matrix.shape[1]
            out_end = out_start + weight_matrix.shape[0]

            input_vertices = range(in_start, in_end)
            output_vertices = range(out_start, out_end)

            for in_ in input_vertices:
                for out in output_vertices:
                    if abs(weight_matrix[out - out_start, in_ - in_start]) > 0:
                        edges.append([in_, out])

            input_top_offset = (max_width - weight_matrix.shape[1]) / 2

            layout.extend([[i, in_ - in_start + input_top_offset] for in_ in input_vertices])

            num_vertices += weight_matrix.shape[1]

            if i == max_matrix_index:
                num_vertices += weight_matrix.shape[0]
                output_top_offset = (max_width - weight_matrix.shape[0]) / 2
                layout.extend([[i + 1, out - out_start + output_top_offset] for out in output_vertices])

        graph = igraph.Graph(n=num_vertices, edges=edges, directed=True)

        visual_style = {
            "vertex_size": 10,
            "edge_arrow_size": 0.1,
            "vertex_color": self._color_vertices(graph.vs),
            "edge_color": "#778899"
        }
        return igraph.plot(graph, save_to, layout=layout, **visual_style)
