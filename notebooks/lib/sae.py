import torch

from huggingface_hub import hf_hub_download

class SparseAutoEncoder(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.device = device
        self.encoder_linear = torch.nn.Linear(d_in, d_hidden)
        self.decoder_linear = torch.nn.Linear(d_hidden, d_in)
        self.dtype = dtype
        self.to(self.device, self.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of data using a linear, followed by a ReLU."""
        return torch.nn.functional.relu(self.encoder_linear(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a batch of data using a linear."""
        return self.decoder_linear(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """SAE forward pass. Returns the reconstruction and the encoded features."""
        f = self.encode(x)
        return self.decode(f), f

def download_and_load_sae(
    sae_name: str,
    d_model: int,
    expansion_factor: int,
    device: torch.device = torch.device("cpu"),
):
    file_path = hf_hub_download(
        repo_id=f"Goodfire/{sae_name}",
        filename=f"{sae_name}.pth",
        repo_type="model"
    )

    return load_sae(
        file_path,
        d_model=d_model,
        expansion_factor=expansion_factor,
        device=device,
    )


def load_sae(
    path: str,
    d_model: int,
    expansion_factor: int,
    device: torch.device = torch.device("cpu"),
):
    sae = SparseAutoEncoder(
        d_model,
        d_model * expansion_factor,
        device,
    )
    sae_dict = torch.load(
        path, weights_only=True, map_location=device
    )
    sae.load_state_dict(sae_dict)

    return sae


def create_sae_steering_vector_latents(sae, features):
    feature_encoding = torch.zeros(sae.d_hidden, device=sae.device, dtype=torch.bfloat16)
    for index, strength in features.items():
        feature_encoding[index] = strength

    return feature_encoding

def create_sae_steering_vector(sae, features):
    feature_encoding = create_sae_steering_vector_latents(sae, features)
    
    # Project SAE feature back to residual stream space
    sae_vector = sae.decode(feature_encoding.unsqueeze(0).unsqueeze(0))
    sae_vector = sae_vector / torch.norm(sae_vector)
    
    return sae_vector

def latents_to_feature_map(latents):
    features = {}

    for i, value in enumerate(latents):
        if value != 0:
            features[i] = value

    return features
