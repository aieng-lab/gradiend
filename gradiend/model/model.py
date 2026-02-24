"""
GRADIEND core model definitions (weights-only).

This module defines:
- GradiendModel: weights-only GRADIEND encoder/decoder (no base-model context).

For the parameter mapping-aware variant, see gradiend.model.param_mapped.ParamMappedGradiendModel.
For the variant in combination of a base model, see gradiend.model.model_with_gradiend.ModelWithGradiend.
"""

import copy
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from gradiend.model.layers import LargeLinear
from gradiend.model.utils import get_activation
from gradiend.util.logging import get_logger
from gradiend.util import convert_tuple_keys_recursively

logger = get_logger(__name__)



class GradiendModel(nn.Module):
    """
    GRADIEND - GRADIent ENcoder Decoder model implementation (weights-only):
    maps model gradients to a low-dimensional latent space and back.

    Proposed by Drechsel et al. 2025 (https://arxiv.org/abs/2502.01406).

    This class holds ONLY the neural components (encoder/decoder) + utilities that depend solely on
    GRADIEND parameters:
    - forward / forward_encoder (tensor input space)
    - weight-derived importance scores (encoder/decoder/decoder-bias/decoder-sum)
    - internal prune primitive that physically reduces input_dim (slices weights; no mapping logic)
    - save_pretrained / from_pretrained for weights + architecture + metadata

    Saving:
    - Weights: model.safetensors if available, else pytorch_model.bin
    - Config: config.json (format_version=0)
    - Run info: training.json (optional; if kwargs contains "training")

    Use ParamMappedGradiendModel when you need a parameter mapping or dict-of-gradients I/O.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        activation_encoder: str = "tanh",
        activation_decoder: str = "id",
        bias_decoder: bool = True,
        torch_dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        device_encoder: Optional[torch.device] = None,
        device_decoder: Optional[torch.device] = None,
        lazy_init: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a weights-only GRADIEND model (i.e., a GRADIEND encoder-decoder without base-model context
        but ).

        Activation functions (case-insensitive):
        tanh, relu, leakyrelu, gelu, silu, elu, sigmoid, smht (hardtanh), id (identity).
        Defaults (paper): encoder tanh, decoder id.

        Args:
            input_dim: Size of the GRADIEND input space (total selected gradient entries).
            latent_dim: Size of the latent bottleneck.
            activation_encoder: Encoder activation name (case-insensitive).
            activation_decoder: Decoder activation name. If falsy, uses encoder activation
                but with decoder-appropriate defaults via get_activation.
            bias_decoder: Whether the decoder linear layer uses a bias term.
            torch_dtype: dtype used for model parameters.
            device: Optional default device for both encoder and decoder when specific
                devices are not provided.
            device_encoder: Device for encoder parameters.
            device_decoder: Device for decoder parameters.
            lazy_init: If True, do not create encoder/decoder weights here. Build them later
                via prune (with pruned size) or _build_encoder_decoder (full size).
            **kwargs: Additional metadata stored in `self.kwargs` and serialized into config.json metadata
                on save. Non-JSONable values are stringified in a safe way.
        """
        super().__init__()
        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_encoder = device_encoder or device or default_device
        self.device_decoder = device_decoder or device or default_device

        self.latent_dim = int(latent_dim)
        self.input_dim = int(input_dim)
        if self.input_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("Input and latent dimensions must be positive integers.")

        self.activation = activation_encoder.lower()
        self.activation_decoder = activation_decoder
        self.bias_decoder = bool(bias_decoder)
        self.torch_dtype = torch_dtype
        self._lazy_init = bool(lazy_init)

        self.kwargs = kwargs
        if "base_model" in self.kwargs and hasattr(self.kwargs["base_model"], "name_or_path"):
            self.kwargs["base_model"] = self.kwargs["base_model"].name_or_path

        if activation_decoder:
            self.activation_decoder = activation_decoder
        else:
            self.activation_decoder = self.activation

        if self._lazy_init:
            self.encoder = None
            self.decoder = None
        else:
            activation_fnc = get_activation(self.activation, encoder=True)
            if activation_decoder:
                activation_fnc_decoder = get_activation(activation_decoder)
            else:
                activation_fnc_decoder = get_activation(self.activation, encoder=False)

            self.encoder = nn.Sequential(
                LargeLinear(self.input_dim, self.latent_dim, dtype=torch_dtype, device=self.device_encoder),
                activation_fnc,
            )
            self.decoder = nn.Sequential(
                LargeLinear(self.latent_dim, self.input_dim, bias=self.bias_decoder, dtype=torch_dtype, device=self.device_decoder),
                activation_fnc_decoder,
            )

            # initialize decoder similar scale as encoder
            x = self.encoder[0].weight.max().item()
            nn.init.uniform_(self.decoder[0].weight, -x, x)
            if self.bias_decoder:
                nn.init.uniform_(self.decoder[0].bias, -x, x)

        self.ctr = 0

    def _build_encoder_decoder(self, input_dim: int) -> None:
        """
        Instantiate encoder and decoder with given input_dim. Used for lazy init:
        either after prune (with pruned size) or before first use (with full size).
        """
        if self.encoder is not None and self.decoder is not None:
            return
        self.input_dim = int(input_dim)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        activation_fnc = get_activation(self.activation, encoder=True)
        if self.activation_decoder and self.activation_decoder != self.activation:
            activation_fnc_decoder = get_activation(self.activation_decoder)
        else:
            activation_fnc_decoder = get_activation(self.activation, encoder=False)

        self.encoder = nn.Sequential(
            LargeLinear(self.input_dim, self.latent_dim, dtype=self.torch_dtype, device=self.device_encoder),
            activation_fnc,
        )
        self.decoder = nn.Sequential(
            LargeLinear(self.latent_dim, self.input_dim, bias=self.bias_decoder, dtype=self.torch_dtype, device=self.device_decoder),
            activation_fnc_decoder,
        )

        # initialize decoder similar scale as encoder
        x = self.encoder[0].weight.max().item()
        nn.init.uniform_(self.decoder[0].weight, -x, x)
        if self.bias_decoder:
            nn.init.uniform_(self.decoder[0].bias, -x, x)

    def to(
        self,
        device: Union[str, torch.device, None] = None,
        *,
        device_encoder: Optional[Union[str, torch.device]] = None,
        device_decoder: Optional[Union[str, torch.device]] = None,
    ) -> "GradiendModel":
        """
        Move encoder and decoder to the requested devices.

        - If device_encoder or device_decoder is provided, moves only those submodules.
        - If device is provided (and no split devices), moves both to that device.
        - If device_encoder/device_decoder is None, leaves that submodule's placement unchanged.
        - When encoder/decoder are not yet built (lazy init), only updates target device attributes.
        """
        if device_encoder is not None or device_decoder is not None:
            if device_encoder is not None:
                self.device_encoder = torch.device(device_encoder) if isinstance(device_encoder, str) else device_encoder
                if self.encoder is not None:
                    self.encoder.to(self.device_encoder)
            if device_decoder is not None:
                self.device_decoder = torch.device(device_decoder) if isinstance(device_decoder, str) else device_decoder
                if self.decoder is not None:
                    self.decoder.to(self.device_decoder)
            return self
        if device is not None:
            dev = torch.device(device) if isinstance(device, str) else device
            self.device_encoder = dev
            self.device_decoder = dev
            if self.encoder is not None:
                self.encoder.to(dev)
            if self.decoder is not None:
                self.decoder.to(dev)
        return self

    def cpu(self) -> "GradiendModel":
        """Move encoder and decoder to CPU."""
        return self.to("cpu")

    def cuda(self, device: Optional[Union[int, str, torch.device]] = None) -> "GradiendModel":
        """Move encoder and decoder to CUDA."""
        if device is None:
            return self.to("cuda")
        if isinstance(device, int):
            return self.to(f"cuda:{device}")
        return self.to(device)

    # ----------------- norms -----------------
    def _require_built(self) -> None:
        """Raise if encoder/decoder are not yet built (lazy init)."""
        if self.encoder is None or self.decoder is None:
            raise RuntimeError(
                "Encoder/decoder weights not yet built. For lazy-init gradiend with pre_prune, "
                "call prune() first. Otherwise call _build_encoder_decoder(input_dim)."
            )

    def _ensure_built(self) -> None:
        """Build encoder/decoder with current input_dim if not yet built (lazy init)."""
        if self.encoder is None or self.decoder is None:
            self._build_encoder_decoder(self.input_dim)

    @property
    def base_model_id(self) -> str:
        """
        Base model identifier stored in kwargs.

        Raises:
            ValueError: If base_model is missing from kwargs.
        """
        base_model = self.kwargs.get("base_model")
        if not base_model:
            raise ValueError("base_model missing from gradiend.kwargs")
        return base_model

    @property
    def decoder_norm(self) -> float:
        """
        L2 norm of the decoder weight matrix.

        Returns:
            Scalar float of the decoder's weight L2 norm.
        """
        self._require_built()
        return torch.norm(self.decoder[0].weight, p=2).item()

    @property
    def encoder_norm(self) -> float:
        """
        L2 norm of the encoder weight matrix.

        Returns:
            Scalar float of the encoder's weight L2 norm.
        """
        self._require_built()
        return torch.norm(self.encoder[0].weight, p=2).item()

    # ----------------- importance -----------------
    def get_weight_importance(self, part: str = "decoder-weight") -> torch.Tensor:
        """
        Importance per GRADIEND input dimension (length = input_dim), on CPU.
        Args:
            part: Which component to use for importance aggregation:
                - "encoder-weight": L1 over encoder weight columns
                - "decoder-weight": L1 over decoder weight rows
                - "decoder-bias": absolute decoder bias
                - "decoder-sum": absolute(sum(weight_row) + bias)

        Returns:
            1D CPU float tensor of length input_dim, where higher means more
            influential according to the chosen aggregation.
        """
        self._require_built()
        part = (part or "decoder-weight").lower()
        vec = self.get_update_vector(part=part).detach().cpu()

        if part in ("decoder-sum", "decoder-bias"):
            return vec.abs()

        if part == "decoder-weight":
            w = vec.view(self.decoder[0].linear.weight.shape)
            return w.abs().sum(dim=1)
        if part == "encoder-weight":
            w = vec.view(self.encoder[0].linear.weight.shape)
            return w.abs().sum(dim=0)

        raise ValueError(
            f"part must be 'encoder-weight', 'decoder-weight', 'decoder-bias', or 'decoder-sum', got {part!r}"
        )


    def get_update_vector(self, part: str = "decoder-weight") -> torch.Tensor:
        """
        Return a flattened weight-derived update vector.

        Args:
            part: Which component to use for the update vector:
                - "decoder-weight": decoder weight vector (flattened)
                - "decoder-bias": decoder bias vector
                - "decoder-sum": decoder weight vector + bias
                - "encoder-weight": encoder weight vector (flattened)

        Returns:
            1D tensor in GRADIEND input space derived from the requested component.
        """
        self._require_built()
        part = (part or "decoder-weight").lower()

        if part == "decoder-weight":
            return self.decoder[0].linear.weight.flatten()
        if part == "decoder-bias":
            b = self.decoder[0].linear.bias
            if b is None:
                return torch.zeros(self.input_dim, dtype=self.torch_dtype, device=self.decoder[0].linear.weight.device)
            return b
        if part == "decoder-sum":
            w = self.decoder[0].linear.weight
            b = self.decoder[0].linear.bias
            row_sum = w.sum(dim=1)
            return row_sum if b is None else row_sum + b
        if part == "encoder-weight":
            return self.encoder[0].linear.weight.flatten()
        raise ValueError(f"part must be 'encoder-weight', 'decoder-weight', 'decoder-bias', or 'decoder-sum', got {part!r}")

    def get_topk_weights(self, part: str = "decoder-weight", topk: Union[int, float] = 1000) -> List[int]:
        """
        Return the top-k input indices by importance score.

        Args:
            part: Importance source passed to get_weight_importance.
                Options: "encoder-weight", "decoder-weight", "decoder-bias", "decoder-sum".
            topk: Number of indices to return (clipped to input_dim) or a proportion in (0, 1].

        Returns:
            List of input indices (length k) sorted by descending importance.
        """
        imp = self.get_weight_importance(part=part)
        if isinstance(topk, float):
            if not (0.0 < topk <= 1.0):
                raise ValueError("topk float must be in (0, 1]")
            k = int(math.ceil(topk * imp.numel()))
        else:
            k = int(topk)
        k = min(max(k, 1), imp.numel())
        _, idx = torch.topk(imp, k=k, largest=True, sorted=True)
        return idx.tolist()

    def _ensure_input(self, x: torch.Tensor):
        if x.numel() != self.input_dim:
            raise ValueError(f"Input tensor has incorrect size {x.numel()}, expected {self.input_dim}")

        # check if reshape is needed (e.g., from (input_dim, 1) or (1, input_dim))
        if x.dim() > 1:
            x = x.view(-1)

        if x.dtype != self.torch_dtype:
            x = x.to(self.torch_dtype)

        if x.device != self.device_encoder:
            x = x.to(self.device_encoder)

        return x

    # ----------------- forward -----------------
    def forward(
        self, x: torch.Tensor, return_encoded: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for tensor input already in GRADIEND input space.

        Args:
            x: 1D tensor of shape (input_dim,) representing a flattened
                gradient vector in GRADIEND input space.
            return_encoded: If True, also return the latent encoding.

        Returns:
            If return_encoded is False:
                Decoded tensor of shape (input_dim,).
            If return_encoded is True:
                Tuple (decoded, encoded), where:
                - decoded: tensor of shape (input_dim,)
                - encoded: tensor of shape (latent_dim,)
        """
        self._require_built()
        self._ensure_input(x)

        encoded = self.encoder(x)
        if encoded.device != self.device_decoder:
            encoded = encoded.to(self.device_decoder)
        decoded = self.decoder(encoded)
        self.ctr += 1

        return (decoded, encoded) if return_encoded else decoded

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder-only forward for tensor input.

        Args:
            x: 1D tensor of shape (input_dim,) in GRADIEND input space.

        Returns:
            Encoded tensor of shape (latent_dim,).
        """
        self._require_built()
        x = self._ensure_input(x)
        return self.encoder(x)

    # ----------------- prune primitive -----------------
    def _prune_input_dims(
        self,
        keep_idx: torch.Tensor,
        *,
        inplace: bool = False,
        return_index_map: bool = False,
    ):
        """
        INTERNAL: physically prune input_dim and output_dim by slicing encoder/decoder weights.

        - encoder: slice columns (latent_dim, input_dim) -> (latent_dim, new_in)
        - decoder: slice rows    (input_dim, latent_dim) -> (new_in, latent_dim)
        - bias:   slice entries  (input_dim,)            -> (new_in,)
        """
        if not torch.is_tensor(keep_idx):
            raise TypeError(f"keep_idx must be a torch.Tensor, got {type(keep_idx)}")
        if keep_idx.dim() != 1 or keep_idx.numel() == 0:
            raise ValueError("keep_idx must be a non-empty 1D tensor")
        if keep_idx.dtype not in (torch.int64, torch.long, torch.int32):
            raise TypeError(f"keep_idx must be integer dtype, got {keep_idx.dtype}")

        m = self if inplace else copy.deepcopy(self)

        enc_ll = m.encoder[0].linear
        dec_ll = m.decoder[0].linear
        enc_w, enc_b = enc_ll.weight, enc_ll.bias
        dec_w, dec_b = dec_ll.weight, dec_ll.bias

        keep_idx_dev = keep_idx.detach().to(dtype=torch.long, device=enc_w.device)
        if keep_idx_dev.min().item() < 0 or keep_idx_dev.max().item() >= m.input_dim:
            raise ValueError(f"keep_idx out of bounds for input_dim={m.input_dim}")

        new_in = int(keep_idx_dev.numel())

        new_enc = LargeLinear(new_in, m.latent_dim, bias=True, dtype=m.torch_dtype, device=m.device_encoder)
        new_dec = LargeLinear(m.latent_dim, new_in, bias=m.bias_decoder, dtype=m.torch_dtype, device=m.device_decoder)

        with torch.no_grad():
            new_enc.linear.weight.copy_(enc_w[:, keep_idx_dev].to(new_enc.linear.weight.device))
            if new_enc.linear.bias is not None and enc_b is not None:
                new_enc.linear.bias.copy_(enc_b.to(new_enc.linear.bias.device))

            keep_idx_dec = keep_idx_dev if dec_w.device == keep_idx_dev.device else keep_idx_dev.to(dec_w.device)
            new_dec.linear.weight.copy_(dec_w[keep_idx_dec, :].to(new_dec.linear.weight.device))
            if new_dec.linear.bias is not None:
                if dec_b is None:
                    new_dec.linear.bias.zero_()
                else:
                    new_dec.linear.bias.copy_(dec_b[keep_idx_dec].to(new_dec.linear.bias.device))

        m.encoder[0] = new_enc
        m.decoder[0] = new_dec
        m.input_dim = new_in

        return (m, keep_idx.detach().to("cpu").long()) if return_index_map else m

    def prune(
        self,
        *,
        topk: Union[int, float, None] = None,
        threshold: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
        part: str = "decoder-weight",
        importance: Optional[torch.Tensor] = None,
        inplace: bool = False,
        return_mask: bool = False,
    ) -> Union["GradiendModel", Tuple["GradiendModel", torch.Tensor]]:
        """
        Physically prune the model (reduce input_dim) by selecting important input dimensions.
        
        Selection order: mask -> threshold -> topk.
        
        Args:
            topk: int (absolute) or float in (0,1] (relative fraction among remaining dims).
            threshold: keep dims with importance >= threshold.
            mask: optional bool tensor of shape (input_dim,) in current input space.
            part: 'encoder-weight' | 'decoder-weight' | 'decoder-bias' | 'decoder-sum' (used when importance is None).
            importance: optional 1D tensor of length input_dim; used instead of get_weight_importance(part) when provided.
            inplace: modify this instance if True, else return a deepcopy.
            return_mask: if True, also return final combined_mask (original input space).
        
        Returns:
            If return_mask is False:
                The pruned GradiendModel (self or a deepcopy depending on `inplace`).
            If return_mask is True:
                Tuple (model, combined_mask) where combined_mask is a bool tensor
                of shape (old_input_dim,) indicating kept dimensions.
        """
        if topk is None and threshold is None and mask is None:
            raise ValueError("At least one of topk, threshold, mask must be provided.")

        # topk=1.0 (float) means no pruning (return self); topk int 1 means keep top-1 dimension
        if topk is not None and isinstance(topk, float) and topk == 1.0:
            m = self if inplace else copy.deepcopy(self)
            if return_mask:
                old_input_dim = int(self.input_dim)
                full_mask = torch.ones(old_input_dim, dtype=torch.bool, device="cpu")
                return m, full_mask
            return m

        old_input_dim = int(self.input_dim)
        combined = torch.ones(old_input_dim, dtype=torch.bool, device="cpu")
        
        if mask is not None:
            if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.shape != (old_input_dim,):
                raise ValueError(f"mask must be bool tensor with shape ({old_input_dim},)")
            combined &= mask.detach().to("cpu")
        
        importance_scores = None
        if threshold is not None or topk is not None:
            if importance is not None:
                importance_scores = importance.detach().to("cpu")
            else:
                importance_scores = self.get_weight_importance(part=part)
            
            if threshold is not None:
                combined &= (importance_scores >= threshold)
            
            if topk is not None:
                if isinstance(topk, float):
                    if not (0.0 < topk <= 1.0):
                        raise ValueError("topk float must be in (0, 1]")
                    k = int(math.ceil(topk * combined.sum().item()))
                else:
                    k = int(topk)
                k = min(max(k, 1), combined.sum().item())
                if k < combined.sum().item():
                    masked_importance = importance_scores.clone()
                    masked_importance[~combined] = float('-inf')
                    _, top_indices = torch.topk(masked_importance, k=k, largest=True, sorted=True)
                    new_combined = torch.zeros(old_input_dim, dtype=torch.bool)
                    new_combined[top_indices] = True
                    combined = new_combined
        
        keep_idx = torch.nonzero(combined, as_tuple=False).squeeze(-1)
        if keep_idx.numel() == 0:
            raise ValueError("Pruning resulted in zero dimensions")
        
        result = self._prune_input_dims(keep_idx, inplace=inplace, return_index_map=return_mask)
        if return_mask:
            pruned_model, index_map = result
            # Convert index_map to bool mask
            final_mask = torch.zeros(old_input_dim, dtype=torch.bool)
            final_mask[index_map] = True
            return pruned_model, final_mask
        return result

    # ----------------- save/load (weights-only) -----------------
    def save_pretrained(self, save_directory: str, use_safetensors: Optional[bool] = None, **kwargs: Any) -> None:
        """
        Save weights + config.json (+ optional training.json).

        Notes:
        - safetensors is used if available unless use_safetensors=False.
        - training info: if kwargs contains "training", it is written to training.json and removed from config metadata.

        Args:
            save_directory: Folder to write model files into.
            use_safetensors: If True, require safetensors. If False, force PyTorch
                bin format. If None, prefer safetensors when available.
            **kwargs: Extra metadata to store in config.json.

        Returns:
            None.
        """
        self._require_built()
        os.makedirs(save_directory, exist_ok=True)

        prefer_safetensors = (use_safetensors is not False)
        used_safetensors = False

        # ---- weights ----
        if prefer_safetensors:
            try:
                from safetensors.torch import save_file
                save_file(self.state_dict(), os.path.join(save_directory, "model.safetensors"))
                used_safetensors = True
            except ImportError as e:
                if use_safetensors is True:
                    raise ImportError("safetensors not installed, cannot save as safetensors") from e

        if not used_safetensors:
            torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

        # ---- config ----
        self.kwargs.update(kwargs)
        meta = self._serialize_kwargs()

        run_info = meta.pop("training", None)

        config = {
            "format_version": 0,
            "architecture": {
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "activation_encoder": self.activation,
                "activation_decoder": self.activation_decoder,
                "bias_decoder": self.bias_decoder,
                "torch_dtype": str(self.torch_dtype).replace("torch.", ""),
            },
            "mapping": None,  # filled by ParamMappedGradiendModel; core leaves None
            "metadata": meta,
        }

        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        if run_info is not None:
            with open(os.path.join(save_directory, "training.json"), "w") as f:
                json.dump(run_info, f, indent=2)

    def _serialize_kwargs(self) -> Dict[str, Any]:
        """Serialize kwargs, filtering out non-JSON objects."""
        kwargs = self.kwargs.copy()
        out: Dict[str, Any] = {}
        for k, v in kwargs.items():
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                if hasattr(v, "id"):
                    out[k] = {"id": v.id, "_type": type(v).__name__}
                else:
                    out[k] = {"_type": type(v).__name__, "_repr": str(v)[:120]}
        return convert_tuple_keys_recursively(out)

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
        device_encoder: Optional[torch.device] = None,
        device_decoder: Optional[torch.device] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "GradiendModel":
        """
        Load weights + config.json (weights-only).
        ParamMappedGradiendModel overrides to also load mapping.

        Args:
            load_directory: Directory containing model files.
            device_encoder: Optional device override for encoder parameters.
            device_decoder: Optional device override for decoder parameters.
            torch_dtype: Optional dtype override. If None, uses dtype stored in config.json.

        Returns:
            Instantiated GradiendModel with loaded weights and metadata.
        """
        cfg_path = os.path.join(load_directory, "config.json")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        arch = cfg["architecture"]

        # dtype
        if torch_dtype is None:
            td = arch.get("torch_dtype", "float32")
            torch_dtype = getattr(torch, td, torch.float32)

        # weights
        st_path = os.path.join(load_directory, "model.safetensors")
        bin_path = os.path.join(load_directory, "pytorch_model.bin")
        if not os.path.exists(st_path) and not os.path.exists(bin_path):
            model_dir = os.path.join(load_directory, "model")
            st_path = os.path.join(model_dir, "model.safetensors")
            bin_path = os.path.join(model_dir, "pytorch_model.bin")

        if os.path.exists(st_path):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(st_path)
            except ImportError:
                # fallback to bin if present
                if os.path.exists(bin_path):
                    state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
                else:
                    raise
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No model weights found in {load_directory}")

        # instantiate
        meta = cfg.get("metadata") or {}
        model = cls(
            input_dim=arch["input_dim"],
            latent_dim=arch["latent_dim"],
            activation_encoder=arch.get("activation_encoder", "tanh"),
            activation_decoder=arch.get("activation_decoder", "id"),
            bias_decoder=arch.get("bias_decoder", True),
            torch_dtype=torch_dtype,
            device_encoder=device_encoder,
            device_decoder=device_decoder,
            **meta,
        )

        # attach training info (optional)
        training_path = os.path.join(load_directory, "training.json")
        if os.path.exists(training_path):
            try:
                with open(training_path, "r") as f:
                    model.kwargs.setdefault("training", json.load(f))
            except Exception:
                pass

        model.load_state_dict(state_dict)
        model.name_or_path = load_directory
        return model


    def pruned_length(self) -> int:
        """
        Return the current input_dim after pruning.

        Returns:
            Current input_dim as an integer.
        """
        return self.input_dim

    def __len__(self) -> int:
        """
        Return the current input_dim after pruning.

        Returns:
            Current input_dim as an integer.
        """
        return self.pruned_length()