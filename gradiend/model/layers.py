import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LargeLinear(nn.Module):
    """
    A linear layer that handles both standard and very large input feature sizes
    by using chunked computation when the input dimension exceeds the limit
    for standard CUDA BLAS GEMM operations (approximately 2.14 billion).

    Internally, it uses a standard `torch.nn.Linear` layer for parameter
    management and efficient computation for smaller inputs.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``.
        in_chunk_size (int, optional): The size of the chunks to process the
            input dimension in when it exceeds the standard limit.
            Default: 2,000,000,000 (safety margin under int32 GEMM limits).
        out_chunk_size (int, optional): The size of the chunks to process the
            output dimension in when it exceeds the standard limit.
            Default: 2,000,000,000 (safety margin under int32 GEMM limits).
        dtype (torch.dtype, optional): Data type for the layer. Default: torch.float32.
        device (torch.device, optional): Device for the layer. Default: cuda if available, else cpu.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        in_chunk_size=2000000000,
        out_chunk_size=2000000000,
        dtype=torch.float32,
        device=None,
    ):
        super().__init__()
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        max_int32 = int(np.iinfo(np.int32).max)
        self.in_features = in_features
        self.out_features = out_features
        self.in_chunk_size = min(int(in_chunk_size), max_int32)
        self.out_chunk_size = min(int(out_chunk_size), max_int32)
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype, device=device)

    def forward(self, input):
        if input.device != self.linear.weight.device:
            input = input.to(self.linear.weight.device)

        input_size = input.size(-1)
        output_size = self.out_features
        max_size = np.iinfo(np.int32).max
        if input_size <= max_size and output_size <= max_size:
            return self.linear(input)
        elif output_size > max_size:
            num_out_chunks = (output_size + self.out_chunk_size - 1) // self.out_chunk_size
            output_shape = list(input.shape[:-1]) + [output_size]
            output = torch.zeros(*output_shape, device=input.device, dtype=input.dtype)

            for i in range(num_out_chunks):
                out_start = i * self.out_chunk_size
                out_end = min((i + 1) * self.out_chunk_size, output_size)
                weight_chunk = self.linear.weight[out_start:out_end, :].to(input.device)
                bias_chunk = self.linear.bias[out_start:out_end] if self.linear.bias is not None else None

                if input_size > max_size:
                    num_in_chunks = (input_size + self.in_chunk_size - 1) // self.in_chunk_size
                    intermediate_parts = []
                    for j in range(num_in_chunks):
                        in_start = j * self.in_chunk_size
                        in_end = min((j + 1) * self.in_chunk_size, input_size)
                        input_chunk = input[..., in_start:in_end]
                        weight_in_chunk = weight_chunk[:, in_start:in_end]
                        output_in_chunk = torch.matmul(
                            input_chunk.unsqueeze(-2), weight_in_chunk.T.unsqueeze(-1)
                        ).squeeze(-1)
                        intermediate_parts.append(output_in_chunk)
                    output_part = torch.sum(torch.stack(intermediate_parts, dim=-1), dim=-1)
                else:
                    output_part = torch.matmul(input, weight_chunk.T).squeeze(-1)

                if bias_chunk is not None:
                    output_part += bias_chunk.to(input.device)

                output[..., out_start:out_end] = output_part

            return output
        elif input_size > max_size:
            num_chunks = (input_size + self.in_chunk_size - 1) // self.in_chunk_size
            outputs = []
            for i in range(num_chunks):
                start = i * self.in_chunk_size
                end = min((i + 1) * self.in_chunk_size, input_size)
                input_chunk = input[..., start:end]
                weight_chunk = self.linear.weight[:, start:end].to(input.device)
                output_chunk = F.linear(input_chunk, weight_chunk, None)
                outputs.append(output_chunk)

            output = torch.sum(torch.stack(outputs, dim=-1), dim=-1)
            bias = self.linear.bias
            if bias is not None:
                output += bias.to(input.device)
            return output
        else:
            raise ValueError()

    @property
    def weight(self):
        return self.linear.weight

    @weight.setter
    def weight(self, value):
        if not isinstance(value, torch.nn.Parameter):
            value = torch.nn.Parameter(value)
        self.linear.weight = value

    @property
    def bias(self):
        return self.linear.bias

    @bias.setter
    def bias(self, value):
        if value is not None and not isinstance(value, torch.nn.Parameter):
            value = torch.nn.Parameter(value)
        self.linear.bias = value
