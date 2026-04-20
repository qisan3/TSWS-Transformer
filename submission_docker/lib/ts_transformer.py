from typing import Optional, Any
import math
import logging
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer


def model_factory(config, data):
    logger = logging.getLogger(__name__)

    task = config['task']
    feat_dim = data.feature_df.shape[1]
    max_seq_len = config['data_window_len'] if config['data_window_len'] is not None else config['max_seq_len']

    if max_seq_len is None:
        try:
            max_seq_len = data.max_seq_len
        except AttributeError as x:
            print(
                "Data class does not define a maximum sequence length, so it must be defined with the script argument `max_seq_len`")
            raise x

    if (task == "imputation") or (task == "transduction"):
        if config['model'] == 'LINEAR':
            return DummyTSTransformerEncoder(feat_dim, max_seq_len, config['d_model'], config['num_heads'],
                                             config['num_layers'], config['dim_feedforward'], dropout=config['dropout'],
                                             pos_encoding=config['pos_encoding'], activation=config['activation'],
                                             norm=config['normalization_layer'], freeze=config['freeze'])
        elif config['model'] == 'transformer':
            return TSTransformerEncoder(feat_dim, max_seq_len, config['d_model'], config['num_heads'],
                                        config['num_layers'], config['dim_feedforward'], dropout=config['dropout'],
                                        pos_encoding=config['pos_encoding'], activation=config['activation'],
                                        norm=config['normalization_layer'], freeze=config['freeze'])

        # 这是我们添加的块
        elif config['model'] == 'cnn_transformer':
            logger.info("使用 CNN-Transformer (CNNTransformerEncoder) 模型。")
            return CNNTransformerEncoder(feat_dim, max_seq_len, config['d_model'], config['num_heads'],
                                         config['num_layers'], config['dim_feedforward'], dropout=config['dropout'],
                                         pos_encoding=config['pos_encoding'], activation=config['activation'],
                                         norm=config['normalization_layer'],
                                         cnn_channels=config['cnn_channels'],
                                         freeze=config['freeze'])

    if (task == "classification") or (task == "regression"):
        num_labels = len(data.class_names) if task == "classification" else data.labels_df.shape[
            1]  # dimensionality of labels
        if config['model'] == 'LINEAR':
            return DummyTSTransformerEncoderClassiregressor(feat_dim, max_seq_len, config['d_model'],
                                                            config['num_heads'],
                                                            config['num_layers'], config['dim_feedforward'],
                                                            num_classes=num_labels,
                                                            dropout=config['dropout'],
                                                            pos_encoding=config['pos_encoding'],
                                                            activation=config['activation'],
                                                            norm=config['normalization_layer'], freeze=config['freeze'])
        elif config['model'] == 'transformer':
            return TSTransformerEncoderClassiregressor(feat_dim, max_seq_len, config['d_model'],
                                                       config['num_heads'],
                                                       config['num_layers'], config['dim_feedforward'],
                                                       num_classes=num_labels,
                                                       dropout=config['dropout'], pos_encoding=config['pos_encoding'],
                                                       activation=config['activation'],
                                                       norm=config['normalization_layer'], freeze=config['freeze'])

        # --- 添加这个新分支 ---
        elif config['model'] == 'cnn_transformer':
            logger.info("使用 CNN-Transformer (CNNTransformerEncoderClassiregressor) 模型进行回归/分类。")
            return CNNTransformerEncoderClassiregressor(feat_dim, max_seq_len, config['d_model'],
                                                        config['num_heads'],
                                                        config['num_layers'], config['dim_feedforward'],
                                                        num_classes=num_labels,
                                                        dropout=config['dropout'], pos_encoding=config['pos_encoding'],
                                                        activation=config['activation'],
                                                        norm=config['normalization_layer'],
                                                        cnn_channels=config['cnn_channels'],
                                                        freeze=config['freeze'])
        # --- 新代码结束 ---

    else:
        raise ValueError("Model class for task '{}' does not exist".format(task))


def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                is_causal: bool = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: if specified, applies a causal mask as attention mask (optional).

        Shape:
            see the docs in Transformer class.
        """
        # 处理PyTorch 2.0+的is_causal参数
        attn_kwargs = {}
        if is_causal:
            # 如果is_causal为True，需要创建因果掩码
            seq_len = src.size(0)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=src.device) * float('-inf'), diagonal=1)
            attn_kwargs['attn_mask'] = causal_mask
        elif src_mask is not None:
            attn_kwargs['attn_mask'] = src_mask

        src2 = self.self_attn(src, src, src,
                              key_padding_mask=src_key_padding_mask,
                              **attn_kwargs)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                             dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer

        # 兼容PyTorch 2.0+的调用方式
        try:
            # PyTorch 2.0+ 版本
            output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks, is_causal=False)
        except TypeError:
            # PyTorch 1.x 版本
            output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)

        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output


class TSTransformerEncoderClassiregressor(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    
    (!!!) UPDATED: Replaced the final layer with a 128 -> 64 -> 32 -> 1 MLP head (!!!)
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                             dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        
        # (!!!) This now builds the 128 -> 64 -> 32 -> 1 head (!!!)
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)

    def build_output_module(self, d_model, max_len, num_classes):
            """
            (!!!) This is the CORRECTED 256 -> 128 -> 64 -> 32 -> 1 MLP head (!!!)
            
            It takes the d_model (e.g., 256) tensor from the
            global average pooling in the forward() pass.
            
            Note: 'max_len' is no longer used here.
            """
            return nn.Sequential(
                # d_model (256) -> 128
                nn.Linear(d_model, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                
                # 128 -> 64
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                
                # 64 -> 32
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                
                # Final output layer (32 -> 1)
                nn.Linear(32, num_classes)
            )

    # def forward(self, X, padding_masks):
    #     """
    #     Args:
    #         X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
    #         padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    #     Returns:
    #         output: (batch_size, num_classes)
    #     """

    #     # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
    #     inp = X.permute(1, 0, 2)
    #     inp = self.project_inp(inp) * math.sqrt(
    #         self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
    #     inp = self.pos_enc(inp)  # add positional encoding
    #     # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer

    #     # try:
    #     #     output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks, is_causal=False)
    #     # except TypeError:
    #     #     output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)
    #     # 兼容PyTorch 2.0+的调用方式
    #     # (!!!) FIX: Removed src_key_padding_mask=~padding_masks to avoid MPS error (!!!)
    #     try:
    #         # PyTorch 2.0+ 版本
    #         output = self.transformer_encoder(inp, is_causal=False)
    #     except TypeError:
    #         # PyTorch 1.x 版本
    #         output = self.transformer_encoder(inp)

    #     output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
    #     output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
    #     output = self.dropout1(output)

    #     # (!!!) This is the Global Average Pooling (GAP) section (!!!)
    #     # It was already correct and works with the new MLP head.
    #     output = output * padding_masks.unsqueeze(-1)
    #     sum_features = output.sum(dim=1)
    #     num_non_padding = padding_masks.sum(dim=1).unsqueeze(1)
    #     num_non_padding = torch.clamp(num_non_padding, min=1e-9)
    #     output = sum_features / num_non_padding # (batch_size, d_model)
        
    #     # (!!!) Feed the (B, d_model) tensor into the new MLP head (!!!)
    #     output = self.output_layer(output) # (batch_size, num_classes)
    #     return output
    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding

        # (!!!) FIX: Removed src_key_padding_mask=~padding_masks to avoid MPS error (!!!)
        try:
            # PyTorch 2.0+ 版本
            output = self.transformer_encoder(inp, is_causal=False)
        except TypeError:
            # PyTorch 1.x 版本
            output = self.transformer_encoder(inp)

        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # (!!!) This is the Global Average Pooling (GAP) section (!!!)
        # This logic is correct.
        output = output * padding_masks.unsqueeze(-1)
        sum_features = output.sum(dim=1)
        num_non_padding = padding_masks.sum(dim=1).unsqueeze(1)
        num_non_padding = torch.clamp(num_non_padding, min=1e-9)
        output = sum_features / num_non_padding # Shape: (batch_size, d_model)
        
        # (!!!) Feed the (B, d_model) tensor into the MLP head (!!!)
        # This will now work for ANY batch size (including 1)
        output = self.output_layer(output) # Shape: (batch_size, num_classes)
        
        return output


# Dummy models (保持原样)
class DummyTSTransformerEncoder(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(DummyTSTransformerEncoder, self).__init__()
        self.feat_dim = feat_dim
        self.output_layer = nn.Linear(feat_dim, feat_dim)

    def forward(self, X, padding_masks):
        return self.output_layer(X)


class DummyTSTransformerEncoderClassiregressor(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(DummyTSTransformerEncoderClassiregressor, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = nn.Linear(feat_dim, num_classes)

    def forward(self, X, padding_masks):
        # 简单的全局平均池化
        output = X.mean(dim=1)
        return self.output_layer(output)


# ---------------------------------------------------------------------------------
# ------------------ 新增代码开始 (用于CNN+Transformer) ------------------
# ---------------------------------------------------------------------------------

class Downsampler(nn.Module):
    def __init__(self, in_channels, d_model, activation_fn, cnn_channels=[64, 128]):


        super(Downsampler, self).__init__()


        self.activation = activation_fn

        # Block 1:
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, cnn_channels[0], kernel_size=7, stride=1, padding=3),
            self.activation, # (无括号)
            nn.MaxPool1d(kernel_size=2, stride=2)
        )


        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=5, stride=1, padding=2),
            self.activation,  # <-- 删除括号
            nn.MaxPool1d(kernel_size=5, stride=5)
        )

        # 投影层: 将通道数 cnn_channels[1] 投影到 d_model
        self.projection = nn.Conv1d(cnn_channels[1], d_model, kernel_size=1, stride=1)

    def forward(self, x):
        # x_in shape: (Batch, C_in, L_in)
        x = self.block1(x)  # Shape: (Batch, cnn_channels[0], L_in / 2)
        x = self.block2(x)  # Shape: (Batch, cnn_channels[1], L_in / 10)
        x = self.projection(x)  # Shape: (Batch, d_model, L_in / 10)
        return x


class Upsampler(nn.Module):
    def __init__(self, d_model, out_channels, activation_fn, cnn_channels=[64, 128]):
        super(Upsampler, self).__init__()

        self.activation = activation_fn

        self.projection = nn.Conv1d(d_model, cnn_channels[1], kernel_size=1, stride=1)

        # Block 2:
        self.block2 = nn.Sequential(
            nn.ConvTranspose1d(cnn_channels[1], cnn_channels[0], kernel_size=5, stride=5, padding=2, output_padding=4),
            self.activation # (无括号)
        )
        self.block1 = nn.Sequential(
            nn.ConvTranspose1d(cnn_channels[0], out_channels, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def forward(self, x):
        # x_in shape: (B, d_model, L_latent)
        x = self.projection(x)  # (B, cnn_channels[1], L_latent)
        x = self.block2(x)  # (B, cnn_channels[0], L_latent * 5)
        x = self.block1(x)  # (B, feat_dim, L_latent * 10)
        return x


class CNNTransformerEncoder(nn.Module):
    """
    新的模型类，结合了CNN下采样器和Transformer编码器。
    用于 `task="imputation"` (无监督学习)。
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm',
                 cnn_channels=[64, 128],  # 接收 cnn_channels
                 freeze=False):
        super(CNNTransformerEncoder, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.feat_dim = feat_dim
        self.downsample_factor = 10  # 硬编码的下采样因子
        self.max_len = max_len # e.g., 10000

        # 关键: 确保 max_len 可以被10整除
        assert max_len % self.downsample_factor == 0, \
            "max_len (e.g., 10000) 必须能被下采样因子 (10) 整除"

        # 潜（Latent）序列长度, e.g., 10000 / 10 = 1000
        self.latent_len = max_len // self.downsample_factor

        self.act = _get_activation_fn(activation)

        # 1. CNN 编码器 (Downsampler) - 传递 cnn_channels
        self.project_inp = Downsampler(feat_dim, d_model, self.act, cnn_channels)

        # 2. 位置编码 (使用潜序列长度)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=self.latent_len)

        # 3. Transformer 编码器 (核心)
        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                             dropout * (1.0 - freeze), activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # 4. CNN 解码器 (Upsampler) - 传递 cnn_channels
        self.output_layer = Upsampler(d_model, feat_dim, self.act, cnn_channels)

        self.dropout1 = nn.Dropout(dropout)

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) - 原始长序列 (e.g., 10000)
            padding_masks: (batch_size, seq_length) - 原始长序列的掩码 (e.g., 10000)
        Returns:
            output: (batch_size, seq_length, feat_dim) - 重构的长序列 (e.g., 10000)
        """

        # 1. CNN 编码
        # 输入 X: (B, L_in, C_in) -> (B, C_in, L_in)
        inp = X.permute(0, 2, 1)
        # (B, C_in, L_in) -> (B, d_model, L_latent)
        inp = self.project_inp(inp)  # L_latent = L_in / 10

        # 2. 下采样 padding_masks
        # (B, L_in) -> (B, 1, L_in)
        padding_masks_latent = padding_masks.unsqueeze(1).float()
        # (B, 1, L_in) -> (B, 1, L_latent)
        padding_masks_latent = F.max_pool1d(padding_masks_latent,
                                            kernel_size=self.downsample_factor,
                                            stride=self.downsample_factor)
        # (B, 1, L_latent) -> (B, L_latent)
        padding_masks_latent = padding_masks_latent.bool().squeeze(1)

        # 3. Transformer 编码
        # (B, d_model, L_latent) -> (L_latent, B, d_model)
        inp = inp.permute(2, 0, 1)
        inp = self.pos_enc(inp)  # 添加位置编码 (长度为 L_latent)

        try:
            output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks_latent, is_causal=False)
        except TypeError:
            output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks_latent)

        output = self.act(output)

        # 4. CNN 解码
        # (L_latent, B, d_model) -> (B, d_model, L_latent)
        output = output.permute(1, 2, 0)
        output = self.dropout1(output)
        # (B, d_model, L_latent) -> (B, feat_dim, L_in)
        output = self.output_layer(output)

        # (B, feat_dim, L_in) -> (B, L_in, feat_dim)
        output = output.permute(0, 2, 1)

        return output


# ---------------------------------------------------------------------------------
# ------------------ 新增代码开始 (用于 CNN 回归/分类) -----------------
# ---------------------------------------------------------------------------------

class CNNTransformerEncoderClassiregressor(nn.Module):
    """
    新的模型类，结合了CNN下采样器和用于分类/回归的 Transformer。
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm',
                 cnn_channels=[64, 128],  # 接收 cnn_channels
                 freeze=False):
        super(CNNTransformerEncoderClassiregressor, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.feat_dim = feat_dim
        self.downsample_factor = 10  # 硬编码的下采样因子
        self.max_len = max_len  # e.g., 10000

        assert max_len % self.downsample_factor == 0, \
            "max_len (e.g., 10000) 必须能被下采样因子 (10) 整除"

        self.latent_len = max_len // self.downsample_factor  # e.g., 1000

        self.act = _get_activation_fn(activation)

        # 1. CNN 编码器 (Downsampler)
        self.project_inp = Downsampler(feat_dim, d_model, self.act, cnn_channels)

        # 2. 位置编码 (使用潜序列长度)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=self.latent_len)

        # 3. Transformer 编码器 (核心)
        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                             dropout * (1.0 - freeze), activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.dropout1 = nn.Dropout(dropout)

        self.num_classes = num_classes

        # 4. 输出层 (来自 TSTransformerEncoderClassiregressor)
        self.output_layer = self.build_output_module(d_model, self.latent_len, num_classes)

    def build_output_module(self, d_model, latent_len, num_classes):
        """
        这是从 TSTransformerEncoderClassiregressor 复制而来的
        它将 (B, latent_len, d_model) 展平并投影到 (B, num_classes)
        """
        output_layer = nn.Linear(d_model * latent_len, num_classes)
        return output_layer

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) - 原始长序列 (e.g., 10000)
            padding_masks: (batch_size, seq_length) - 原始长序列的掩码 (e.g., 10000)
        Returns:
            output: (batch_size, num_classes) - 回归值
        """

        # 1. CNN 编码
        inp = X.permute(0, 2, 1)  # (B, C, L_in)
        inp = self.project_inp(inp)  # (B, d_model, L_latent)

        # 2. 下采样 padding_masks
        padding_masks_latent = padding_masks.unsqueeze(1).float()
        padding_masks_latent = F.max_pool1d(padding_masks_latent,
                                            kernel_size=self.downsample_factor,
                                            stride=self.downsample_factor)
        padding_masks_latent = padding_masks_latent.bool().squeeze(1)  # (B, L_latent)

        # 3. Transformer 编码
        inp = inp.permute(2, 0, 1)  # (L_latent, B, d_model)
        inp = self.pos_enc(inp)

        try:
            output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks_latent, is_causal=False)
        except TypeError:
            output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks_latent)

        output = self.act(output)
        output = output.permute(1, 0, 2)  # (B, L_latent, d_model)
        output = self.dropout1(output)

        # 4. 输出层
        output = output * padding_masks_latent.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.output_layer(output)

        return output
# ---------------------------------------------------------------------------------
# ------------------ 新增代码结束 ------------------------------------
# ---------------------------------------------------------------------------------