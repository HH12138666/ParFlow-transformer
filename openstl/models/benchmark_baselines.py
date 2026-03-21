import torch
import torch.nn as nn


class _BaseForecastModel(nn.Module):
    """Shared config parsing for lightweight benchmark backbones."""

    def __init__(self, **kwargs):
        super().__init__()
        model_config = kwargs.get('model_config', {})
        self.pre_seq = int(kwargs.get('pre_seq_length', model_config.get('pre_seq', 12)))
        self.aft_seq = int(kwargs.get('aft_seq_length', model_config.get('after_seq', 12)))
        self.input_channels = int(model_config.get('input_channels', kwargs.get('input_channels', 1)))
        self.in_channels = self.input_channels
        self.out_channels = int(model_config.get('out_channels', kwargs.get('loss_channels', 1)))


class CNNForecastModel(_BaseForecastModel):
    """Simple 2D CNN baseline over stacked temporal channels."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_config = kwargs.get('model_config', {})
        hidden_dim = int(model_config.get('hidden_dim', 64))
        kernel_size = int(model_config.get('kernel_size', 3))
        padding = kernel_size // 2
        in_ch = self.pre_seq * self.input_channels
        out_ch = self.aft_seq * self.out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_ch, kernel_size=1),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        if t != self.pre_seq:
            raise ValueError(f"CNN baseline expects T={self.pre_seq}, got {t}.")
        if c != self.input_channels:
            raise ValueError(f"CNN baseline expects C={self.input_channels}, got {c}.")
        x = x.reshape(b, t * c, h, w)
        y = self.net(x)
        y = y.reshape(b, self.aft_seq, self.out_channels, h, w)
        return y


class PixelRNNForecastModel(_BaseForecastModel):
    """Per-pixel RNN baseline (shared weights across all grid cells)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_config = kwargs.get('model_config', {})
        hidden_dim = int(model_config.get('hidden_dim', 64))
        num_layers = int(model_config.get('num_layers', 1))
        dropout = float(model_config.get('dropout', 0.0))
        if num_layers <= 1:
            dropout = 0.0
        self.encoder = nn.RNN(
            input_size=self.input_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.RNN(
            input_size=self.input_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.readout = nn.Linear(hidden_dim, self.out_channels)

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        if c != self.input_channels:
            raise ValueError(f"RNN baseline expects C={self.input_channels}, got {c}.")
        x_seq = x.permute(0, 3, 4, 1, 2).contiguous().reshape(b * h * w, t, c)
        _, h_n = self.encoder(x_seq)
        dec_in = x_seq.new_zeros(b * h * w, self.aft_seq, c)
        dec_out, _ = self.decoder(dec_in, h_n)
        y = self.readout(dec_out)
        y = y.reshape(b, h, w, self.aft_seq, self.out_channels).permute(0, 3, 4, 1, 2).contiguous()
        return y


class PixelLSTMForecastModel(_BaseForecastModel):
    """Per-pixel LSTM baseline (shared weights across all grid cells)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_config = kwargs.get('model_config', {})
        hidden_dim = int(model_config.get('hidden_dim', 64))
        num_layers = int(model_config.get('num_layers', 1))
        dropout = float(model_config.get('dropout', 0.0))
        if num_layers <= 1:
            dropout = 0.0
        self.encoder = nn.LSTM(
            input_size=self.input_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=self.input_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.readout = nn.Linear(hidden_dim, self.out_channels)

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        if c != self.input_channels:
            raise ValueError(f"LSTM baseline expects C={self.input_channels}, got {c}.")
        x_seq = x.permute(0, 3, 4, 1, 2).contiguous().reshape(b * h * w, t, c)
        _, (h_n, c_n) = self.encoder(x_seq)
        dec_in = x_seq.new_zeros(b * h * w, self.aft_seq, c)
        dec_out, _ = self.decoder(dec_in, (h_n, c_n))
        y = self.readout(dec_out)
        y = y.reshape(b, h, w, self.aft_seq, self.out_channels).permute(0, 3, 4, 1, 2).contiguous()
        return y


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x, h, c):
        gates = self.gates(torch.cat([x, h], dim=1))
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMForecastModel(_BaseForecastModel):
    """Single-layer ConvLSTM encoder-decoder baseline."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_config = kwargs.get('model_config', {})
        hidden_dim = int(model_config.get('hidden_dim', 32))
        kernel_size = int(model_config.get('kernel_size', 3))
        self.hidden_dim = hidden_dim
        self.encoder_cell = ConvLSTMCell(self.input_channels, hidden_dim, kernel_size=kernel_size)
        self.decoder_cell = ConvLSTMCell(self.input_channels, hidden_dim, kernel_size=kernel_size)
        self.head = nn.Conv2d(hidden_dim, self.out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        if c != self.input_channels:
            raise ValueError(f"ConvLSTM baseline expects C={self.input_channels}, got {c}.")
        h_t = x.new_zeros(b, self.hidden_dim, h, w)
        c_t = x.new_zeros(b, self.hidden_dim, h, w)

        for step in range(t):
            h_t, c_t = self.encoder_cell(x[:, step], h_t, c_t)

        dec_in = x[:, -1]
        aux = None
        if self.input_channels > self.out_channels:
            aux = x[:, -1, self.out_channels:self.input_channels]

        outputs = []
        for _ in range(self.aft_seq):
            h_t, c_t = self.decoder_cell(dec_in, h_t, c_t)
            pred = self.head(h_t)
            outputs.append(pred.unsqueeze(1))
            if aux is None:
                dec_in = pred
            else:
                dec_in = torch.cat([pred, aux], dim=1)

        return torch.cat(outputs, dim=1)
