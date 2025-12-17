import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch_wavelets import DWT1D, IDWT1D


class RecurrentCycle(torch.nn.Module):
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(
            cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length,
                        device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]


class WaveletDenoise(nn.Module):

    def __init__(self, dec_lev, wavelet='sym8'):
        super().__init__()
        self.dwt = DWT1D(wave=wavelet, J=dec_lev, mode='symmetric')
        self.idwt = IDWT1D(wave=wavelet, mode='symmetric')
        self.dec_lev = dec_lev
        self.register_buffer('a', torch.tensor(0.85))

    def forward(self, x):

        B, T, N = x.shape
        x_reshaped = x.permute(0, 2, 1).contiguous().view(B * N, 1, T)
        coeffs_low, coeffs_high_list = self.dwt(x_reshaped)
        cd1 = coeffs_high_list[0]
        sigma = torch.median(torch.abs(cd1), dim=-1, keepdim=True)[0] / 0.6745
        lamda = sigma * math.sqrt(2.0 * math.log(T))
        coeffs_high_thresh = []

        for c in coeffs_high_list:
            thresh_val = self.a * lamda
            c_thresh = torch.sign(c) * F.relu(torch.abs(c) - thresh_val)
            coeffs_high_thresh.append(c_thresh)

        x_denoised_reshaped = self.idwt((coeffs_low, coeffs_high_thresh))
        x_denoised_reshaped = x_denoised_reshaped[..., :T]
        x_denoised = x_denoised_reshaped.view(B, N, T).permute(0, 2, 1)

        return x_denoised


class MovingAvg(nn.Module):

    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size,
                                stride=stride, padding=0)

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        pad_size = (self.kernel_size - 1) // 2
        x_padded = F.pad(x_permuted, (pad_size, pad_size), mode='replicate')
        trend = self.avg(x_padded).permute(0, 2, 1)
        
        return trend


class SeriesDecomp(nn.Module):

    def __init__(self, kernel_size, dec_lev):
        super().__init__()
        self.denoise_module = WaveletDenoise(dec_lev)
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend

        trend_dn = self.denoise_module(trend)
        seasonal_dn = self.denoise_module(seasonal)

        trend_noise = trend - trend_dn
        seasonal_noise = seasonal - seasonal_dn

        return trend_noise, trend_dn, seasonal_noise, seasonal_dn


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual
        self.use_revin = configs.use_revin
        self.dropout = nn.Dropout(configs.dropout)

        kernel_size = 25
        self.decomposition = SeriesDecomp(kernel_size, configs.dec_lev)

        self.cycle_len = configs.cycle
        self.cycleQueue = RecurrentCycle(
            cycle_len=self.cycle_len, channel_size=self.channels)

        self.clean_linears = nn.ModuleList()
        for _ in range(2):  # trend_clean, season_clean
            if self.individual:
                self.clean_linears.append(nn.ModuleList([
                    nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)
                ]))
            else:
                self.clean_linears.append(
                    nn.Linear(self.seq_len, self.pred_len))

        if not self.individual:
            self.clean_interaction = nn.Sequential(
                nn.Linear(2 * self.pred_len, self.pred_len),
                nn.ReLU(),
                nn.Linear(self.pred_len, self.pred_len)
            )
        else:
            self.clean_interaction = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * self.pred_len, self.pred_len),
                    nn.ReLU(),
                    nn.Linear(self.pred_len, self.pred_len)
                ) for _ in range(self.channels)
            ])

        self.noise_linears = nn.ModuleList()
        for _ in range(2):  # trend_noise, season_noise
            if self.individual:
                self.noise_linears.append(nn.ModuleList([
                    nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)
                ]))
            else:
                self.noise_linears.append(
                    nn.Linear(self.seq_len, self.pred_len))

        self.noise_weight = nn.Parameter(torch.tensor(0.1))

    def _project_clean(self, trend_part, season_part, trend_linear, season_linear):
        trend_part = trend_part.permute(0, 2, 1)
        season_part = season_part.permute(0, 2, 1)

        if self.individual:
            trend_weights = torch.stack(
                [layer.weight for layer in trend_linear], dim=0)
            trend_biases = torch.stack(
                [layer.bias for layer in trend_linear], dim=0)
            season_weights = torch.stack(
                [layer.weight for layer in season_linear], dim=0)
            season_biases = torch.stack(
                [layer.bias for layer in season_linear], dim=0)

            trend_proj = torch.einsum(
                'bct,cpt->bcp', trend_part, trend_weights) + trend_biases.unsqueeze(0)
            season_proj = torch.einsum(
                'bct,cpt->bcp', season_part, season_weights) + season_biases.unsqueeze(0)

            B, C, P = trend_proj.shape
            clean_outputs = []
            for c in range(C):
                combined = torch.cat(
                    [trend_proj[:, c, :], season_proj[:, c, :]], dim=1)
                interacted = self.clean_interaction[c](combined)
                clean_outputs.append(interacted.unsqueeze(1))

            output = torch.cat(clean_outputs, dim=1)  # -> [B, C, P]

        else:
            trend_proj = trend_linear(trend_part)    # -> [B, C, P]
            season_proj = season_linear(season_part)  # -> [B, C, P]

            B, C, P = trend_proj.shape
            trend_flat = trend_proj.view(B * C, P)
            season_flat = season_proj.view(B * C, P)

            combined = torch.cat([trend_flat, season_flat], dim=1)
            interacted = self.clean_interaction(combined)

            output = interacted.view(B, C, P)

        return output

    def _project_noise(self, part, linear_module):
        part = part.permute(0, 2, 1)

        if self.individual:
            weights = torch.stack(
                [layer.weight for layer in linear_module], dim=0)
            biases = torch.stack(
                [layer.bias for layer in linear_module], dim=0)
            output = torch.einsum('bct,cpt->bcp', part,
                                  weights) + biases.unsqueeze(0)
            return output
        else:
            return linear_module(part)

    def forward(self, x, cycle_index):
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x = self.dropout(x)

        x = x - self.cycleQueue(cycle_index, self.seq_len)

        trend_noise, trend_clean, season_noise, season_clean = self.decomposition(
            x)

        clean_output = self._project_clean(
            trend_clean, season_clean,
            self.clean_linears[0], self.clean_linears[1]
        )

        noise_trend_output = self._project_noise(
            trend_noise, self.noise_linears[0])
        noise_season_output = self._project_noise(
            season_noise, self.noise_linears[1])

        noise_contribution = self.noise_weight * \
            (noise_trend_output + noise_season_output)
        output = clean_output + noise_contribution

        output = output.permute(0, 2, 1)

        output = output + \
            self.cycleQueue((cycle_index + self.seq_len) %
                            self.cycle_len, self.pred_len)

        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean

        return output
