from typing import Literal, List
import math
import torch
from dsp_board.features import mel_generalized_cepstrum, mel_cepstrum, mfcc
from fastdtw import fastdtw  # type: ignore

from metric_board.utils.tensor import to_numpy, from_numpy, channelize
from metric_board.interface import MetricBase, MeanMetric, MetricOutput


class MCD(MetricBase):
    def __init__(
        self,
        sample_rate: int,
        fft_size: int,
        hop_size: int,
        mcep_type: Literal["mcep", "mgc", "mfcc"]="mgc",
        order: int=24,
        stage: int=5,
        n_mels: int=80,
    ):
        super().__init__()
        self.mcep_type = mcep_type
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.order = order
        self.stage = stage
        self.n_mels = 80
        self.metric = MeanMetric()

    def calc_mcep(self, x):
        if self.mcep_type == "mcep":
            y = mel_cepstrum(
                x,
                sample_rate=self.sample_rate,
                fft_size=self.fft_size,
                hop_size=self.hop_size,
                order=self.order,
            )
        elif self.mcep_type == "mgc":
            y = mel_generalized_cepstrum(
                x,
                sample_rate=self.sample_rate,
                fft_size=self.fft_size,
                hop_size=self.hop_size,
                order=self.order,
                stage=self.stage,
            )
        elif self.mcep_type == "mfcc":
            y = mfcc(
                x,
                sample_rate=self.sample_rate,
                fft_size=self.fft_size,
                hop_size=self.hop_size,
                n_mels=self.n_mels,
                n_mfcc=self.order
            )
        else:
            raise ValueError(f"Unsupported mel cepstrum type: {self.mcep_type}. Choose from 'mcep' or 'mgc'.")

        return y
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):

        preds = channelize(preds, keep_dims=1)
        target = channelize(target, keep_dims=1)
        preds = preds + torch.randn_like(preds) * 1e-6
        target = target + torch.randn_like(target) * 1e-6
        
        # calculate mcep ---------------------
        try:
            mgc = self.calc_mcep(target)
            mgc_preds = self.calc_mcep(preds)
        except Exception as e:
            print(f"Error calculating MGC: {e}")
            return

        # dtw --------------------------------
        for i in range(mgc.shape[0]):
            _mgc = mgc[i].permute(1,0)  # (frame, order)
            _mgc_preds = mgc_preds[i].permute(1, 0) # (frame, order)
            _mgc = to_numpy(_mgc)
            _mgc_preds = to_numpy(_mgc_preds)
    
            _, path = fastdtw(_mgc, _mgc_preds)
            pathx = list(map(lambda l: l[0], path))
            pathy = list(map(lambda l: l[1], path))
            _mgc, _mgc_preds = _mgc[pathx], _mgc_preds[pathy]

            # aggregate ---------------------------
            _mgc = from_numpy(_mgc, self.device, torch.float32)
            _mgc_preds = from_numpy(_mgc_preds, self.device, torch.float32)
            distortion = torch.sqrt(2 * torch.sum((_mgc - _mgc_preds)**2, dim=-1))
            distortion = (10 / math.log(10)) * distortion
            self.metric.update(distortion.flatten())

        return
    
    def compute(self) -> MetricOutput:
        return self.metric.compute()