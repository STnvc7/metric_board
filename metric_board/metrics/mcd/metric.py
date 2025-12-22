from typing import Literal, List
import math
import torch
from dsp_board.features import mel_generalized_cepstrum, mel_cepstrum, mfcc
from fastdtw import fastdtw  # type: ignore

from metric_board.utils.tensor import to_numpy, from_numpy
from metric_board.interface import MetricBase, MetricOutput


class MCD(MetricBase):
    distortions: List[torch.Tensor]
    
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
        
        self.add_state("distortions", default=[], dist_reduce_fx="cat")

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

        # Create a mask to remove small values
        # Extremely small values can cause numerical instability
        nonzero_indices = torch.logical_and(
            ~torch.isclose(target, torch.zeros_like(target), atol=1e-6), 
            ~torch.isclose(preds, torch.zeros_like(preds), atol=1e-6)
        )

        # calculate mcep ---------------------
        try:
            mgc = self.calc_mcep(target[nonzero_indices])
            mgc_preds = self.calc_mcep(preds[nonzero_indices])
        except Exception as e:
            print(f"Error calculating MGC: {e}")
            return

        # dtw --------------------------------
        mgc = mgc.permute(1,0)  # (frame, order)
        mgc_preds = mgc_preds.permute(1, 0) # (frame, order)
        mgc = to_numpy(mgc)
        mgc_preds = to_numpy(mgc_preds)

        _, path = fastdtw(mgc, mgc_preds)
        pathx = list(map(lambda l: l[0], path))
        pathy = list(map(lambda l: l[1], path))
        mgc, mgc_preds = mgc[pathx], mgc_preds[pathy]

        # aggregate ---------------------------
        mgc = from_numpy(mgc, self.device, torch.float32)
        mgc_preds = from_numpy(mgc_preds, self.device, torch.float32)
        distortion_per_frame = torch.sqrt(2 * torch.sum((mgc - mgc_preds).pow(2), dim=-1))
        self.distortions.append(distortion_per_frame)

        return
    
    def compute(self) -> MetricOutput:
        distortions = torch.cat(self.distortions, dim=-1).flatten()
        mcd = (10 / math.log(10)) * distortions
        return self.calc_output(mcd)