from typing import Literal, Optional
import math
import torch
from torchmetrics import Metric

from dsp_board.processor import Processor
from fastdtw import fastdtw

from utils.tensor import to_numpy

class MCD(Metric):
    def __init__(
        self,
        dsp_processor: Processor,
        feature_type: Literal["mcep", "mgc", "mfcc"]="mgc",
    ):
        super().__init__()
        self.dsp_processor = dsp_processor
        self.feature_type = feature_type
        self.add_state("aggregation", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_frames", default=torch.tensor(0.), dist_reduce_fx="sum")

    def calc_mcep(self, x):
        if self.feature_type == "mcep":
            y = self.dsp_processor.mel_cepstrum(x)
        elif self.feature_type == "mgc":
            y = self.dsp_processor.mel_generalized_cepstrum(x)
        elif self.feature_type == "mfcc":
            y = self.dsp_processor.mfcc(x)
        else:
            raise ValueError(f"Unsupported mel cepstrum type: {self.mcep_type}. Choose from 'mcep' or 'mgc'.")

        return y
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):

        # # Create a mask to remove small values
        # # Extremely small values can cause numerical instability
        # nonzero_indices = torch.logical_and(
        #     ~torch.isclose(target, torch.zeros_like(target), atol=1e-6), 
        #     ~torch.isclose(preds, torch.zeros_like(preds), atol=1e-6)
        # )
        # target = target[nonzero_indices]
        # preds = preds[nonzero_indices]

        # calculate mcep ---------------------
        mgc = self.calc_mcep(target)
        mgc_preds = self.calc_mcep(preds)
        
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
        mgc = torch.FloatTensor(mgc)
        mgc_preds = torch.FloatTensor(mgc_preds)
        distortion = torch.sqrt(2*torch.sum((mgc - mgc_preds).pow(2), dim=-1))
        self.aggregation = self.aggregation + torch.sum(distortion)
        self.total_frames = self.total_frames + mgc.shape[0]

        return
    
    def compute(self) -> torch.Tensor:
        mcd = (10 / math.log(10)) * self.aggregation / self.total_frames
        return mcd