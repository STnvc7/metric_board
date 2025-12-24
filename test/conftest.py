import pytest
import os
from glob import glob
import torch
import torchaudio
from metric_board.evaluator import Evaluator
from metric_board import metrics
from metric_board.utils.tensor import fix_length

_script_path = os.path.abspath(__file__)
_script_dir = os.path.dirname(_script_path)
_root_dir = os.path.dirname(_script_dir)
SAMPLE_DIR = os.path.join(_root_dir, "samples")

@pytest.fixture(scope="function")
def evaluator():
    def get_evaluator(device):
        sample_rate = 48000
        fft_size = 1024
        hop_size = 256
        m = {
            "dnsmos": metrics.DNSMOS(sample_rate),
            # "mcd_mgc": metrics.MCD(sample_rate, fft_size, hop_size, mcep_type="mgc"),
            "mcd_mcep": metrics.MCD(sample_rate, fft_size, hop_size, mcep_type="mcep"),
            "mcd_mfcc": metrics.MCD(sample_rate, fft_size, hop_size, mcep_type="mfcc"),
            "spc": metrics.Spectrogram(fft_size, hop_size),
            "mel_spc": metrics.MelSpectrogram(sample_rate, fft_size, hop_size, n_mels=80),
            "pesq_wb": metrics.PESQ(sample_rate, mode="wb"),
            "pesq_nb": metrics.PESQ(sample_rate, mode="nb"),
            "stoi": metrics.STOI(sample_rate, extended=True),
            "pitch": metrics.Pitch(sample_rate, hop_size),
            "vuv": metrics.VUVErrorRate(sample_rate, hop_size),
            "vuv_f1": metrics.VUVF1(sample_rate, hop_size),
            "utmos": metrics.UTMOS(sample_rate),
        }
        device = torch.device(device)
        return Evaluator(m).to(device)
    return get_evaluator
    
def audio_files():
    preds = sorted(glob(os.path.join(SAMPLE_DIR, "preds", "*.wav")))
    target = sorted(glob(os.path.join(SAMPLE_DIR, "target", "*.wav")))
    return {"preds": preds, "target": target}
    
def audio_tensor(device):
    preds = sorted(glob(os.path.join(SAMPLE_DIR, "preds", "*.wav")))
    target = sorted(glob(os.path.join(SAMPLE_DIR, "target", "*.wav")))
    
    device = torch.device(device)
    preds = [torchaudio.load(pred)[0].to(device) for pred in preds]
    target = [torchaudio.load(target)[0].to(device) for target in target]
    
    preds = [torch.stack([preds[i], fix_length(preds[i+1], preds[i].shape[1])]) for i in range(0, len(preds), 2)]
    target = [torch.stack([target[i], fix_length(target[i+1], target[i].shape[1])]) for i in range(0, len(target), 2)]
    return {"preds": preds, "target": target}
    
@pytest.fixture(scope="session")
def audio():
    def get_audio(type, device):
        if type == "file":
            return audio_files()
        elif type == "tensor":
            return audio_tensor(device)
        else:
            raise ValueError(f"Invalid type: {type}")
    return get_audio