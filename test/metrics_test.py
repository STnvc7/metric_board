import pytest

@pytest.mark.parametrize("audio_type", ["file", "tensor"])
@pytest.mark.parametrize("evaluator_device", ["cpu", "cuda"])
@pytest.mark.parametrize("audio_device", ["cpu", "cuda"])
def test_metrics(evaluator, audio, audio_type, evaluator_device, audio_device):
    audio = audio(audio_type, audio_device)
    evaluator = evaluator(evaluator_device)
    result = evaluator.evaluate(audio["preds"], audio["target"])
    for key, value in result.items():
        print(f"{key}: {value.mean:.4f} +- {value.std:.4f}, {value.std_error:.4f}")