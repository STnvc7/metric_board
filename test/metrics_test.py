from typing import Dict, List, Literal
from metric_board.evaluator import Evaluator

def test_metrics(evaluator: Evaluator, audio_files: Dict[Literal["preds", "target"], List[str]]):
    result = evaluator.evaluate(audio_files["preds"], audio_files["target"])
    for key, value in result.items():
        print(f"{key}: {value.mean}")