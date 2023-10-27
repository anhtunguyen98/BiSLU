import json
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

@dataclass
class TrainerState:


    epoch: int = 0 
    global_step: int = 0
    max_steps: int = 0
    num_train_epochs: int = 0
    loss: float = 0

    def to_string(self):
        json_string = json.dumps(dataclasses.asdict(self), sort_keys=True)+'\n'
        return json_string

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside :obj:`json_path`."""
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Create an instance from the content of :obj:`json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))