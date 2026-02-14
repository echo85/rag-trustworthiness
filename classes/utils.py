import json
import os
from datetime import datetime
import numpy as _np

class Utils:
    def to_serializable(self,obj):
        try:
            if isinstance(obj, (_np.integer, _np.floating)):
                return obj.item()
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
        except Exception:
            pass
        try:
            if isinstance(obj, bytes):
                return obj.decode()
        except Exception:
            pass
        return str(obj)

    def save(self, results: dict, generation: bool = False, data_path: str = "") -> str:

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if(generation):
            out_path = os.path.join(f"data/data_{ts}.json")
        else:
            out_path = os.path.join(f"results/results_{ts}_w_{data_path}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, default=self.to_serializable, indent=2)

        return out_path

    def load(self, file_path: str) -> dict:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data