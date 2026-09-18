import json
from collections import UserDict
from pathlib import Path

from prettytable import PrettyTable


class EvaluationResult(UserDict):
    """Mapping of ranking cutoffs to metric values."""

    def __str__(self) -> str:
        if not self.data:
            return super().__str__()

        metric_names = sorted(next(iter(self.values())).keys())
        table = PrettyTable()

        table.field_names = ["K"] + [item.replace("_", " ").title() for item in metric_names]
        table.align["K"] = "r"

        for k, metrics in self.items():
            table.add_row([k] + [f"{metrics[name]:.4f}" for name in metric_names])

        return table.get_string()

    def save(self, path: str | Path) -> None:
        """Save as ``.txt`` or ``.json``, inferred from the file extension."""

        path = Path(path)
        if path.suffix == ".txt":
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(str(self))
        elif path.suffix == ".json":
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.data, handle, indent=2)
        else:
            raise ValueError(f"Format '{path.suffix}' not supported")
