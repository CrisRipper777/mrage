from __future__ import annotations

import importlib


def build_model(cfg, data_info):
    model_name = str(cfg.model.name).lower()
    task_name = str(cfg.task.name).lower()
    if model_name == "mrage":
        if task_name == "nc":
            module_name = "mrage"
        elif task_name == "lp":
            module_name = "mrage_lp"
        else:
            raise ValueError(f"MRAGE does not support task {task_name!r}")
    else:
        module_name = model_name
    module = importlib.import_module(f"src.models.{module_name}")
    return module.Model(cfg, data_info)
