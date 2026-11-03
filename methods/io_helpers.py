"""Shared JSON and CSV/JSONL output helpers."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path


def json_value(value):
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist") and getattr(value,"ndim",0)>0:
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2, default=json_value) + "\n")
    temp.replace(path)

def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        temp = path.with_name(path.name + ".tmp." + str(os.getpid()))
        with temp.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, default=json_value) + "\n")
    else:
        temp = path.with_name(path.name + ".tmp." + str(os.getpid()))
        with temp.open("w", newline="") as f:
            fields = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, default=json_value) if isinstance(value,(dict,list,tuple)) else value
                                 for key,value in row.items()})
    temp.replace(path)
