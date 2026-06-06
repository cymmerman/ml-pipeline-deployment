"""
simulator.py — NASA C-MAPSS Engine Simulator
Replays real turbofan sensor data as a live stream with configurable
fault injection. Powers the WebSocket endpoint and live dashboard.

Datasets:
    FD001 — 1 fault mode, 1 operating condition  (100 engines)
    FD002 — 1 fault mode, 6 operating conditions (260 engines)
    FD003 — 2 fault modes, 1 operating condition (100 engines)
    FD004 — 2 fault modes, 6 operating conditions(249 engines)
"""

import asyncio
import logging
import random
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(r"C:\Users\tyler\data")

# The 9 sensors used by the model (from metadata.json)
MODEL_SENSORS = ["s9", "s14", "s4", "s3", "s17", "s7", "s12", "s11", "s2"]

# Full C-MAPSS column names
CMAPSS_COLUMNS = (
    ["engine_id", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)

# Fault modes mapped to which sensors they primarily affect
FAULT_PROFILES = {
    "none": {},
    "bearing_wear": {
        "s4":  {"drift_rate": 0.002, "noise_scale": 1.5},
        "s11": {"drift_rate": 0.003, "noise_scale": 1.8},
        "s14": {"drift_rate": 0.001, "noise_scale": 1.2},
    },
    "compressor_stall": {
        "s2":  {"drift_rate": 0.004, "noise_scale": 2.0},
        "s3":  {"drift_rate": 0.005, "noise_scale": 2.5},
        "s17": {"drift_rate": 0.003, "noise_scale": 1.6},
    },
    "fuel_system_degradation": {
        "s7":  {"drift_rate": 0.003, "noise_scale": 1.4},
        "s9":  {"drift_rate": 0.002, "noise_scale": 1.3},
        "s12": {"drift_rate": 0.004, "noise_scale": 1.7},
    },
    "turbine_erosion": {
        "s11": {"drift_rate": 0.005, "noise_scale": 2.2},
        "s12": {"drift_rate": 0.004, "noise_scale": 2.0},
        "s14": {"drift_rate": 0.003, "noise_scale": 1.5},
        "s9":  {"drift_rate": 0.002, "noise_scale": 1.3},
    },
}


# ── Data loader ───────────────────────────────────────────────────────────────
class CMAPSSLoader:
    _cache: dict = {}

    @classmethod
    def load(cls, dataset: str = "FD001") -> pd.DataFrame:
        if dataset in cls._cache:
            return cls._cache[dataset]

        path = DATA_DIR / f"train_{dataset}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        df = pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)
        cls._cache[dataset] = df
        logger.info(f"Loaded {dataset}: {len(df)} rows, {df['engine_id'].nunique()} engines")
        return df

    @classmethod
    def get_engine_ids(cls, dataset: str = "FD001") -> list:
        df = cls.load(dataset)
        return sorted(df["engine_id"].unique().tolist())

    @classmethod
    def get_engine_data(cls, dataset: str, engine_id: int) -> pd.DataFrame:
        df = cls.load(dataset)
        engine_df = df[df["engine_id"] == engine_id].copy()
        if engine_df.empty:
            raise ValueError(f"Engine {engine_id} not found in {dataset}")
        return engine_df.sort_values("cycle").reset_index(drop=True)

    @classmethod
    def get_random_engine(cls, dataset: str = "FD001") -> int:
        ids = cls.get_engine_ids(dataset)
        return random.choice(ids)


# ── Fault injector ────────────────────────────────────────────────────────────
class FaultInjector:
    """
    Applies progressive degradation to sensor readings.
    Models realistic fault propagation — starts subtle, worsens over cycles.
    """

    def __init__(self, fault_mode: str, start_cycle: int, total_cycles: int):
        self.fault_mode   = fault_mode
        self.start_cycle  = start_cycle
        self.total_cycles = total_cycles
        self.profile      = FAULT_PROFILES.get(fault_mode, {})
        self.rng          = np.random.default_rng(int(time.time()))

    def apply(self, sensors: dict, current_cycle: int) -> dict:
        if self.fault_mode == "none" or not self.profile:
            return sensors

        # Progressive severity: 0 at fault start, 1 at end of engine life
        cycles_since_fault = max(0, current_cycle - self.start_cycle)
        severity = min(1.0, cycles_since_fault / max(1, self.total_cycles - self.start_cycle))

        modified = sensors.copy()
        for sensor, params in self.profile.items():
            if sensor not in modified:
                continue
            drift  = params["drift_rate"] * cycles_since_fault * severity
            noise  = self.rng.normal(0, params["noise_scale"] * 0.01 * (1 + severity))
            modified[sensor] = float(modified[sensor]) + drift + noise

        return modified


# ── Simulator engine ──────────────────────────────────────────────────────────
class EngineSimulator:
    """
    Streams sensor readings from a single engine cycle by cycle.
    Optionally injects faults at a configurable cycle.
    """

    def __init__(
        self,
        dataset: str = "FD001",
        engine_id: Optional[int] = None,
        fault_mode: str = "none",
        fault_start_pct: float = 0.6,   # inject fault at 60% of engine life
        speed_multiplier: float = 1.0,  # 1.0 = real-time (1 cycle/sec), 10.0 = 10x faster
    ):
        self.dataset          = dataset
        self.fault_mode       = fault_mode
        self.speed_multiplier = speed_multiplier

        # Load engine data
        self.engine_data = CMAPSSLoader.get_engine_data(
            dataset, engine_id or CMAPSSLoader.get_random_engine(dataset)
        )
        self.engine_id   = int(self.engine_data["engine_id"].iloc[0])
        self.total_cycles = len(self.engine_data)

        # Fault injection setup
        fault_cycle = int(self.total_cycles * fault_start_pct)
        self.injector = FaultInjector(fault_mode, fault_cycle, self.total_cycles)

        # State
        self.current_cycle = 0
        self.is_running    = False
        self.session_id    = None

        logger.info(
            f"Simulator ready: {dataset} engine {self.engine_id}, "
            f"{self.total_cycles} cycles, fault={fault_mode}"
        )

    def get_info(self) -> dict:
        return {
            "dataset":       self.dataset,
            "engine_id":     self.engine_id,
            "total_cycles":  self.total_cycles,
            "current_cycle": self.current_cycle,
            "fault_mode":    self.fault_mode,
            "is_running":    self.is_running,
            "progress_pct":  round(self.current_cycle / self.total_cycles * 100, 1),
        }

    async def stream(self) -> AsyncGenerator[dict, None]:
        """
        Async generator — yields one reading per cycle.
        Each reading is a dict ready to POST to /predict.
        """
        self.is_running = True
        delay = 1.0 / self.speed_multiplier

        for idx, row in self.engine_data.iterrows():
            if not self.is_running:
                break

            self.current_cycle += 1
            cycle = int(row["cycle"])

            # Extract model sensor values
            raw_sensors = {s: float(row[s]) for s in MODEL_SENSORS if s in row}

            # Apply fault injection
            sensors = self.injector.apply(raw_sensors, cycle)

            # Build feature vector in model's expected order
            features = [sensors.get(s, 0.0) for s in MODEL_SENSORS]

            reading = {
                "engine_id":     self.engine_id,
                "dataset":       self.dataset,
                "cycle":         cycle,
                "cycle_pct":     round(cycle / self.total_cycles * 100, 1),
                "fault_mode":    self.fault_mode,
                "fault_active":  cycle >= self.injector.start_cycle and self.fault_mode != "none",
                "features":      features,
                "sensor_values": sensors,
                "timestamp":     time.time(),
            }

            yield reading
            await asyncio.sleep(delay)

        self.is_running = False
        logger.info(f"Simulator finished: engine {self.engine_id}, {self.current_cycle} cycles")

    def stop(self):
        self.is_running = False


# ── Active simulator registry ─────────────────────────────────────────────────
# Tracks running simulators by session key so WebSocket can reference them
_active_simulators: dict = {}


def create_simulator(
    dataset: str = "FD001",
    engine_id: Optional[int] = None,
    fault_mode: str = "none",
    fault_start_pct: float = 0.6,
    speed_multiplier: float = 5.0,
) -> tuple[str, EngineSimulator]:
    """Create and register a new simulator. Returns (session_key, simulator)."""
    sim = EngineSimulator(
        dataset=dataset,
        engine_id=engine_id,
        fault_mode=fault_mode,
        fault_start_pct=fault_start_pct,
        speed_multiplier=speed_multiplier,
    )
    key = f"{dataset}_{sim.engine_id}_{int(time.time())}"
    _active_simulators[key] = sim
    return key, sim


def get_simulator(key: str) -> Optional[EngineSimulator]:
    return _active_simulators.get(key)


def stop_simulator(key: str):
    if key in _active_simulators:
        _active_simulators[key].stop()
        del _active_simulators[key]


def list_simulators() -> list:
    return [
        {"key": k, **v.get_info()}
        for k, v in _active_simulators.items()
    ]


# ── Dataset metadata ──────────────────────────────────────────────────────────
def get_dataset_info() -> dict:
    info = {}
    for ds in ["FD001", "FD002", "FD003", "FD004"]:
        try:
            df = CMAPSSLoader.load(ds)
            info[ds] = {
                "engines":     int(df["engine_id"].nunique()),
                "total_rows":  len(df),
                "avg_cycles":  round(df.groupby("engine_id")["cycle"].max().mean(), 1),
                "max_cycles":  int(df.groupby("engine_id")["cycle"].max().max()),
                "fault_modes": 2 if ds in ["FD003", "FD004"] else 1,
                "op_conditions": 6 if ds in ["FD002", "FD004"] else 1,
            }
        except FileNotFoundError:
            info[ds] = {"error": "Dataset file not found"}
    return info


def get_available_fault_modes() -> list:
    return [
        {
            "mode":        mode,
            "description": _fault_descriptions[mode],
            "affected_sensors": list(profile.keys()),
        }
        for mode, profile in FAULT_PROFILES.items()
    ]


_fault_descriptions = {
    "none":                    "No fault — baseline healthy engine operation",
    "bearing_wear":            "Progressive bearing degradation causing vibration and heat",
    "compressor_stall":        "Airflow disruption in compressor stages",
    "fuel_system_degradation": "Fuel delivery inefficiency reducing thrust output",
    "turbine_erosion":         "High-pressure turbine blade material loss",
}