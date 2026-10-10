"""Facade over the whole experiment: teacher -> data -> distill -> evaluate.
Scripts and ablation runner only talk to this class."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from it2distill.config.schema import ExperimentConfig


class ExperimentPipeline:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.out = Path(cfg.out_dir) / cfg.name
        self.out.mkdir(parents=True, exist_ok=True)

    def train_teacher(self) -> Path:
        """WP1: TeacherTrainer(...).train()"""
        from it2distill.teacher.trainer import TeacherTrainer
        
        print(f"[{self.cfg.name}] Eğitmen (Teacher) PPO politikası eğitiliyor...")
        trainer = TeacherTrainer(
            env_cfg=self.cfg.env, 
            cfg=self.cfg.teacher, 
            out_dir=self.out, 
            seed=self.cfg.seed
        )
        model_path = trainer.train()
        print(f"Eğitim tamamlandı. Model kaydedildi: {model_path}.zip")
        return model_path

    def collect_data(self) -> Path:
        """WP2: TeacherDataCollector(...).collect() -> split_by_episode -> save npz"""
        raise NotImplementedError

    def distill_student(self, seed: int):
        """WP3-5: build via FuzzyPolicyBuilder.from_config, train via DistillationTrainer."""
        raise NotImplementedError

    def evaluate(self, students: dict) -> pd.DataFrame:
        """WP7: NoiseRobustnessEvaluator(...).run(students)"""
        raise NotImplementedError

    def run_all_seeds(self) -> pd.DataFrame:
        raise NotImplementedError
