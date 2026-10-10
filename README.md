# IT2-Distill
**Derin Pekiştirmeli Öğrenme Politikalarının Gürültüye Dirençli Aralık Tip-2 Bulanık Mantık Sistemlerine Damıtılması**

PPO (öğretmen) → durum-eylem verisi → türevlenebilir **Tip-1 / IT2 TSK bulanık ağ** (öğrenci) → gürültü altında
fidelity / getiri / kural sayısı (Pareto) analizi.

## Hızlı başlangıç
```bash
pip install -e ".[dev]"
pytest -q
make teacher data distill noise ENV=cartpole
make ablate ENV=cartpole
```

## Dizin yapısı
```
configs/            base.yaml, env/*.yaml, ablation/all.yaml (A1-A11)
docs/               PROJECT_PLAN.md, ARCHITECTURE.md, EXPERIMENT_PROTOCOL.md
scripts/            01..06 ince CLI sarmalayıcıları
src/it2distill/
  core/             Policy/Explainable arayüzleri, Registry, seeding, tipler
  config/           dataclass şema + YAML yükleyici (_base_ ve dotted override)
  envs/             Gym factory + GaussianObservationNoise wrapper
  teacher/          SB3 PPO eğitimi + SB3 -> Policy adapter
  data/             collector (BC/DAgger), dataset, episode bazlı split, augment
  fuzzy/            membership (T1/IT2), tnorm, type_reduction, layers, model, builder, initializers
  distill/          trainer, losses, regularizers, callbacks
  explain/          kural çıkarımı, dilsel etiketleme, raporlama
  evaluation/       rollout, metrics, noise_eval, pareto, statistics, plots
  baselines/        Karar ağacı (VIPER-benzeri), parametre-eşit MLP
  experiments/      pipeline (Facade), ablation (Command), runner
tests/              birim testleri
```

## Durum
Çekirdek bulanık mantık katmanları (MF, t-norm, type-reduction, TSK, model, builder, initializer),
loss/regularizer/trainer, metrikler, istatistik ve Pareto **uygulanmıştır**. `NotImplementedError` ile
işaretli yerler (`TODO(WPx)`) iskelet fonksiyonlardır; `docs/PROJECT_PLAN.md` hangi iş paketinde
dolacaklarını gösterir.

> Not: torch gerektiren testler (`test_membership.py`, `test_model.py`) iskelet hazırlanırken çalıştırma
> ortamında torch yüklenemediği için koşturulamadı; ilk iş `pytest -q` ile doğrulayın.
