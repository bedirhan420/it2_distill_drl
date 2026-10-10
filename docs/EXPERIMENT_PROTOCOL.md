# Deney Protokolü (akademik gereksinimler)

## Araştırma soruları / hipotezler
- **H1 (Fidelity):** Temiz gözlemde IT2 öğrenci, eşit kural sayısında T1'e göre fidelity açısından kötü değildir (non-inferiority, marj δ=0.02).
- **H2 (Gürültü direnci):** σ >≈ 2–3 gürültü için IT2, T1'den ve karar ağacından daha dirençli. PPO ise referans.”
- **H3 (Pareto):** (kural sayısı, robustness AUC) uzayında IT2 fronts, T1 ve karar ağacını domine eder / hypervolume daha yüksektir.
- **H4 (Neden?):** Kazanç FOU'dan gelir, parametre fazlalığından değil (A2: FOU→0 ⇒ fark kaybolur; parametre-eşit T1 kontrolü).
- **H5 (Adil karşılaştırma):** Gürültü-farkında eğitim (A7) T1'in açığını kapatıyorsa bunu raporla — dürüst negatif sonuç da yayınlanabilir.

## Karşılaştırılan yöntemler
PPO (öğretmen) · **IT2-Distill** (2 MF varyantı) · T1-Fuzzy (aynı kod yolu) · Karar ağacı (max_leaf_nodes = kural sayısı, VIPER-benzeri) · Parametre-eşit MLP (kara kutu öğrenci, fidelity üst sınırı) · (opsiyonel) PPO + gürültü-augmente eğitimli öğretmen.

## Metrikler
Tanımlar `evaluation/metrics.py` docstring'inde. (1) Ortalama getiri (on-policy, gürültülü ortam) (2) Normalize getiri ve `relative_degradation(σ)` (3) `robustness_auc` (4) Fidelity: öğrencinin **gürültülü** gözleme eylemi ↔ öğretmenin **temiz** gözleme eylemi (5) Karmaşıklık: #kural, ort. öncül uzunluğu (don't-care budama sonrası), #parametre (6) Opsiyonel: kural kararlılığı (seed'ler arası).

## Gürültü modeli
`σ_rel × std_j(train obs)` Gauss gürültüsü; σ_rel ∈ {0, .05, .1, .2, .3, .5}. Ek stres testleri (sonraki aşama): sensör dropout, sabit bias, heavy-tailed (Student-t), eylem gecikmesi.

## Ablasyonlar (`configs/ablation/all.yaml`)
A1 MF tipi (T1/IT2-σ/IT2-μ) · A2 FOU genişliği ve öğrenilebilirliği · A3 kural sayısı (2–32) · A4 başlatma · A5 type-reduction (Nie-Tan / BM / KM-referans yaklaşım farkı) · A6 TSK derecesi, t-norm · A7 gürültü-farkında eğitim · A8 loss (CE vs KL soft-label) · A9 DAgger turları · A10 regularizerlar · A11 veri boyutu. Her ablasyon **tek faktör** değiştirir; etkileşimler için seçili 2-faktörlü grid.

## İstatistiksel prosedür
- ≥10 bağımsız seed (öğretmen sabit; öğrenci seed + ayrıca ≥3 öğretmen seed'i ile tekrar).
- **Eşleştirilmiş tasarım:** aynı (seed, σ) için tüm yöntemler aynı gürültü akışını görür.
- Ortalama + %95 bootstrap GA (10k); eşleştirilmiş Wilcoxon + **Holm** düzeltmesi; etki büyüklüğü Cliff's δ; çoklu yöntem için Friedman + post-hoc.
- Hiperparametre ayarı **yalnız val** setinde; test seti ve test σ değerleri tek kez kullanılır.
- Rapor: yapılandırma, git hash, seed'ler, donanım, çalışma süresi.

## Geçerlilik tehditleri
Kovaryans kayması (BC) → DAgger ablasyonu · Öğretmenin kendisinin gürültüye kırılganlığı (referans: "temiz-etiket" fidelity) · Tek ortam riski → CartPole + LunarLander (+ MountainCarContinuous/Pendulum ile sürekli eylem) · Kural sayısı ≠ okunabilirlik → ort. öncül uzunluğu + kullanıcı çalışması (opsiyonel) · IT2'nin ek parametresi → parametre-eşit kontrol · Gürültü ölçeği seçimi → göreli σ + duyarlılık analizi.

## Tekrarlanabilirlik kontrol listesi
Sabit seed'ler · `pyproject` sürümleri · config'ler repoda · öğretmen checkpoint + veri hash'i · tek komutla (`make`) tüm tablolar/şekiller.
