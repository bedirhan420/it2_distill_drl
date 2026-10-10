# Proje Planı (14 hafta)

| Hafta | İş Paketi | Çıktı | Kabul kriteri |
|---|---|---|---|
| 1 | **WP0** Kurulum | Repo, CI (ruff+pytest), config sistemi | `pytest` yeşil |
| 1-2 | **WP1** Öğretmen | `TeacherTrainer`; CartPole ≥495, LunarLander ≥250 | 20 bölüm ort. ödül eşik üstü, checkpoint kayıtlı |
| 3 | **WP2** Veri | `TeacherDataCollector`, `split_by_episode`, DAgger | Episode sızıntısı yok (test), 50k+ geçiş |
| 3-5 | **WP3** Bulanık katmanlar | MF(T1/IT2), t-norm, TSK, reducer, builder | Birim testler: şekil, `lower≤upper`, FOU→0 ⇒ T1, gradyan akışı |
| 5-6 | **WP4** Damıtma | Trainer, loss'lar, regularizer'lar, initializer'lar | CartPole'da T1 fidelity ≥0.95 (temiz) |
| 6 | **WP5** Type-reduction | KM referansı, yaklaşım farkı ölçümü | Nie-Tan vs KM farkı raporlandı |
| 7 | **WP6** Açıklanabilirlik | `RuleExtractor`, `LinguisticLabeler`, rapor | Kural listesi okunur; don't-care budama |
| 7-8 | **WP7** Gürültü değerlendirme | Wrapper, `NoiseRobustnessEvaluator`, baseline'lar | Tüm yöntemler × σ × seed tablosu |
| 9-11 | **WP8** Ablasyonlar | A1–A11 koşuları, Pareto, istatistik | Tüm tablolar/şekiller `make report` ile |
| 11-12 | **WP9** İkinci ortam | LunarLander (+ sürekli eylem ortamı) | Aynı pipeline, yalnız config değişir |
| 13 | **WP10** Analiz | Hipotez testleri, negatif sonuç analizi, kural yorumlama | H1–H5 sonuçlandı |
| 14 | **WP11** Yazım | Makale/rapor, repro paketi | Tekrarlanabilirlik listesi ✓ |

## Risk ve önlemler
| Risk | Önlem |
|---|---|
| IT2 gürültüde T1'den iyi çıkmaz | A7 + A2 ile nedenini analiz et; negatif sonuç + FOU analizi katkıdır |
| LunarLander'da az kuralla düşük fidelity | Kural sayısı taraması (A3), DAgger, TSK-1, Pareto'da dürüst raporla |
| Kovaryans kayması | DAgger (A9), ε-pertürbe toplama |
| Hesap maliyeti (ablasyon × seed × σ) | Rollout önbelleği, paralel seed'ler, CartPole ile ön tarama |
| Okunabilirlik ölçülememesi | Kural sayısı + öncül uzunluğu + dilsel etiket tutarlılığı |

## Öncelik sırası (MVP → tam)
1. MVP: CartPole + T1 vs IT2 + gürültü eğrisi.  2. + DT/MLP baseline + A1–A3.  3. + DAgger, A4–A11.  4. + LunarLander, istatistik, yazım.
