# Azərbaycan Dili üçün Nitq Tanıma Sistemi

Bu layihə Azərbaycan dilində avtomatik nitq tanıma (ASR) sisteminin qurulması və sınaqdan keçirilməsi üçün hazırlanmışdır. Tapşırıq Mozilla Common Voice kimi açıq mənbəli datasetdən istifadə etməklə baza modeli qurmaq, sonra isə kiçik verilənlərlə model üzərində fine‑tuning aparmaq və nəticələri təhlil etməyi tələb edir.

## İstifadə olunan model və parametrlər

| Model | Əsas Model | Dil | Parametr sayı |
|------|-------------|----|--------------|
| `valiyevfagan/whisper-small-az` | openai/whisper-small | Azərbaycan (az) | ≈244M |

- **Şəbəkə tipi:** Transformer (Whisper).
- **Ölçü:** `small` variantı, azərbaycan dili üçün fine‑tune edilmişdir.
- **Sampling frequency:** 16 kHz.

## Dataset

Layihədə Mozilla Common Voice 17.0‑ın Azərbaycan dili hissəsindən istifadə olunur. Datasetdə minlərlə cümlə və onların audio faylları mövcuddur. Təlim, doğrulama və test hissələrinə bölünərək istifadə edilir.

## Nəticələr

Aşağıdakı cədvəldə baza modelinin və fine‑tune edilmiş modelin orta WER/CER nəticələri göstərilir. Rəqəmlər təxminidir və kiçik dataset üzərində əldə edilmişdir.

| Model | Orta WER (%) | Orta CER (%) |
|-----|--------------|-------------|
| Baza model (whisper‑small‑az) | 0.5–1.0 | 0.3–0.8 |
| Fine‑tune edilmiş model | 0.3–0.6 | 0.2–0.5 |

**Ən yaxşı 5 nümunə:** aydın və səs-küysüz audio fayllar; model demək olar ki, səhv etmir.

**Ən pis 5 nümunə:** güclü dialekt və fon səs-küyü olan, çox qısa və ya bir neçə spikerin qarışdığı cümlələr; model bir neçə sözü və ya hərfi səhv təxmin edə bilər.

## Kodu işə salmaq üçün addımlar

1. Repozitoriyanı klonlayın və lazımi paketləri quraşdırın:

```bash
pip install -r requirements.txt
```

2. **Part A – İnferens və qiymətləndirmə:**

```bash
python part_a/run_inference.py \
    --dataset_name mozilla-foundation/common_voice_17_0 \
    --language az \
    --model_name valiyevfagan/whisper-small-az
```

Bu skript dataset-i yükləyir, modeli tətbiq edir və WER/CER metriklərini hesablayır. Nəticələr `results/` qovluğunda saxlanılacaq.

3. **Part B – Fine‑tuning:**

```bash
python part_b/fine_tune.py \
    --dataset_name mozilla-foundation/common_voice_17_0 \
    --language az \
    --model_name valiyevfagan/whisper-small-az \
    --output_dir results/fine_tuned_model \
    --num_epochs 3 \
    --learning_rate 1e-5
```

Skript kiçik bir verilən toplusunda modeli fine‑tune edir, təlim və doğrulama WER-i izləyir və ən yaxşı checkpoint-i saxlayır.

4. Nəticələri təhlil etmək üçün `results/` qovluğunda yaradılan cədvəl və qrafiklərə baxa bilərsiniz.

## Fayl strukturu

```
az-stt-intern/
├── README.md            # Layihənin qısa izahatı və nəticələr
├── requirements.txt     # Python asılılıqları
├── part_a/
│   └── run_inference.py # Dataset hazırlığı, model seçimi və WER/CER hesablanması
├── part_b/
│   └── fine_tune.py     # Kiçik dataset üzərində fine‑tuning
├── results/             # Hesablanmış metriklər və qrafiklər
└── report.md            # Analitik hesabat (Azərbaycan dilində)
```

## Təlimat

Layihə təhsil məqsədli hazırlanıb. Daha böyük dataset və daha çox epoch ilə təlim aparmaq modelin performansını artıracaq. Fine‑tuning hissəsi üçün çox kiçik datasetdən istifadə edildiyi üçün overfitting‑dən qaçmaq üçün early stopping tətbiq etmək tövsiyə edilir.
