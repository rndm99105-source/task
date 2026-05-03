# Azərbaycan Dili üçün Nitq Tanıma Sistemi (ASR)

> **Model:** `openai/whisper-small`  
> **Dataset:** Mozilla Common Voice 17.0 — Azerbaijani (`az`)  
> **Tapşırıq:** AI Engineer Intern — Texniki Tapşırıq

---

## 📁 Repo Strukturu

```
az-stt-intern/
├── README.md
├── requirements.txt
├── part_a/
│   └── inference.py        ← Baseline inferens + WER/CER
├── part_b/
│   └── finetune.py         ← Fine-tuning + müqayisə + qrafiklər
├── results/
│   ├── part_a_results.csv
│   ├── part_a_summary.csv
│   ├── part_a_wer_distribution.png
│   ├── part_b_comparison.csv
│   └── part_b_training_curves.png
└── report.md               ← Analitik hesabat (Azərbaycan dilində)
```

---

## 🤖 İstifadə Olunan Model

| Parametr | Dəyər |
|---|---|
| Model | `openai/whisper-small` |
| Parametr sayı | ~244M |
| Dil dəstəyi | 99 dil (o cümlədən Azərbaycan) |
| Task | `transcribe` |
| Sampling rate | 16 kHz |

**Seçim əsası:** Whisper-small Azərbaycan dilini nativ dəstəkləyir, Google Colab Free GPU-da fine-tune edilə bilir və sürət/keyfiyyət balansı baxımından optimaldır.

---

## 📊 WER / CER Nəticələri

> ⚠️ Aşağıdakı cədvəl skriptlər işlədikdən sonra `results/` qovluğundan real dəyərlərlə yenilənəcəkdir.

### Hissə A — Baseline

| Model | Samples | WER (%) | CER (%) |
|---|---|---|---|
| Whisper-small (Baseline) | 200 | *(run script)* | *(run script)* |

### Hissə B — Fine-Tuning Müqayisəsi

| Model | WER (%) | CER (%) | ΔWER |
|---|---|---|---|
| Whisper-small (Baseline)   | — | — | — |
| Whisper-small (Fine-tuned) | — | — | — |

---

## ⚙️ Kodu İşə Salmaq (Google Colab)

### Quraşdırma

```bash
# Repo-nu klonla
!git clone https://github.com/<username>/az-stt-intern.git
%cd az-stt-intern

# Asılılıqları quraşdır
!pip install -r requirements.txt -q
```

### Hissə A — Baseline İnferens

```bash
%cd part_a
!python inference.py
```

Çıxış: `../results/part_a_results.csv`, `part_a_summary.csv`, `part_a_wer_distribution.png`

### Hissə B — Fine-Tuning

```bash
%cd ../part_b
!python finetune.py
```

Çıxış: `../results/part_b_comparison.csv`, `part_b_training_curves.png`, `./whisper-small-az/` (checkpoint)

> **Tövsiyə:** Colab-da `Runtime → Change runtime type → T4 GPU` seçin.  
> Part A: ~15-25 dəq | Part B: ~40-60 dəq

---

## 🔬 Fine-Tuning Parametrləri

| Parametr | Dəyər |
|---|---|
| Learning rate | 1e-5 |
| Train batch size | 8 |
| Gradient accumulation | 2 (effective batch = 16) |
| Max steps | 500 |
| Warmup steps | 50 |
| Eval/Save steps | hər 100 addımda |
| Early stopping patience | 3 eval |
| FP16 | GPU varsa aktivdir |
| Best model metric | Validation WER (↓) |

---

## 📈 Qrafiklər

Skriptlər tamamlandıqdan sonra `results/` qovluğunda:
- `part_a_wer_distribution.png` — WER histoqramı
- `part_b_training_curves.png` — Train/Val Loss + WER per step

---

## 📄 Analitik Hesabat

Tam analitik hesabat üçün → [`report.md`](./report.md)
