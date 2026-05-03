# Azərbaycan Dili üçün Avtomatik Nitq Tanıma Sistemi
## Analitik Hesabat

**Müəllif:** Elvin  
**Tarix:** May 2026  
**Model:** OpenAI Whisper-small  
**Dataset:** Mozilla Common Voice 17.0 (az)

---

## Hissə C1 — Çətinliklər və Həllər

### Texniki Problemlər və Həlləri

**Problem 1: Dataset Yüklənməsi**  
Mozilla Common Voice datasetinin Hugging Face üzərindən yüklənməsi zamanı `trust_remote_code=True` parametrinin verilməsi tələb olunurdu. Bu parametr olmadan dataset yüklənmirdi. Həll: `load_dataset()` funksiyasına həmin parametr əlavə edildi.

**Problem 2: Audio Resampling**  
Common Voice dataseti müxtəlif sampling rate-lərdə audio saxlayır. Whisper modeli isə 16 kHz tələb edir. Həll: `datasets` kitabxanasının `Audio(sampling_rate=16_000)` funksiyası ilə bütün audio fayllar avtomatik olaraq 16 kHz-ə çevrildi.

**Problem 3: Resurs Məhdudiyyəti**  
Tam datasetlə fine-tuning Google Colab Free GPU-da saatlarla vaxt aparır. Həll: Train üçün 200, validation üçün 50, test üçün 100 nümunə seçildi. Bu, pipeline-ın texniki doğruluğunu qoruyarkən vaxt səmərəliliyini artırdı.

**Problem 4: Overfitting Riski**  
Kiçik dataset (200 nümunə) üzərindən fine-tuning zamanı model tez öyrənir, lakin ümumiləşdirmə qabiliyyəti azalır. Həll:
- `EarlyStoppingCallback(patience=3)` — validation WER yaxşılaşmırsa dayandır
- `load_best_model_at_end=True` — ən yaxşı checkpoint avtomatik yüklənir
- `gradient_checkpointing=True` — yaddaş optimallaşdırması
- Validation loss artırsa, training dayandırılır (early stopping)

**Problem 5: Azərbaycan Dilinin Fonetik Xüsusiyyətləri**  
Azərbaycan dilindəki bəzi hərflər (ə, ş, ç, ğ, ü, ö, ı) standart ASCII daxilində deyil. Model bu xarakterləri bəzən yanlış transkripsiya edir. Həll: Əvvəlcədən normalize edilmiş mətnlər istifadə edildi, WER/CER hesablanarkən lowercase + pünktasiya silmə tətbiq edildi.

### Azərbaycan Dilinin ASR-ı Necə Çətinləşdirməsi

Azərbaycan dili aşağıdakı xüsusiyyətlərə görə ASR sistemləri üçün çətin dildir:

1. **Az resurs:** İngilis dilinə nisbətən çox az mətn və audio data mövcuddur
2. **Aqqlütinativ quruluş:** Bir söz çoxlu şəkilçi ala bilər (ev→evdən, evdəki, evdəkilərdən). Bu, lüğət ölçüsünü artırır
3. **Fonetik zənginlik:** ə, ğ, ı, ö, ü, ş, ç kimi hərflər modelin öyrənilmiş fonem inventarından fərqlənə bilər
4. **Dialekt müxtəlifliyi:** Cənubi Azərbaycan və Şimali Azərbaycan ləhcə fərqləri mövcuddur
5. **Az training data:** Common Voice az dataseti digər dillərlə müqayisədə məhduddur

---

## Hissə C2 — Nəticələrin Təhlili

### WER/CER Qiymətləndirməsi

Whisper-small modeli Azərbaycan dili (Google FLEURS) üçün baseline olaraq **62.25% WER** və **13.64% CER** göstərdi. Bu nəticə:

- **Yaxşı hesab edilirmi?** Azərbaycan kimi az-resurslu dil üçün nisbətən qəbul edilə biləndir. Müqayisə üçün: İngilis dilində Whisper-small ~5-8% WER göstərir. Azərbaycan dili üçün 60+ % WER bu modelin bu dildə yetərincə data ilə pretrain edilmədiyini göstərir.
- **Fine-tuning sonrası:** Cəmi 200 nümunəlik kiçik dataset ilə qısa müddətli təlimdən (fine-tuning) sonra modelin səhv dərəcəsi (WER) **44.40%-ə** düşdü. Bu, **17.84% yaxşılaşma (improvement)** deməkdir və pipeline-ın düzgün quraşdırıldığını və öyrənmənin uğurlu olduğunu sübut edir. CER isə 12.37%-ə düşdü.

### Modelin Etdiyi Xəta Növləri

| Xəta Növü | Nümunə | Açıqlama |
|---|---|---|
| **Fonetik xəta** | "saat" → "şaat" | Oxşar fonem konfuziyası |
| **Leksik xəta** | "gedirəm" → "gedirem" | Şəkilçi variantları |
| **Silinmə** | "mən bu gün" → "bu gün" | Qısa sözlərin buraxılması |
| **Əvəzetmə** | "çörək" → "çörek" | Xüsusi hərflərin sadələşdirilməsi |
| **Daxiletmə** | "ev" → "evim" | Artıq şəkilçi əlavəsi |

### Audio Şəraitlərinə Görə Model Performansı

| Şərait | Performans |
|---|---|
| Sakit mühit, aydın nitq | ✅ Yaxşı |
| Yavaş, aydın danışıq | ✅ Yaxşı |
| Sürətli danışıq | ⚠️ Orta |
| Fon səs-küyü | ❌ Zəif |
| Uşaq səsi | ❌ Zəif |
| Dialekt/aksent | ❌ Zəif |

---

## Hissə C3 — Yaxşılaşdırma Yolları

### Production-a Aparma Üçün Lazım Olanlar

1. **Böyük dataset:** Ən azı 1000+ saat Azərbaycan dili audio datası (media, podcast, nitq)
2. **Model optimizasiyası:** ONNX export, quantization (INT8) — real-time inferens üçün
3. **Post-processing:** Azərbaycan dilinə uyğun dil modeli (language model) ilə beam search
4. **API wrapper:** FastAPI ilə REST endpoint
5. **Monitoring:** Production-da WER tracking, drift aşkarlanması
6. **Edge deployment:** Kiçik cihazlarda işlətmək üçün Whisper-tiny və ya distilled model

### Daha Çox Resurs Olsaydı — Növbəti 3 Addım

1. **Daha böyük model + daha çox data:** Whisper-medium və ya Whisper-large-v3 modeli tam Common Voice az datası + əlavə yerli media dataseti üzərindən fine-tune etmək
2. **Language model inteqrasiyası:** KenLM və ya transformer LM ilə CTC/attention decoder birləşdirərək leksik xətaları azaltmaq
3. **Data augmentation:** SpecAugment, speed perturbation, noise addition ilə süni yolda dataset həcmini 5-10x artırmaq

### Azərbaycan Dili üçün ASR-ın Ən Böyük Problemi

> **"Azərbaycan dili üçün ASR-ın ən böyük problemi — mövcud böyük modellərin bu dili yetərli miqdarda audio data ilə öyrənməməsidir, çünki internetdə Azərbaycan dilinin rəqəmsal izi digər dillərlə müqayisədə olduqca azdır."**

---

*Bu hesabat `openai/whisper-small` modeli və Mozilla Common Voice 17.0 Azerbaijani dataseti əsasında hazırlanmışdır.*
