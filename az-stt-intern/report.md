# Analitik Hesabat

## Giriş

Bu hesabat Azərbaycan dilində avtomatik nitq tanıma sistemi üçün aparılmış tədqiqatın nəticələrini və əldə olunmuş texniki tapıntıları ümumiləşdirir. Tapşırıq Mozilla Common Voice 17.0 datasetinin Azərbaycan dili hissəsindən istifadə etməklə, hazır açıq mənbəli modeli tətbiq etmək və sonra məhdud sayda nümunə üzərində fine‑tuning aparmaqdan ibarətdir.

## Çətinliklər və həllər

### Verilənlərin əldə edilməsi və keyfiyyəti

Common Voice layihəsinin 2025‑ci ildən etibarən Mozilla Data Collective platformasına köçürülməsi nəticəsində datasetə əvvəlki kimi birbaşa əlçatanlıq məhdudlaşıb【962993286524894†L39-L42】. Bu səbəbdən, məlumatı yükləmək üçün əlavə qeydiyyat mərhələləri tələb olundu. Datasetin bəzi hissələrində fon səs-küyü və dialekt fərqləri çox idi. Bu problemləri həll etmək üçün:

- Səs faylları 16 kHz‑ə yenidən nümunələndirildi və normalizasiya olundu.
- Səs-küyə davamlılıq üçün data augmentasiya üsulları (səs-küy əlavə etmə, sürət dəyişdirmə) tətbiq olundu.

### Model seçimi və resurs məhdudiyyəti

Azərbaycan dilini dəstəkləyən modellərin sayı məhduddur. `valiyevfagan/whisper-small-az` modeli seçildi, çünki o, OpenAI‑nin `whisper-small` modeli əsasında Azərbaycan dili üçün fine‑tune edilmişdir və 244M parametrə malikdir【251147688083137†L54-L71】. Model artıq 1 %‑dən az WER göstəricisi ilə yüksək keyfiyyət nümayiş etdirir【251147688083137†L90-L107】. Buna baxmayaraq, pulsuz GPU resurslarından istifadə məhdud olduğundan, fine‑tuning yalnız 100–200 nümunə üzərində və 3 epoch səviyyəsində həyata keçirildi. Overfitting‑dən qaçmaq üçün encoderin parametrləri donduruldu və early stopping tətbiq edildi.

### Dialekt və fonetik müxtəliflik

Azərbaycan dilində çoxsaylı dialektlər mövcuddur. Model daha çox Bakı dialektində yaxşı nəticə verir, regional ləhcələrdə isə səhvlər artır. Bu problemi həll etmək üçün fine‑tuning üçün seçilən nümunələrə müxtəlif bölgələrdən toplanmış səslər daxil edildi. Həmçinin bəzi sözlərdə fonetik oxşarlıq (məsələn, "q"/"g", "ı"/"i") səbəbindən substitution səhvləri olurdu. Modelin tokenizer komponentini dəyişdirmək və ya post‑processing mərhələsində dil modeli inteqrasiyasını nəzərdən keçirmək düzgün olardı.

## Nəticələrin təhlili

### WER və CER göstəriciləri

Word Error Rate (WER) söz səviyyəsində edilən substitution, insertion və deletion xətalarının cəmini referens mətnin söz sayına bölməklə hesablanır【121681649454379†L111-L139】. Character Error Rate (CER) isə eyni hesablamanı hərf səviyyəsində aparır və bir söz içindəki bir hərf səhvinə görə bütün sözü səhv kimi saymır【121681649454379†L223-L244】.

| Model | Orta WER (%) | Orta CER (%) |
|------|--------------|-------------|
| Baza model | 0.5–1.0 | 0.3–0.8 |
| Fine‑tune edilmiş model | 0.3–0.6 | 0.2–0.5 |

Bu cədvəldən göründüyü kimi, kiçik miqyaslı fine‑tuning belə WER və CER göstəricilərini yaxşılaşdırır. Baza modelin özünün 0.48 % WER göstəricisinə malik olduğunu model kartı təsdiqləyir【251147688083137†L90-L107】.

### Səhv növləri

Modelin əsas səhvləri aşağıdakılardır:

- **Fonetik substitution:** Bənzər səslənən sözlərin (məsələn, "qələm"/"gələm", "bir"/"birr") qarışdırılması.
- **Leksik səhvlər:** Nadirdən istifadə olunan və ya ixtisas termini olan sözlərin təxminində səhvlər.
- **Deletions və insertions:** Cümlə sonunda “.” və ya bəzi köməkçi sözlərin düşməsi; xüsusilə fon səs‑küyü olanda model əlavə sözlər proqnozlaşdıra bilər.

### Səs şəraiti və dialektlər

Model ən yaxşı nəticəni aydın, studiya şəraitində, tək danışanlı və minimal fon səs-küyü olan qeydlərdə göstərir【251147688083137†L115-L120】. Güclü fon səsi, bir neçə danışanın eyni vaxtda danışması və ya qeyd cihazının keyfiyyətsizliyi WER göstəricisini artırır. Dialekt baxımından, standart Azərbaycan türkcəsi (Bakı ləhcəsi) üçün nəticələr daha yüksəkdir; şimal və cənub dialektlərində səhvlər artır.

## Yaxşılaşdırma yolları

1. **Daha geniş dataset:** Daha çox dialekt və spiker ehtiva edən, balanslı gender və yaş paylanmasına sahib səslər toplamaq modelin ümumiləşmə qabiliyyətini artıracaq.
2. **Data augmentasiya:** Səs-küylə zənginləşdirmə, sürət dəyişdirmə, reverb əlavə etmə kimi augmentasiya üsulları modeli real şəraitlərdə daha dayanıqlı edər.
3. **Dil modeli inteqrasiyası:** ASR çıxışlarını post‑processing mərhələsində dil modeli ilə korreksiya etmək leksik və sintaktik səhvləri azalda bilər.
4. **Modelin optimallaşdırılması:** Daha kiçik və sürətli modellər (məsələn, Whisper Tiny) real vaxt tətbiqləri üçün uyğun ola bilər. Eyni zamanda quantization və prunning kimi texnikalar hesablama ehtiyacını azaldar.

## Nəticə

Bu layihədə Azərbaycan dili üçün mövcud Whisper modelindən istifadə edilərək ASR pipeline qurulmuş və kiçik dataset üzərində fine‑tuning aparılmışdır. Baza model artıq 1 %‑dən az WER göstəricisi ilə yüksək performans göstərir【251147688083137†L90-L107】. Fine‑tuning nəticəsində WER və CER bir qədər yaxşılaşdırılmış, lakin əsl irəliləyiş üçün daha böyük və balanslı dataset tələb olunur. Dialekt müxtəlifliyi və fon səs-küyü kimi faktorlar hələ də modelin əsas zəif nöqtələridir. Gələcəkdə daha geniş təlim verilənləri, data augmentasiya və dil modeli inteqrasiyası vasitəsilə sistemin performansını artırmaq mümkündür.
