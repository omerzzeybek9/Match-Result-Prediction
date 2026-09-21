# Match Result Prediction v2 — Başlangıç

Bu pakette yeni kod, indirilen maç geçmişi, eğitilmiş modeller ve bağımsız test raporları var. ZIP'i açın ve terminalde proje klasörüne geçin. Python 3.11 veya 3.12 kullanın.

## Çalıştırma

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Sonra:

```bash
python -m pip install -r requirements-tested.txt
python -m streamlit run streamlit/user_interface.py
```

Terminalde gösterilen yerel adresi tarayıcıda açın. Lig ve iki takım seçip “Tahmin et” düğmesine basın. Paket eğitilmiş modelleri içerdiği için ilk açılışta tekrar eğitim gerekmez. Daha farklı kütüphane sürümleriyle çalışacaksanız modelleri yeniden eğitin.

## Neler gösteriliyor?

- Ev sahibi / beraberlik / deplasman olasılıkları.
- En olası beş skor ve ayrı ayrı olasılıkları.
- Modelin beklediği gol sayıları, toplam gol ve iki takımın gol atma olasılıkları.
- Takım formu ve Elo güç özeti.
- Eğitimden ayrı tutulan dönemde ölçülen başarı ve basit modellerle karşılaştırma.

En olası skor, kesinleşmiş sonuç değildir. Galibiyet olasılığı %60 olan bir takımın yaklaşık %40 kazanamama olasılığı vardır. Modelin beklediği gol sayısı, şutlardan hesaplanan xG ölçümü değildir.

## Güncelleme ve eğitim

```bash
python -m match_predictor download --start 2026 --end 2026
python -m match_predictor train
python -m match_predictor report
```

Sezon kodu başlangıç yılıdır: 2026, 2026/27 sezonunu ifade eder. Farklı sezonda komutu buna göre değiştirin. Daha eski bütün veriyi almak için `--start 2018` kullanın. Yeni modeller üretildikten sonra uygulamayı yeniden açın.

Altı ulusal lig varsayılan olarak eğitilir. Şampiyonlar Ligi, eski ve çok az verili deneysel modeldir; güncel tahmin için yeterli değildir. Bu modelin son test bölümü sadece 16 maçtır. İsteğe bağlı eğitim: `python -m match_predictor train --leagues cl`.

## Bu sürümün sonucu

Altı ulusal ligde toplam **2.058 bağımsız test maçında %51,2 maç sonucu doğruluğu** ölçüldü. Geçmiş gol ortalamalarını kullanan referans model %48,8 aldı. Olasılık kalitesini ölçen log loss 1,031'den 0,996'ya düştü (daha düşük daha iyi). Eski notebook'un geleceğe ait bilgi içeren sonuçlarıyla bu rakamlar doğrudan karşılaştırılamaz.

Bunlar bir iyileşme, ancak “çok yüksek başarı” veya gelecekte aynı performans garantisi değildir. Lig bazındaki farkların belirsizlik aralıkları `reports/benchmark.md` dosyasında bulunur. Kadro, sakatlık, transfer ve xG verisi henüz dahil değil.

Test dönemi model seçimine katılmadı. Test raporu, test dönemi öncesinde eğitilmiş modelin sonuçlarıdır. Uygulamadaki model sonrasında mevcut bütün maçlarla yeniden eğitildi. Eski bir maçı bu güncel modele sorarak geçmişe dönük başarı ölçülmesi engellendi.

## Dosyalar

- `reports/benchmark.md`: ölçülen sonuçlar ve sınırlar.
- `artifacts/*_report.json`: ayrıntılı metrikler, kullanılan dönemler ve model karışımları.
- `artifacts/*_test_predictions.csv`: her test maçı için olasılıklar ve gerçek sonuç.
- `match_predictor/`: yeni tahmin altyapısı.
- `tests/`: veri sızıntısı ve tutarlılık testleri.
- `README.md`: yöntem, veri kaynağı ve teknik ayrıntılar.

GitHub'a gönderim yapılmadı. Eski kodda bulunan API anahtarı bu paketten kaldırıldı; sağlayıcı üzerinden eski anahtarı iptal edin, önceki GitHub kayıtlarında kalmış olabilir. Yeni veri indirme akışı API anahtarı istemiyor.
