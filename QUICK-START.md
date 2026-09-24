# Match Result Prediction v3.4 — Başlangıç

Yeni **Injuries & Lineups** sekmesi maç bazında sakatlık/ceza nedenlerini, ilk 11'i, veri zamanını ve eksik API yanıtlarını gösterir. Canlı bağlantı için `API_FOOTBALL_KEY` gerekir. `python -m match_predictor doctor` yerel hazırlık durumunu kontrol eder.

Önemli: Sakatlık ve kadro verisi henüz öğrenilmiş tahmin olasılıklarını değiştirmiyor. Varsayılan bağlam kontrolü eski/eksik veride seçici sinyali durdurur; yeni filtrenin başarı oranı ölçülmedi. Dört yeni lig için lisanslı geçmiş veri ve eğitim hâlâ gerekir. Ayrıntılı kurulum ve kalan işler: [LIVE_USE.md](LIVE_USE.md).

Bu paket kodu, güncel maç geçmişini, eğitilmiş modelleri ve ölçüm raporlarını içerir. Python 3.11 veya 3.12 kullan.

## Uygulamayı aç

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

Tarayıcıdaki uygulamada ligi ve takımları seç. En güçlü kullanım için aynı anda, aynı kaynaktan alınmış üç decimal oranı gir: ev sahibi, beraberlik ve deplasman. Örneğin `2.10 / 3.50 / 3.40`. Sistem oran marjını temizler.

Arayüz artık İngilizce dört bölümden oluşur: **Match Forecast**, **Team Dashboard**, **Player Dashboard** ve **Model Evaluation**. Team Dashboard son maçları, formu ve puan tablosunu; Player Dashboard ise toplanan kadro, oyuncu istatistiği ve sakatlık snapshot'larını gösterir. Sol menüdeki Top-10 league coverage tablosu, hangi liglerin modelinin hazır olduğunu ayrıca belirtir.

## Aktif kullanım kuralı

1. Önce veriyi güncelle ve modeli yeniden eğit.
2. Güncel 1-X-2 oranlarını gir.
3. Her maçta gösterilen **“Tüm maçlar modu”** çifte şans sonucunu incele. Bu çıktı en düşük olasılıklı sonucu eler ve 1X, X2 veya 12 verir.
4. Exact 1-X-2 bölümünde **“Seçim ölçütlerini karşılıyor”** görünürse model bu maçı güçlü exact tahmin grubuna almıştır.
5. Exact bölümünde **“PAS”** görünürse çifte şans yine üretilir; ancak tek sonuç için güçlü iddia yoktur.

Güncelleme:

```bash
python -m match_predictor download --start 2026 --end 2026
python -m match_predictor train
python -m match_predictor report
```

`2026`, 2026/27 sezon başlangıç yılıdır. Yeni sezonda yılı değiştir. Kaynak henüz en son hafta sonuçlarını yayımlamadıysa modelin veri tarihi ekranda görünür; güçlü filtre 14 günden eski veride otomatik PAS verir.

## Kadro, sakatlık ve oran snapshot'ı toplama

API-Football anahtarını kaynak koda yazmadan ortam değişkenine koy:

```bash
export API_FOOTBALL_KEY="..."
python -m match_predictor collect --fixture 123456
```

Bir ligdeki yaklaşan maçları bulup bütün detayları çekmek için:

```bash
python -m match_predictor collect --league premier_league --season 2026 \
  --from-date 2026-09-25 --to-date 2026-09-27 --details
```

Tamamlanmış maçlarda oyuncu dakika, rating, gol ve asist istatistiklerini de al:

```bash
python -m match_predictor collect --fixture 123456 --include-player-stats
```

Ham API yanıtları ve normalize edilmiş veriler `data/api_football/` altında zaman damgasıyla saklanır. Bu kayıtlar şu anda veri toplama katmanıdır; model bunları hemen tahmine katmaz. Önce aynı zaman pencerelerinden yeterli geçmiş birikmeli, ardından sadece ileri tarihli testte doğrulanan özellikler modele alınmalıdır. API-Football'dan gelen maç içi istatistikler pre-match tahmine eklenmez.

## Her maçta %70 üzeri ne demek?

2025/26 bağımsız test sezonundaki 2.058 maçın tamamında çifte şans doğruluğu sadece istatistikle **%77,3**, güncel oran desteğiyle **%79,3** oldu. Her maça aynı `12` seçimini veren en güçlü sabit yöntem %74,4'te kaldı; modele göre değişen seçim 4,8 puan ekledi. Oran destekli altı lig sonucu %78,4–80,5 aralığındadır. Çifte şans üç olası sonuçtan ikisini kapsadığı için bu rakam exact 1-X-2 doğruluğu değildir.

## Exact %70 sonucu tam olarak ne demek?

Eşik yalnız eski doğrulama döneminde seçildi. Sonraki 2025/26 sezonundaki 2.058 maçta:

| Mod | Seçilen maç | Kapsama | Doğruluk |
|---|---:|---:|---:|
| Sadece istatistik | 318 | %15,5 | **%72,6** |
| Güncel oran destekli | 404 | %19,6 | **%73,3** |

Bunun ardından 2026/27 sezonunun mevcut ilk 256 maçında oran destekli filtre 63 maç seçti ve **%74,6** doğru çıktı. Bu ikinci örnek daha küçüktür.

Bu rakam tek bir maçın %73 kesinlikle doğru olduğu anlamına gelmez. Oran destekli sistem yaklaşık her beş maçtan birini seçiyor. Sonuç altı lig toplamıdır; bazı liglerin tek başına küçük örnek sonucu %70'in altında kalmıştır. 2025/26 oran destekli sonuç için yaklaşık %95 belirsizlik aralığı %68,7–77,3'tür. Uzun dönem başarısının kesin olarak %70 üstünde olduğu henüz kanıtlanmış değildir.

## Uygulamada neler var?

- Ev sahibi / beraberlik / deplasman olasılıkları.
- Her maç için 1X, X2 veya 12 çifte şans çıktısı.
- İstatistik modeli, oranlardan çıkarılan piyasa olasılığı ve birleşik sonuç karşılaştırması.
- Güçlü tahmin veya PAS kararı.
- En olası beş skor, beklenen gol, 2,5 üst ve iki takım gol atar çıktıları.
- Form ve Elo özeti.
- Eski doğrulama, bağımsız sezon ve güncel sezon denetimi.

Skor, 2,5 üst ve iki takım gol atar çıktıları ayrıca %70 hedefiyle doğrulanmadı. Kadro, sakatlık, transfer ve xG akışı henüz doğrudan dahil değildir. Sistem için kârlılık veya bahis piyasasına üstünlük gösterilmemiştir.

Ayrıntılar `reports/benchmark.md` dosyasındadır. Football-Data ücretsiz verisinin özel kullanım ve ticari/otomatik kullanım şartlarını güncel kaynaktan kontrol et; ticari kullanım için lisanslı veri kullan.

GitHub'a otomatik gönderim yapılmadı. Eski koddaki API anahtarı kaldırıldı; önceki GitHub geçmişinde kalabileceği için sağlayıcı üzerinden iptal edilmelidir.
