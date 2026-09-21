# Match Result Prediction v3.1 — Başlangıç

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
