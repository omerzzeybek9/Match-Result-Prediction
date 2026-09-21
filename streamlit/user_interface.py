"""Run: python -m streamlit run streamlit/user_interface.py"""
from datetime import date, timedelta
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import joblib
import pandas as pd
import streamlit as st
from match_predictor.data import LEAGUES
from match_predictor.predict import predict_match

st.set_page_config(page_title="Match Forecast", page_icon="⚽", layout="wide")
st.title("⚽ Match Forecast v3.1")
st.caption("Her maç için çifte şans, ayrıca exact 1-X-2 olasılıkları ve isteğe bağlı piyasa konsensüsü")

@st.cache_resource
def load_bundle(path, modified):
    return joblib.load(path)

def pct(value):
    return "—" if value is None else f"%{100*value:.1f}"

available = [name for name in LEAGUES if (ROOT/"artifacts"/f"{name}.joblib").exists()]
if not available:
    st.info("Henüz eğitilmiş model yok. Proje klasöründe aşağıdaki komutları çalıştırın.")
    st.code("python -m match_predictor download\npython -m match_predictor train", language="bash")
    st.stop()
league = st.sidebar.selectbox("Lig", available, format_func=lambda x: LEAGUES[x][0])
path = ROOT/"artifacts"/f"{league}.joblib"
bundle = load_bundle(str(path), path.stat().st_mtime_ns)
if bundle.get("schema_version") != 3:
    st.error("Bu model eski sürüme ait. `python -m match_predictor train` ile yeniden eğitin.")
    st.stop()
st.sidebar.caption(f"Son maç verisi: {bundle['last_match_date']}")
st.sidebar.caption(f"Geçmiş maç sayısı: {bundle['report']['data']['matches']:,}")
show_all = st.sidebar.checkbox("Önceki sezonların takımlarını da göster")
teams = bundle["teams"] if show_all else bundle["recent_teams"]
latest = date.fromisoformat(bundle["last_match_date"])
age = (date.today() - latest).days
if age > 14:
    st.warning(f"Veri {age} gündür güncellenmemiş. Güçlü tahmin filtresi veri yenilenene kadar pas verir.")
if league == "cl":
    st.warning("Şampiyonlar Ligi modeli sınırlı eski veriyle eğitilmiştir; güçlü tahmin filtresi kapalıdır.")
forecast_tab, evaluation_tab = st.tabs(["Maç tahmini", "Ölçülen başarı"])
with forecast_tab:
    left, right, when = st.columns([2,2,1])
    home = left.selectbox("Ev sahibi", teams)
    away = right.selectbox("Deplasman", [team for team in teams if team != home])
    match_date = when.date_input("Maç tarihi", value=max(date.today(),latest+timedelta(days=1)), min_value=latest+timedelta(days=1))
    use_odds=st.checkbox("Güncel 1-X-2 decimal oranlarını ekle (önerilir)")
    odds=None
    if use_odds:
        st.caption("Aynı kaynaktan ve aynı anda alınmış ev sahibi / beraberlik / deplasman oranlarını gir.")
        o1,ox,o2=st.columns(3)
        home_odds=o1.number_input(f"{home} oranı",min_value=1.01,max_value=100.0,value=2.20,step=.01)
        draw_odds=ox.number_input("Beraberlik oranı",min_value=1.01,max_value=100.0,value=3.40,step=.01)
        away_odds=o2.number_input(f"{away} oranı",min_value=1.01,max_value=100.0,value=3.20,step=.01)
        odds=[home_odds,draw_odds,away_odds]
    if st.button("Tahmin et", type="primary", width="stretch"):
        try:
            result = predict_match(bundle,home,away,match_date,odds)
        except ValueError as error:
            st.error(str(error))
        else:
            for warning in result["warnings"]:
                st.warning(warning)
            double_chance=result["double_chance"]
            st.success(f"Tüm maçlar modu: **{double_chance['label']} ({double_chance['code']})** · model olasılığı {pct(double_chance['probability'])}")
            mode_report=(bundle.get("selective_report") or {}).get("modes",{}).get(result["prediction_mode"],{})
            full_coverage=mode_report.get("all_match_double_chance")
            if full_coverage:
                league_result=mode_report.get("all_match_double_chance_by_league",{}).get(league)
                evidence=f"Bağımsız 2025/26 testinde tüm liglerde {full_coverage['total_matches']:,} maçın {pct(full_coverage['accuracy'])}'inde tuttu; kapsama %100. Sabit en iyi çifte şans: {pct(full_coverage['best_constant_accuracy'])}."
                if league_result:
                    evidence += f" {LEAGUES[league][0]}: {pct(league_result['accuracy'])} ({league_result['correct']}/{league_result['total_matches']})."
                st.caption(evidence+" Çifte şans iki sonucu kapsar; exact 1-X-2 ile aynı hedef değildir.")
            verdict=result["selection"]
            if verdict["selected"]:
                st.success(f"Exact 1-X-2 güçlü filtresi: **{result['predicted_outcome']}** · seçim ölçütlerini karşılıyor")
            else:
                st.info("Exact 1-X-2 güçlü filtresi: **PAS**")
                for reason in verdict["reasons"]:
                    st.caption(f"• {reason}")
            st.caption(verdict["note"])
            st.subheader(f"{home} — {away}")
            p = result["probabilities"]
            columns = st.columns(3)
            for column,label,value in zip(columns,[f"{home} kazanır","Beraberlik",f"{away} kazanır"],p.values()):
                column.metric(label,f"%{100*value:.1f}")
            st.bar_chart(pd.DataFrame({"Olasılık (%)":[100*p["home"],100*p["draw"],100*p["away"]]}, index=[home,"Beraberlik",away]),color="#16a085",horizontal=True)
            if result["market_probabilities"]:
                with st.expander("İstatistik modeli ve piyasa konsensüsü"):
                    labels=[home,"Beraberlik",away]
                    statistical=result["statistical_probabilities"];market=result["market_probabilities"]
                    st.dataframe(pd.DataFrame({"Sonuç":labels,
                        "İstatistik modeli (%)":[100*v for v in statistical.values()],
                        "Oranlardan çıkarılan (%)":[100*v for v in market.values()],
                        "Birleşik tahmin (%)":[100*v for v in p.values()]}).round(1),hide_index=True,width="stretch")
                    st.caption("Oranlardan marj temizlenir. Birleşim ağırlığı yalnız eski doğrulama döneminde seçilmiştir.")
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**En olası skorlar**")
                st.dataframe(pd.DataFrame([{"Skor":f"{s['home']}–{s['away']}","Olasılık":f"%{100*s['probability']:.1f}"} for s in result["top_scores"]]),hide_index=True,width="stretch")
                st.caption("En olası skorun gerçekleşme olasılığı yine de düşük olabilir.")
            with c2:
                expected=result["expected_goals"]
                st.metric("Modelin beklediği gol",f"{expected['home']:.2f} — {expected['away']:.2f}")
                st.caption("Bu değerler maç içi şut kalitesinden hesaplanan xG değildir.")
                st.metric("Toplam 2,5 gol üstü",f"%{100*result['over_2_5']:.1f}")
                st.metric("İki takım da gol atar",f"%{100*result['both_teams_score']:.1f}")
            with st.expander("Takımların form ve güç özeti"):
                form=result["form"]
                st.dataframe(pd.DataFrame({"Takım":[home,away],"Elo":[round(form['home_elo']),round(form['away_elo'])],"Son 5 maç puan ortalaması (dengelenmiş)":[round(form['home_points_last5'],2),round(form['away_points_last5'],2)]}),hide_index=True)
            st.caption("Olasılıklar kesin sonuç veya kazanç garantisi değildir. Kadro ve sakatlık verisi henüz doğrudan modele dahil değildir.")
with evaluation_tab:
    report=bundle["report"]
    selected=report["selected"]
    period=report["test_period"]
    st.subheader("Model seçiminde kullanılmayan tam sezon")
    st.caption(f"{period['from']} — {period['to']} · {period['matches']} maç")
    comparison=[{"Model":"İstatistik modeli","Doğruluk (%)":100*selected['accuracy'],"Log loss":selected['log_loss']},
                {"Model":"Piyasa destekli", "Doğruluk (%)":100*report['market_assisted']['accuracy'],"Log loss":report['market_assisted']['log_loss']}]
    for name,baseline in report["baselines"].items():
        comparison.append({"Model":{"league_mean":"Lig ortalaması","rolling_poisson":"Geçmiş gol ortalamaları"}[name],"Doğruluk (%)":100*baseline['accuracy'],"Log loss":baseline['log_loss']})
    st.dataframe(pd.DataFrame(comparison).round(3),hide_index=True,width="stretch")
    st.caption("Düşük log loss daha iyidir. Piyasa destekli satır, tarihsel maç öncesi oranlar bulunan maçları kullanır.")
    selective=bundle.get("selective_report")
    if selective:
        st.subheader("Her maç için çifte şans")
        all_rows=[]
        for mode,label in [("stats","Sadece istatistik"),("assisted","Piyasa destekli")]:
            details=selective["modes"][mode]
            result=details.get("all_match_double_chance")
            if result:
                interval=result["accuracy_95_interval"]
                all_rows.append({"Mod":label,"Bağımsız test maçı":result["total_matches"],
                    "Kapsama":pct(result["coverage"]),"Doğruluk":pct(result["accuracy"]),
                    "En iyi sabit seçim":pct(result["best_constant_accuracy"]),
                    "Model farkı":pct(result["improvement_vs_best_constant"]),
                    "%95 aralık":f"{pct(interval[0])}–{pct(interval[1])}"})
        if all_rows:
            st.dataframe(pd.DataFrame(all_rows),hide_index=True,width="stretch")
            st.caption("Her maçta en düşük olasılıklı sonuç elenir ve kalan iki sonuç verilir. Bu, exact 1-X-2'den daha kolay ayrı bir hedeftir.")
        st.subheader("%70 hedefli exact 1-X-2 filtresi")
        rows=[]
        for mode,label in [("stats","Sadece istatistik"),("assisted","Piyasa destekli")]:
            details=selective["modes"][mode]
            policy=details["policy"]
            confirmation=details["confirmation"]
            audit=details.get("latest_partial_season_audit")
            rows.append({"Mod":label,"Seçim eşiği":pct(policy.get('threshold')),
                         "Bağımsız test maçları":confirmation['selected_matches'],
                         "Bağımsız test kapsama":pct(confirmation['coverage']),
                         "Bağımsız test doğruluk":pct(confirmation['accuracy']),
                         "Güncel sezon doğruluk":pct(audit['accuracy']) if audit else "—",
                         "Güncel sezon seçilen":audit['selected_matches'] if audit else "—"})
        st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
        st.caption("Eşik yalnız eski doğrulama maçlarında belirlendi. Kapsama, modelin PAS demediği maçların oranıdır. Güncel sezon satırı daha az maç içerdiği için daha belirsizdir.")
    st.markdown("**Model güveni gerçek başarıyla ne kadar örtüşüyor?**")
    st.dataframe(pd.DataFrame([{"Tahmin güveni":b['range'],"Maç":b['matches'],"Ortalama güven (%)":round(100*b['mean_probability'],1),"Gerçek doğruluk (%)":round(100*b['accuracy'],1)} for b in selected['confidence_buckets']]),hide_index=True)
    st.info("Test ölçümleri geçmiş dönemin başında dondurulmuş modele aittir. Uygulamadaki model sonrasında eldeki tüm sonuçlarla yeniden eğitilmiştir.")
    with st.expander("Deney ayrıntıları"):
        st.write(report['protocol'])
        st.json({"weights":report['weights'],"temperature":report['temperature'],"market_weight":report['market_weight'],"validation_log_loss":report['validation_log_loss']})
