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
st.title("⚽ Match Forecast")
st.caption("Maç öncesi form, takım gücü ve geçmiş performanstan olasılıklı tahminler")

@st.cache_resource
def load_bundle(path, modified):
    return joblib.load(path)

available = [name for name in LEAGUES if (ROOT/"artifacts"/f"{name}.joblib").exists()]
if not available:
    st.info("Henüz eğitilmiş model yok. Proje klasöründe aşağıdaki komutları çalıştırın.")
    st.code("python -m match_predictor download\npython -m match_predictor train", language="bash")
    st.stop()
league = st.sidebar.selectbox("Lig", available, format_func=lambda x: LEAGUES[x][0])
path = ROOT/"artifacts"/f"{league}.joblib"
bundle = load_bundle(str(path), path.stat().st_mtime_ns)
st.sidebar.caption(f"Son maç verisi: {bundle['last_match_date']}")
st.sidebar.caption(f"Geçmiş maç sayısı: {bundle['report']['data']['matches']:,}")
show_all = st.sidebar.checkbox("Önceki sezonların takımlarını da göster")
teams = bundle["teams"] if show_all else bundle["recent_teams"]
latest = date.fromisoformat(bundle["last_match_date"])
age = (date.today() - latest).days
if age > 30:
    st.warning(f"Veri {age} gündür güncellenmemiş. Bu tahminler güncel kadro ve sonuçları yansıtmayabilir.")
if league == "cl":
    st.warning("Şampiyonlar Ligi modeli sınırlı eski veriyle eğitilmiştir; sonuçlar deneyseldir.")
forecast_tab, evaluation_tab = st.tabs(["Maç tahmini", "Ölçülen başarı"])
with forecast_tab:
    left, right, when = st.columns([2,2,1])
    home = left.selectbox("Ev sahibi", teams)
    away = right.selectbox("Deplasman", [team for team in teams if team != home])
    match_date = when.date_input("Maç tarihi", value=max(date.today(),latest+timedelta(days=1)), min_value=latest+timedelta(days=1))
    if st.button("Tahmin et", type="primary", width="stretch"):
        try:
            result = predict_match(bundle,home,away,match_date)
        except ValueError as error:
            st.error(str(error))
        else:
            for warning in result["warnings"]:
                st.warning(warning)
            st.subheader(f"{home} — {away}")
            p = result["probabilities"]
            columns = st.columns(3)
            for column,label,value in zip(columns,[f"{home} kazanır","Beraberlik",f"{away} kazanır"],p.values()):
                column.metric(label,f"%{100*value:.1f}")
            st.bar_chart(pd.DataFrame({"Olasılık (%)":[100*p["home"],100*p["draw"],100*p["away"]]}, index=[home,"Beraberlik",away]),color="#16a085",horizontal=True)
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
            st.caption("Tahminler kadro, sakatlık ve transfer bilgisi içermez. Olasılıklar kesin sonuç anlamına gelmez.")
with evaluation_tab:
    report=bundle["report"]
    selected=report["selected"]
    period=report["test_period"]
    st.subheader("Model seçiminde kullanılmayan maçlardaki sonuçlar")
    st.caption(f"{period['from']} — {period['to']} · {period['matches']} maç")
    c1,c2,c3=st.columns(3)
    c1.metric("Maç sonucu doğruluğu",f"%{100*selected['accuracy']:.1f}")
    c2.metric("Log loss · düşük daha iyi",f"{selected['log_loss']:.3f}")
    c3.metric("Tam skor doğruluğu",f"%{100*selected['exact_score_accuracy']:.1f}")
    interval=selected["accuracy_95_interval"]
    st.caption(f"Maç sonucu doğruluğu için yaklaşık %95 aralık: %{interval[0]*100:.1f}–%{interval[1]*100:.1f}")
    comparison=[{"Model":"Seçilen karışım","Doğruluk (%)":100*selected['accuracy'],"Log loss":selected['log_loss']}]
    for name,baseline in report["baselines"].items():
        comparison.append({"Model":{"league_mean":"Lig ortalaması","rolling_poisson":"Geçmiş gol ortalamaları"}[name],"Doğruluk (%)":100*baseline['accuracy'],"Log loss":baseline['log_loss']})
    st.dataframe(pd.DataFrame(comparison).round(3),hide_index=True,width="stretch")
    st.markdown("**Model güveni gerçek başarıyla ne kadar örtüşüyor?**")
    st.dataframe(pd.DataFrame([{"Tahmin güveni":b['range'],"Maç":b['matches'],"Ortalama güven (%)":round(100*b['mean_probability'],1),"Gerçek doğruluk (%)":round(100*b['accuracy'],1)} for b in selected['confidence_buckets']]),hide_index=True)
    st.caption("Az maç içeren aralıklar belirsizdir. Test sırasında her yeni gün önceki sonuçlar özelliklere eklenir; model yeniden eğitilmez.")
    st.info("Bu ölçümler test döneminden önce eğitilen modele aittir. Tahmin ekranındaki model sonrasında eldeki tüm maçlarla yeniden eğitilmiştir.")
    with st.expander("Deney ayrıntıları"):
        st.write(report['protocol'])
        st.json({"weights":report['weights'],"temperature":report['temperature'],"validation_log_loss":report['validation_log_loss']})
