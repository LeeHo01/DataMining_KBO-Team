import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ✅ 모델 및 인코더 불러오기
model = joblib.load("rf_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ✅ 질문 목록 정의
questions = [
    "본인이 응원하는 팀의 수비력은 얼마나 뛰어나야 된다고 생각하십니까?",
    "본인이 응원하는 팀의 투수력은 얼마나 강해야 된다고 생각하십니까?",
    "본인이 응원하는 팀의 타격력은 얼마나 우수해야 된다고 생각하십니까?",
    "본인이 응원하는 팀의 구장 시설(좌석, 시설 등)은 얼마나 쾌적해야된다고 생각하십니까?",
    "본인이 응원하는 팀의 구장 내 혹은 구장 주변 먹거리(음식, 간식 등)가 얼마나 있어야 된다고 생각하십니까?",
    "본인이 응원하는 팀은 팬들과 얼마나 활발히 소통해야 한다고 생각하십니까?",
    "본인이 응원하는 팀의 팬덤 규모와 응원 열기는 얼마나 중요하다고 생각하십니까?",
    "본인이 응원하는 팀의 굿즈, 유니폼, 로고 디자인이 얼마나 감각적이어야 한다고 생각하십니까?",
    "팀에 내세울 만한 스타선수가 있어야 된다고 생각하십니까?",
    "응원하는 팀 선수들의 외적인 매력(외모, 스타일 등)도 뛰어나야 한다고 생각합니까?",
    "본인이 응원하는 팀의 성적이 좋아야 된다고 생각하십니까?",
    "본인이 응원하는 팀은 전통과 역사가 깊어야 된다고 생각하십니까?",
    "본인이 응원하는 팀의 응원가가 흥겹고 매력적이어야 된다고 생각하십니까?",
    "본인이 응원하는 팀의 치어리더와 응원 무대가 인상적이어야 된다고 생각하십니까?",
    "본인이 응원하는 팀 선수들은 열정적이고 근성이 강해야된다고 생각하십니까?",
    "본인이 응원하는 팀 선수들의 도덕성과 인성이 중요하다고 생각하십니까?"
]

st.set_page_config(page_title="KBO 팀 추천기", layout="wide")
st.title("📊 KBO 팀 성향 기반 추천기")
st.markdown("""
    성향에 따라 가장 어울리는 KBO 팀을 추천해드립니다! 
    아래 질문에 대한 응답을 1~10점 척도로 선택해 주세요.
""")

# ✅ 사용자 입력 받기
user_input = []
with st.form("survey_form"):
    for q in questions:
        score = st.slider(q, 1, 10, 5)
        user_input.append(score)
    submitted = st.form_submit_button("팀 추천받기")

# ✅ 예측 및 결과 출력
if submitted:
    user_array = np.array(user_input).reshape(1, -1)
    pred = model.predict(user_array)[0]
    proba = model.predict_proba(user_array)[0]

    team = label_encoder.inverse_transform([pred])[0]

    st.subheader(f"🎉 당신에게 추천되는 팀은: 🧢 **{team}**")

    st.markdown("---")
    st.write("🔍 각 팀이 추천될 확률:")
    proba_df = pd.DataFrame({
        "팀": label_encoder.classes_,
        "추천 확률 (%)": np.round(proba * 100, 2)
    }).sort_values("추천 확률 (%)", ascending=False)

    st.dataframe(proba_df.reset_index(drop=True), use_container_width=True)
