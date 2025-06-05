import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# ✅ 모델 및 인코더 불러오기
rf_model = joblib.load("rf_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ✅ 문항 정의
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

# ✅ Streamlit 화면 구성
st.set_page_config(page_title="KBO 팀 추천", layout="wide")
st.title("⚾ 나의 성향 기반 KBO 팀 추천")
st.markdown("16개의 문항에 답해주세요. (1점 = 중요하지 않음, 10점 = 매우 중요)")

# ✅ 사용자 입력 받기
user_input = []
for idx, question in enumerate(questions):
    st.markdown(f"### Q{idx+1}. {question}")
    col1, col2 = st.columns([1, 9])
    with col2:
        selected = st.radio("", options=list(range(1, 11)), horizontal=True, key=f"q_{idx}")
    user_input.append(selected)

# ✅ 버튼 클릭 시 예측
if st.button("✅ 나에게 맞는 팀 추천받기"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = rf_model.predict(input_array)[0]
    predicted_team = label_encoder.inverse_transform([prediction])[0]
    proba = rf_model.predict_proba(input_array)[0]

    # 화면 전환 효과
    st.markdown("""
        <style>
        .big-font {
            font-size:48px !important;
            text-align:center;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<div class='big-font'>🎉 당신에게 어울리는 팀은... <br><br><b>{predicted_team}</b>!</div>", unsafe_allow_html=True)

    # ✅ 이미지 로딩 (확장자 자동 탐색)
    image_found = False
    for ext in ["png", "jpg", "jpeg"]:
        image_path = f"images/{predicted_team}.{ext}"
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                st.image(img, caption=predicted_team, width=500)
                image_found = True
                break
            except:
                st.warning("이미지를 열 수 없습니다.")
                break
    if not image_found:
        st.warning("⚠️ 해당 팀의 로고 이미지를 찾을 수 없습니다.")

    st.markdown("---")
    st.subheader("🔍 각 팀별 예측 확률")
    proba_df = pd.DataFrame({
        '팀명': label_encoder.classes_,
        '예측 확률': np.round(proba * 100, 2)
    }).sort_values(by='예측 확률', ascending=False)

    st.dataframe(proba_df.reset_index(drop=True), use_container_width=True)
