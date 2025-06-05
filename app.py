import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64
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
    proba = rf_model.predict_proba(input_array)[0]
    top_indices = np.argsort(proba)[::-1]  # 확률 기준 내림차순 정렬
    top1_idx, top2_idx, top3_idx = top_indices[:3]

    top1_team = label_encoder.inverse_transform([top1_idx])[0]
    top2_team = label_encoder.inverse_transform([top2_idx])[0]
    top3_team = label_encoder.inverse_transform([top3_idx])[0]

    st.markdown("---")
    st.markdown("""
        <style>
        .centered { text-align: center; }
        .big-font { font-size: 50px !important; }
        .medium-font { font-size: 30px !important; }
        .small-font { font-size: 22px !important; }
        </style>
    """, unsafe_allow_html=True)

    # ✅ 1위
    st.markdown(f"""
        <div class='centered big-font'>
            🎉 당신에게 어울리는 팀은... <br><br>
            <b>{top1_team}</b>
        </div>
    """, unsafe_allow_html=True)

    def get_base64_image(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    def render_team_image(team_name, width):
        for ext in ["png", "jpg", "jpeg"]:
            img_path = f"images/{team_name}.{ext}"
            if os.path.exists(img_path):
                img_base64 = get_base64_image(img_path)
                st.markdown(f"""
                    <div class="centered">
                        <img src="data:image/{ext};base64,{img_base64}" width="{width}"/><br>
                        <p class="small-font">{team_name}</p>
                    </div>
                """, unsafe_allow_html=True)
                return
        st.warning(f"⚠️ {team_name} 로고 이미지가 없습니다.")

    # ✅ 1위 팀 로고 출력 (크게)
    render_team_image(top1_team, width=300)

    # ✅ 2위 팀 (작게, 확률 X)
    st.markdown(f"<div class='centered medium-font'>🥈 2위 후보: <b>{top2_team}</b></div>", unsafe_allow_html=True)
    render_team_image(top2_team, width=200)

    # ✅ 3위 팀 (작게, 확률 X)
    st.markdown(f"<div class='centered medium-font'>🥉 3위 후보: <b>{top3_team}</b></div>", unsafe_allow_html=True)
    render_team_image(top3_team, width=200)
