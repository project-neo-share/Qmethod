import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Q-Method Likert 설문 (멀티 유저)", layout="wide")
st.title("🧠 Q-Method 기반 Likert 설문 및 분석 (응답 누적 기반)")

DATA_PATH = "responses.csv"

statements = [
    "데이터센터는 재생에너지를 사용할 때 환경 책임성을 갖춘 시설로 평가받을 수 있다.",
    "디젤이나 가스 발전기를 백업 전력으로 사용할 경우 환경적 우려가 제기될 수 있다.",
    "물 절약이나 친환경 냉각 기술의 도입은 시민 신뢰에 긍정적 영향을 줄 수 있다.",
    "기술이 최신이더라도 안전성 확보가 부족하면 시민 불안을 유발할 수 있다.",
    "데이터센터 기술은 비용 효율성보다는 사회적 책임을 우선시해야 한다는 견해가 있다.",
    "기술이 낯설거나 복잡하게 인식되면 시민과의 거리감이 커질 수 있다.",
    "데이터센터 건설 과정에 시민 의견이 반영되지 않으면 반발 가능성이 높아질 수 있다.",
    "지역 사회와 장기적 관계를 맺어온 기업은 더 높은 신뢰를 받을 수 있다.",
    "설명회가 형식적으로 보일 경우, 시민 불신을 유발할 수 있다.",
    "정보 접근성이 낮을수록 시민의 불안과 의심이 증가할 수 있다.",
    "갈등 상황에서는 중립적 제3자의 개입이 조정에 도움이 될 수 있다.",
    "동일한 설명이라도 정부가 전달할 경우 기업보다 더 신뢰받을 가능성이 있다.",
    "기존 공장이나 발전소 부지를 재활용한 데이터센터는 수용성이 높아질 수 있다.",
    "지역 정체성과 조화를 이루지 못하는 입지는 거부감을 유발할 수 있다.",
    "자연경관 훼손이 발생하는 경우, 기술 우수성만으로 수용성 확보는 어려울 수 있다.",
    "수도권과 지방은 데이터센터 입지에 대해 서로 다른 기준을 가질 수 있다.",
    "외부 자본 주도의 일방적인 입지 결정은 지역사회의 신뢰를 저해할 수 있다.",
    "지역에 실질적인 혜택이 제공되면 시민 수용성이 높아질 수 있다.",
    "초기 단계에서 정보가 투명하게 공개되면 시민 신뢰가 높아질 수 있다.",
    "환경영향평가 결과는 시민들의 수용 여부에 중요한 판단 기준이 될 수 있다.",
    "기업과 지자체가 공동으로 결정한 프로젝트는 더 높은 신뢰를 얻을 수 있다.",
    "법적 요건을 충족하더라도 시민 신뢰를 확보하려면 추가적인 설명이 필요할 수 있다.",
    "지역 언론이 신속하고 정확하게 정보를 전달하면 신뢰성 제고에 기여할 수 있다.",
    "데이터센터 완공 이후에도 모니터링과 피드백 체계가 지속되면 신뢰 유지에 도움이 될 수 있다."
]

scale_map = {
    "전혀 동의하지 않음": 1,
    "동의하지 않음": 2,
    "보통이다": 3,
    "동의함": 4,
    "매우 동의함": 5
}
scale_labels = list(scale_map.keys())

tab1, tab2 = st.tabs(["✍️ 설문 응답", "📈 누적 분석"])

with tab1:
    st.markdown("#### 아래 문항에 응답해 주세요. (모든 항목 필수)")
    responses = {}
    with st.form(key="likert_form"):
        for idx, stmt in enumerate(statements, 1):
            response = st.radio(f"{idx}. {stmt}", options=scale_labels, key=f"stmt_{idx}")
            responses[f"Q{idx:02}"] = scale_map[response]
        submitted = st.form_submit_button("제출하기")

    if submitted:
        df_new = pd.DataFrame([responses])
        if os.path.exists(DATA_PATH):
            df_old = pd.read_csv(DATA_PATH)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_csv(DATA_PATH, index=False)
        st.success("응답이 저장되었습니다. 감사합니다!")

with tab2:
    st.markdown("#### 현재까지 수집된 응답에 기반한 분석 결과")

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        st.write(f"총 응답 수: {len(df)}명")

        if len(df) >= 10:
            st.success("응답이 10건 이상 누적되어 분석을 수행합니다.")
            X = StandardScaler().fit_transform(df)
            pca = PCA(n_components=2)
            components = pca.fit_transform(X)

            st.write("📊 PCA 요인 점수 (2차원):")
            st.dataframe(pd.DataFrame(components, columns=["요인1", "요인2"]))

            fig, ax = plt.subplots()
            ax.scatter(components[:, 0], components[:, 1], color='teal')
            for i, (x, y) in enumerate(components):
                ax.text(x + 0.02, y + 0.02, f"R{i+1}", fontsize=8)
            ax.set_xlabel("요인1")
            ax.set_ylabel("요인2")
            ax.set_title("Q-Type 요인 공간 (PCA)")
            st.pyplot(fig)
        else:
            st.warning("분석을 위해 최소 10명의 응답이 필요합니다.")
    else:
        st.info("아직 저장된 응답이 없습니다.")
