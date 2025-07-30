import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(page_title="Q-Method", layout="wide")
st.title("데이터센터 지속가능성 인식 조사")

with st.expander("📘 조사 개요", expanded=True):
    st.markdown("""
        본 조사는 데이터센터의 기술·입지·사람·거버넌스에 대한 사회적 수용성과 관련된 다양한 진술문에 대해, 귀하의 인식을 파악하고자 합니다. 
        한국공학대학교 주관 학술 연구 목적으로 수행되는 본 조사는 조사지 자체의 익명성이 유지되며 응답자 고유성을 확인하기 위해 이메일을 수집 후 파기합니다. 
        모든 섹션에 참여하시는 데 10분 이내로 소요되며, 참여해주신 분께는 약소하오나 소중한 시간을 내어주신 데 대한 감사의 의미로 자문 수당(10만원, 세전)을 별도로 송금해 드립니다.
        데이터센터는 인공지능, 클라우드, 디지털 산업 발전을 가능하게 하는 핵심 기반 시설입니다. 하지만 그와 동시에 막대한 전력을 소비하고, 물을 많이 사용하며, 입지 선정 과정에서 시민들과 갈등을 빚기도 합니다.
        동 연구에서는
        - 시민들은 데이터센터에 대해 어떤 생각을 가지고 있을까? 그
        - 리고 그 판단은 어떤 가치나 우선순위에 따라 달라질까? 를 알아보기 위한 목적을 가지고 설문조사를 시행하고 있습니다.
        설문은 총 24개의 문장을 제시하며, 이 문장들은 사람들이 데이터센터에 대해 흔히 하는 주장이나 의견을 정리한 것입니다.
        """)
with st.expander("🧩 섹션 설명", expanded=True):
    st.markdown("""
    설문은 Q-sort 방식으로 진행되며, 제시된 24개 문장을 “나는 이 생각에 얼마나 동의하는가?”의 기준으로 순서대로 정렬해 주세요.
    문장들은 다음 네 개의 관점으로 구성되어 있습니다:
      1) 기술(Technology): 이 영역은 데이터센터가 어떤 기술을 사용하는지를 시민들이 어떻게 바라보는지를 다룹니다. 예를 들어 재생에너지 사용 여부, 친환경 냉각 기술, 백업 전력 방식, 기술의 안전성과 거리감 등이 여기에 포함됩니다. 당신은 기술의 종류나 방식이 시민의 신뢰나 수용성에 어떤 영향을 줄 수 있다고 생각하십니까?
      2) 사람 (People): 이 영역은 데이터센터를 둘러싼 사람들 간의 관계와 신뢰, 참여의 방식에 관한 시민들의 인식을 다룹니다. 예를 들어 시민 의견이 반영되었는지, 설명회가 형식적이지 않았는지, 정부와 기업 중 누가 더 신뢰받는지 등이 포함됩니다. 데이터센터와 관련된 다양한 이해관계자들 사이의 신뢰와 관계가, 시민들의 수용성 판단에 어떤 영향을 준다고 생각하십니까?
      3) 장소 (Place): 이 영역은 데이터센터가 어디에 들어서느냐에 따라 시민들이 어떻게 반응하는지를 다룹니다. 기존 산업 부지의 활용, 지역 정체성과의 조화, 자연환경 훼손, 수도권/지방 차이, 외부 자본 주도의 불신 가능성 등이 포함됩니다. 당신은 입지의 특성과 맥락이 시민 수용성에 어떤 영향을 줄 수 있다고 생각하십니까?
      4) 과정 (Process): 이 영역은 데이터센터가 어떤 절차와 방식으로 결정·운영되었는지를 시민들이 어떻게 평가하는지를 다룹니다. 예를 들어 정보 공개 시점, 환경영향평가의 신뢰도, 기업–지자체 협력 여부, 사후 모니터링의 유무 등이 포함됩니다. 당신은 결정 과정의 투명성과 참여 방식이 시민의 신뢰와 수용에 어떤 영향을 줄 수 있다고 생각하십니까?
    """)

DATA_PATH = "responses.csv"

tabs = st.tabs(["✍️ 설문 응답", "📈 요인 분석", "🔁 피드백 구조"])

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

with tabs[0]:
    st.markdown("#### 아래 문항에 응답해 주세요.")
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
        st.success("응답이 저장되었습니다.")

with tabs[1]:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        st.subheader("📊 응답자 요인분석 (PCA 기반)")
        if len(df) >= 5:
            X = StandardScaler().fit_transform(df)
            pca = PCA(n_components=2)
            comps = pca.fit_transform(X)

            st.write("요인 점수:")
            st.dataframe(pd.DataFrame(comps, columns=["요인1", "요인2"]))

            fig, ax = plt.subplots()
            ax.scatter(comps[:, 0], comps[:, 1], color='green')
            for i, (x, y) in enumerate(comps):
                ax.text(x + 0.02, y + 0.02, f"R{i+1}", fontsize=8)
            ax.set_title("응답자 요인공간 (PCA)")
            st.pyplot(fig)
        else:
            st.warning("요인 분석을 위해 최소 5명의 응답이 필요합니다.")
    else:
        st.info("아직 저장된 응답이 없습니다.")

with tabs[2]:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        st.subheader("🔁 진술 간 피드백 구조 (상관 네트워크)")
        if len(df) >= 5:
            df_t = df.T
            corr = df_t.corr()
            G = nx.Graph()

            for i in range(len(statements)):
                G.add_node(f"Q{i+1}", label=statements[i])

            for i in range(len(statements)):
                for j in range(i+1, len(statements)):
                    weight = corr.iloc[i, j]
                    if abs(weight) > 0.6:
                        G.add_edge(f"Q{i+1}", f"Q{j+1}", weight=round(weight, 2))

            pos = nx.spring_layout(G, seed=42)
            plt.figure(figsize=(12, 10))
            nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=700)
            nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=9)
            edges = G.edges(data=True)
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=1)
            nx.draw_networkx_edge_labels(G, pos,
                edge_labels={(u, v): f"{d['weight']}" for u, v, d in edges},
                font_size=8)
            plt.title("진술 간 상관 기반 피드백 네트워크")
            st.pyplot(plt)
        else:
            st.warning("피드백 구조 시각화를 위해 최소 5명의 응답이 필요합니다.")
    else:
        st.info("응답 데이터가 없습니다.")
