"""
Q-Method Streamlit Application

Author      : Prof. Dr. Songhee Kang  
Last Update : 2025-07-31  
Description : Likert-based Q-Method survey tool with GitHub push integration
"""
from github import Github
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import platform
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import datetime
import networkx as nx
import matplotlib.font_manager as fm

st.set_page_config(page_title="Q-Method Analyzer", layout="wide")
st.title("데이터센터 지속가능성 인식 조사")

DATA_PATH = "responses.csv"
# 사이드바 관리자 로그인 영역
# -----------------------------
# Secrets (GitHub)
# -----------------------------
def _get_secret(path, default=""):
    try:
        cur = st.secrets
        for key in path.split("."):
            cur = cur[key]
        return cur
    except Exception:
        return default

GH_TOKEN   = _get_secret("github.token")
GH_REPO    = _get_secret("github.repo")
GH_BRANCH  = _get_secret("github.branch", "main")
GH_REMOTEP = _get_secret("github.data_path", DATA_PATH)  # 원격 저장 경로
GH_README  = _get_secret("github.readme_path", "README.md")         # (옵션)

# -----------------------------
# 실시간 척도 현황 계산
# -----------------------------
def calc_scale_counts(answers):
    counts = {i: 0 for i in range(1, 6)}
    for v in answers.values():
        counts[v] += 1
    return counts

def _gh_headers(token):
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "streamlit-qmethod-sns"
    }

def gh_get_sha(owner_repo, path, token, branch):
    url = f"https://api.github.com/repos/{owner_repo}/contents/{path}"
    r = requests.get(url, headers=_gh_headers(token), params={"ref": branch}, timeout=20)
    if r.status_code == 200:
        try:
            return r.json().get("sha")
        except Exception:
            return None
    elif r.status_code == 404:
        return None
    else:
        raise RuntimeError(f"GitHub GET 실패: {r.status_code} {r.text}")

def gh_put_file(owner_repo, path, token, branch, content_bytes, message):
    url = f"https://api.github.com/repos/{owner_repo}/contents/{path}"
    b64 = base64.b64encode(content_bytes).decode("ascii")
    sha = gh_get_sha(owner_repo, path, token, branch)
    payload = {"message": message, "content": b64, "branch": branch}
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=_gh_headers(token), data=json.dumps(payload), timeout=30)
    if r.status_code in (200, 201):
        return True, r.json()
    return False, f"{r.status_code}: {r.text}"

def push_csv_to_github(local_path, remote_path=None, note="Update survey_data.csv"):
    if not (GH_TOKEN and GH_REPO):
        return False, "GitHub secrets 누락(github.token, github.repo)"
    if remote_path is None:
        remote_path = GH_REMOTEP
    try:
        with open(local_path, "rb") as f:
            content = f.read()
    except Exception as e:
        return False, f"로컬 CSV 읽기 실패: {e}"
    ok, resp = gh_put_file(GH_REPO, remote_path, GH_TOKEN, GH_BRANCH, content, note)
    return ok, resp

# -----------------------------
# 사이드바 : 실시간 현황 패널
# -----------------------------

with st.sidebar:
    st.subheader("🔐 관리자 / 동기화")
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    admin_pw = st.sidebar.text_input("관리자 비밀번호 (선택)", type="password")
    if st.sidebar.button("로그인"):
        if admin_pw and _get_secret("admin.password") == admin_pw:
            st.session_state.authenticated = True
            st.sidebar.success("인증 성공")
        else:
            st.sidebar.error("인증 실패")

    auto_sync = st.sidebar.checkbox("응답 저장 시 GitHub 자동 푸시", value=True)

    st.subheader("📊 실시간 척도 현황")
    counts = calc_scale_counts(st.session_state['answers'])
    if st.button("🔄 새로고침"):
        st.rerun()

    df_counts = pd.DataFrame({
        "척도": [LIKERT[i-1] for i in range(1, 6)],
        "선택 문항 수": [counts[i] for i in range(1, 6)],
        "최대 허용 개수": [MAX_COUNT[i] for i in range(1, 6)],
    })

    st.dataframe(df_counts, width="content")

    # Plotly 그래프 표시
    fig = go.Figure(data=[
        go.Bar(name="선택 문항 수", x=LIKERT, y=[counts[i] for i in range(1,6)], marker_color='skyblue'),
        go.Bar(name="최대 허용 개수", x=LIKERT, y=[MAX_COUNT[i] for i in range(1,6)], marker_color='salmon')
    ])
    fig.update_layout(
        barmode='group',
        yaxis_title="문항 수",
        xaxis_tickangle=-20,
        template="plotly_white",
        height=350,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Utils
# -----------------------------



def is_valid_email(s: str) -> bool:
    if not s: return False
    s = s.strip()
    if len(s) > 150: return False
    return bool(EMAIL_RE.match(s))

def load_csv_safe(path: str):
    if not os.path.exists(path):
        return None
    try:
        if os.path.getsize(path) == 0:
            return None
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None

def save_csv_safe(df: pd.DataFrame, path: str):
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return True
    except Exception as e:
        st.error(f"CSV 저장 실패: {e}")
        return False

def ensure_q_columns(df: pd.DataFrame, q_count: int):
    cols = [f"Q{i:02d}" for i in range(1, q_count + 1)]
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    return df, cols

def zscore_rows(a: np.ndarray):
    m = a.mean(axis=1, keepdims=True)
    s = a.std(axis=1, ddof=0, keepdims=True)
    s = np.where(s < EPS, 1.0, s)
    return (a - m) / s

def rank_rows(a: np.ndarray):
    df = pd.DataFrame(a)
    return df.rank(axis=1, method="average", na_option="keep").values

def varimax(Phi, gamma=1.0, q=100, tol=1e-6, seed=42):
    Phi = Phi.copy(); p, k = Phi.shape
    R = np.eye(k); d_old = 0
    for _ in range(q):
        Lambda = Phi @ R
        u, s, vh = np.linalg.svd(
            Phi.T @ (Lambda**3 - (gamma/p) * (Lambda @ np.diag(np.sum(Lambda**2, axis=0))))
        )
        R = u @ vh
        d = np.sum(s)
        if d_old != 0 and d/d_old < 1 + tol: break
        d_old = d
    return Phi @ R, R

def choose_n_factors(eigvals, nmax):
    k = int(np.sum(eigvals >= 1.0))
    return max(2, min(nmax, k))

st.sidebar.subheader("🔐 관리자 로그인")

# 세션 상태 초기화
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# 비밀번호 입력 받기
if not st.session_state.authenticated:
    input_password = st.sidebar.text_input("비밀번호 입력", type="password")
    if st.sidebar.button("로그인"):
        if input_password == st.secrets["admin"]["password"]:
            st.session_state.authenticated = True
            st.sidebar.success("인증 성공!")
        else:
            st.sidebar.error("비밀번호가 틀렸습니다.")
if st.session_state.authenticated:
    # 인증된 경우 다운로드 버튼 표시
    st.sidebar.success("관리자 모드 활성화됨")

    if os.path.exists(DATA_PATH):
        try:
            df_download = pd.read_csv(DATA_PATH)
            st.sidebar.download_button(
                label="📥 응답 데이터 다운로드",
                data=df_download.to_csv(index=False).encode("utf-8-sig"),
                file_name="responses.csv",
                mime="text/csv"
            )
        except pd.errors.EmptyDataError:
            st.sidebar.warning("⚠️ 저장된 응답 파일이 비어 있습니다.")
    else:
        st.sidebar.info("ℹ️ 아직 저장된 응답 파일이 없습니다.")
        
def get_korean_fontprop():
    font_path = "fonts/NanumGothic.ttf"
    if os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    else:
        return fm.FontProperties()  # fallback
font_prop = get_korean_fontprop()

def push_to_github(local_file_path):
    g = Github(st.secrets["github"]["token"])
    repo = g.get_repo(st.secrets["github"]["repo"])
    path_in_repo = st.secrets["github"]["path"]

    # 현재 파일을 읽고 base64로 인코딩
    with open(local_file_path, "rb") as file:
        content = file.read()

    try:
        # 기존 파일 정보 가져오기 (SHA 필요)
        contents = repo.get_contents(path_in_repo)
        repo.update_file(
            path=path_in_repo,
            message=f"Update response.csv at {datetime.datetime.now().isoformat()}",
            content=content,
            sha=contents.sha
        )
    except Exception:
        # 파일이 없으면 새로 생성
        repo.create_file(
            path=path_in_repo,
            message=f"Create response.csv at {datetime.datetime.now().isoformat()}",
            content=content
        )

with st.expander("📘 조사 개요", expanded=True):
    st.markdown("""
    본 조사는 데이터센터의 기술·입지·사람·거버넌스에 대한 사회적 수용성과 관련된 다양한 진술문에 대해, 귀하의 인식을 파악하고자 합니다.<br>
    한국공학대학교 주관 학술 연구 목적으로 수행되는 본 조사는 조사지 자체의 익명성이 유지되며 응답자 고유성을 확인하기 위해 이메일을 수집 후 파기합니다.<br>
    모든 섹션에 참여하시는 데 10분 이내로 소요됩니다.<br>
    <br>
    데이터센터는 인공지능, 클라우드, 디지털 산업 발전을 가능하게 하는 핵심 기반 시설입니다. 하지만 그와 동시에 막대한 전력을 소비하고, 물을 많이 사용하며, 입지 선정 과정에서 시민들과 갈등을 빚기도 합니다.<br>
    <br>
    동 연구에서는<br>
    - 시민들은 데이터센터에 대해 어떤 생각을 가지고 있을까?<br>
    - 그리고 그 판단은 어떤 가치나 우선순위에 따라 달라질까?<br>
    를 알아보기 위한 목적을 가지고 설문조사를 시행하고 있습니다.<br><br>
    설문은 총 24개의 문장을 제시하며, 이 문장들은 사람들이 데이터센터에 대해 흔히 하는 주장이나 의견을 정리한 것입니다.
""", unsafe_allow_html=True)
with st.expander("🧩 섹션 설명", expanded=True):
    st.markdown("""
    설문은 리커트 방식으로 진행되며, 제시된 24개 문장을 “나는 이 생각에 얼마나 동의하는가?”의 기준으로 입력해 주세요.<br>
    <b>매우 동의하거나 동의하지 않는 문장은 총 1-3문장 이내로 하시고, 기본적으로 중립적이거나 판단을 유보하시고 싶은 문장은 주로 보통이다로 선택해주세요.</b><br>
    문장들은 다음 네 개의 관점으로 구성되어 있습니다:<br>
      1) 기술(Technology): 이 영역은 데이터센터가 어떤 기술을 사용하는지를 시민들이 어떻게 바라보는지를 다룹니다. 예를 들어 재생에너지 사용 여부, 친환경 냉각 기술, 백업 전력 방식, 기술의 안전성과 거리감 등이 여기에 포함됩니다. 당신은 기술의 종류나 방식이 시민의 신뢰나 수용성에 어떤 영향을 줄 수 있다고 생각하십니까?<br>
      2) 사람 (People): 이 영역은 데이터센터를 둘러싼 사람들 간의 관계와 신뢰, 참여의 방식에 관한 시민들의 인식을 다룹니다. 예를 들어 시민 의견이 반영되었는지, 설명회가 형식적이지 않았는지, 정부와 기업 중 누가 더 신뢰받는지 등이 포함됩니다. 데이터센터와 관련된 다양한 이해관계자들 사이의 신뢰와 관계가, 시민들의 수용성 판단에 어떤 영향을 준다고 생각하십니까?<br>
      3) 장소 (Place): 이 영역은 데이터센터가 어디에 들어서느냐에 따라 시민들이 어떻게 반응하는지를 다룹니다. 기존 산업 부지의 활용, 지역 정체성과의 조화, 자연환경 훼손, 수도권/지방 차이, 외부 자본 주도의 불신 가능성 등이 포함됩니다. 당신은 입지의 특성과 맥락이 시민 수용성에 어떤 영향을 줄 수 있다고 생각하십니까?<br>
      4) 과정 (Process): 이 영역은 데이터센터가 어떤 절차와 방식으로 결정·운영되었는지를 시민들이 어떻게 평가하는지를 다룹니다. 예를 들어 정보 공개 시점, 환경영향평가의 신뢰도, 기업–지자체 협력 여부, 사후 모니터링의 유무 등이 포함됩니다. 당신은 결정 과정의 투명성과 참여 방식이 시민의 신뢰와 수용에 어떤 영향을 줄 수 있다고 생각하십니까?<br>
    """, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["✍️ 설문 응답",  "📊", "🧠"])


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

section_map = {
    "Technology": range(0, 6),
    "People": range(6, 12),
    "Place": range(12, 18),
    "Process": range(18, 24)
}

scale_map = {
    "전혀 동의하지 않음": 1,
    "동의하지 않음": 2,
    "보통이다": 3,
    "동의함": 4,
    "매우 동의함": 5
}
scale_labels = list(scale_map.keys())

with tab1:
    st.subheader("✍️ 설문에 응답해 주세요")
    responses = {}

    with st.form(key="likert_form"):
        email = st.text_input("이메일을 입력해 주세요 (필수 사항)", key="email_input")

        for idx, stmt in enumerate(statements, 1):
            response = st.radio(
                f"{idx}. {stmt}", options=scale_labels, key=f"stmt_{idx}", horizontal=True
            )
            responses[f"Q{idx:02}"] = scale_map[response]

        # 이메일도 응답에 추가
        responses["email"] = email.strip()

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
        push_to_github(DATA_PATH)

with tab2:
    if os.path.exists(DATA_PATH):

        df = pd.read_csv(DATA_PATH)
        st.subheader("📈 유형 분석 및 TPPP 영역별 프로파일링")
        if len(df) >= 5:
            # 수치형 컬럼만 추출
            df_numeric = df.select_dtypes(include=[np.number])

            # 동일한 shape의 노이즈 생성 후 더하기
            noise = np.random.normal(0, 0.001, df_numeric.shape)
            df_noise = df_numeric + noise

            # ▶️ 고유값 기반으로 factor 수 자동 결정
            fa_temp = FactorAnalyzer(rotation=None)
            fa_temp.fit(df_noise)
            eigen_values, _ = fa_temp.get_eigenvalues()
            n_factors = sum(eigen_values >= 1.0)

            st.info(f"🔍 고유값 1.0 이상 기준, 추출된 요인 수: {n_factors}개")

            # ▶️ 실제 분석
            fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
            fa.fit(df_noise)

            # ▶️ 요인 부하 행렬 생성
            loadings = pd.DataFrame(
                fa.loadings_,
                index=[f"Q{idx+1}" for idx in range(df_numeric.shape[1])],
                columns=[f"Type{i+1}" for i in range(n_factors)]
            )

            st.write("📌 유형 부하 행렬:")
            st.dataframe(loadings)

            # ▶️ TPPP 영역별 프로파일 요약
            st.write("📊 유형별 TPPP 평균 프로파일")
            result = []
            for factor in loadings.columns:
                scores = []
                for sec, idxs in section_map.items():
                    mean = loadings.loc[[f"Q{i+1}" for i in idxs], factor].mean()
                    scores.append((sec, mean))
                row = pd.DataFrame(dict(scores), index=[factor])
                result.append(row)
            summary = pd.concat(result)
            st.dataframe(summary.style.background_gradient(axis=1, cmap='Blues'))

            fig, ax = plt.subplots()
            summary.T.plot(kind='bar', ax=ax)
            ax.set_title("유형별 TPPP 영역 점수", fontproperties=font_prop)
            st.pyplot(fig)
        else:
            st.warning("최소 5명의 응답이 필요합니다.")

with tab3:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        st.subheader("🧠 TPPP 인지 흐름 및 피드백 구조 요약")

        if len(df) >= 5:
            # 수치형 컬럼만 추출
            df_numeric = df.select_dtypes(include=[np.number])
            
            # 동일한 shape의 노이즈 생성 후 더하기
            noise = np.random.normal(0, 0.001, df_numeric.shape)
            df = df_numeric + noise
            
            # 상관행렬 계산
            corr = df.corr()
            tp_labels = list(section_map.keys())
            block_corr = pd.DataFrame(index=tp_labels, columns=tp_labels, dtype=float)

            for sec1, idxs1 in section_map.items():
                for sec2, idxs2 in section_map.items():
                    sub_corrs = [corr.iloc[i, j] for i in idxs1 for j in idxs2 if i != j]
                    block_corr.loc[sec1, sec2] = np.mean(sub_corrs)

            # DiGraph 방향성 부여 (강한 방향 기준)
            DG = nx.DiGraph()
            for i in tp_labels:
                DG.add_node(i)

            for i in tp_labels:
                for j in tp_labels:
                    if i != j:
                        weight_ij = block_corr.loc[i, j]
                        weight_ji = block_corr.loc[j, i]
                        if weight_ij > weight_ji and weight_ij > 0.4:
                            DG.add_edge(i, j, weight=round(weight_ij, 2))

            st.markdown("### 🔄 TPPP 인지 흐름 방향 그래프 (DiGraph)")
            pos = nx.circular_layout(DG)
            plt.figure(figsize=(6, 6))
            nx.draw_networkx_nodes(DG, pos, node_color='skyblue', node_size=2000)
            nx.draw_networkx_labels(DG, pos, font_size=12, font_family=font_prop.get_name())
            nx.draw_networkx_edges(DG, pos, width=2, arrows=True, arrowstyle='-|>')
            edge_labels = {(u, v): f"{d['weight']}" for u, v, d in DG.edges(data=True)}
            nx.draw_networkx_edge_labels(DG, pos, edge_labels=edge_labels, font_size=10, font_family=font_prop.get_name())
            plt.title("TPPP 영역 간 인지 흐름 구조 (DiGraph)", fontproperties=font_prop)
            st.pyplot(plt)

            # 루프 탐지
            st.markdown("### 🔁 피드백 루프 구조 감지 결과")
            cycles = [cycle for cycle in nx.simple_cycles(DG) if len(cycle) >= 3]

            if cycles:
                for i, loop in enumerate(cycles, 1):
                    st.markdown(f"- 루프 {i}: {' → '.join(loop)} → {loop[0]}")
            else:
                st.info("루프(자기강화 피드백 구조)는 발견되지 않았습니다.")

            # 히트맵 출력
            st.markdown("### 📊 TPPP 상관 행렬 히트맵")
            fig2, ax2 = plt.subplots()
            sns.heatmap(block_corr.astype(float), annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                        fmt=".2f", linewidths=0.5, ax=ax2, cbar=True)
            ax2.set_title("TPPP 블록 간 상관 히트맵", fontproperties=font_prop)
            ax2.set_xticklabels(ax2.get_xticklabels(), fontproperties=font_prop)
            ax2.set_yticklabels(ax2.get_yticklabels(), fontproperties=font_prop)
            st.pyplot(fig2)
        else:
            st.warning("최소 5명의 응답이 필요합니다.")
    else:
        st.info("응답 데이터가 없습니다.")

