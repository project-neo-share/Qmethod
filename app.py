"""
Q-Method Streamlit Application

Author      : Prof. Dr. Songhee Kang  
Last Update : 2025-12-08  
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

import requests
import base64
import json
import re
import plotly.graph_objects as go

# ---------------------------------
# 기본 설정
# ---------------------------------
st.set_page_config(page_title="Q-Method Analyzer", layout="wide")
st.title("데이터센터 지속가능성 인식 조사")

DATA_PATH = "responses.csv"
EPS = 1e-8
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Likert 스케일 및 허용 개수 (24문항 기준 Q-sort형 분포 예시: 2 + 5 + 10 + 5 + 2 = 24)
LIKERT = ["전혀 동의하지 않음", "동의하지 않음", "보통이다", "동의함", "매우 동의함"]
MAX_COUNT = {
    1: 2,   # 전혀 동의하지 않음
    2: 5,   # 동의하지 않음
    3: 10,  # 보통이다
    4: 5,   # 동의함
    5: 2    # 매우 동의함
}

# ---------------------------------
# 문항 정의 / TPPP 매핑
# ---------------------------------
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

# ---------------------------------
# GitHub secrets helper
# ---------------------------------
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
GH_REMOTEP = _get_secret("github.data_path", DATA_PATH)
GH_README  = _get_secret("github.readme_path", "README.md")

# ---------------------------------
# (선택) REST API 방식 GitHub 업로드 유틸
# ---------------------------------
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

def push_csv_to_github_rest(local_path, remote_path=None, note="Update survey_data.csv"):
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

# ---------------------------------
# 세션 상태 초기화 (answers / auth / auto_sync)
# ---------------------------------
if "answers" not in st.session_state:
    # 기본값: 모두 보통(3점)으로 초기화
    st.session_state["answers"] = {
        f"Q{i:02}": 3 for i in range(1, len(statements) + 1)
    }

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "auto_sync" not in st.session_state:
    st.session_state["auto_sync"] = True

# ---------------------------------
# 유틸 함수
# ---------------------------------
def calc_scale_counts(answers: dict):
    counts = {i: 0 for i in range(1, 6)}
    for v in answers.values():
        if v in counts:
            counts[v] += 1
    return counts

def is_valid_email(s: str) -> bool:
    if not s:
        return False
    s = s.strip()
    if len(s) > 150:
        return False
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
        if c not in df.columns:
            df[c] = np.nan
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
    Phi = Phi.copy()
    p, k = Phi.shape
    R = np.eye(k)
    d_old = 0
    for _ in range(q):
        Lambda = Phi @ R
        u, s, vh = np.linalg.svd(
            Phi.T @ (Lambda**3 - (gamma/p) * (Lambda @ np.diag(np.sum(Lambda**2, axis=0))))
        )
        R = u @ vh
        d = np.sum(s)
        if d_old != 0 and d / d_old < 1 + tol:
            break
        d_old = d
    return Phi @ R, R

def choose_n_factors(eigvals, nmax):
    k = int(np.sum(eigvals >= 1.0))
    return max(2, min(nmax, k))

def get_korean_fontprop():
    font_path = "fonts/NanumGothic.ttf"
    if os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    else:
        return fm.FontProperties()  # fallback

font_prop = get_korean_fontprop()

def push_to_github(local_file_path):
    """PyGithub 기반 CSV 푸시"""
    try:
        g = Github(st.secrets["github"]["token"])
        repo = g.get_repo(st.secrets["github"]["repo"])
        path_in_repo = st.secrets["github"]["path"]

        with open(local_file_path, "rb") as file:
            content = file.read()

        try:
            contents = repo.get_contents(path_in_repo)
            repo.update_file(
                path=path_in_repo,
                message=f"Update response.csv at {datetime.datetime.now().isoformat()}",
                content=content,
                sha=contents.sha
            )
        except Exception:
            repo.create_file(
                path=path_in_repo,
                message=f"Create response.csv at {datetime.datetime.now().isoformat()}",
                content=content
            )
        return True
    except Exception as e:
        st.error(f"GitHub 업로드 실패: {e}")
        return False

# ---------------------------------
# 사이드바 (관리자 로그인 + 실시간 척도 현황)
# ---------------------------------
with st.sidebar:
    st.subheader("🔐 관리자 / 동기화")

    pw_input = st.text_input("관리자 비밀번호 (선택)", type="password")
    if st.button("로그인"):
        if pw_input and _get_secret("admin.password") == pw_input:
            st.session_state["authenticated"] = True
            st.success("인증 성공")
        else:
            st.error("인증 실패")

    auto_sync = st.checkbox(
        "응답 저장 시 GitHub 자동 푸시",
        value=st.session_state.get("auto_sync", True)
    )
    st.session_state["auto_sync"] = auto_sync

    st.subheader("📊 실시간 척도 현황")

    counts_sidebar = calc_scale_counts(st.session_state["answers"])

    if st.button("🔄 새로고침"):
        st.rerun()

    df_counts = pd.DataFrame({
        "척도": LIKERT,
        "선택 문항 수": [counts_sidebar[i] for i in range(1, 6)],
        "최대 허용 개수": [MAX_COUNT[i] for i in range(1, 6)],
    })
    st.dataframe(df_counts, use_container_width=True)

    fig = go.Figure(data=[
        go.Bar(name="선택 문항 수", x=LIKERT, y=[counts_sidebar[i] for i in range(1, 6)]),
        go.Bar(name="최대 허용 개수", x=LIKERT, y=[MAX_COUNT[i] for i in range(1, 6)])
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

    # 관리자 모드에서 응답 CSV 다운로드
    if st.session_state["authenticated"]:
        st.success("관리자 모드 활성화됨")
        if os.path.exists(DATA_PATH):
            try:
                df_download = pd.read_csv(DATA_PATH)
                st.download_button(
                    label="📥 응답 데이터 다운로드",
                    data=df_download.to_csv(index=False).encode("utf-8-sig"),
                    file_name="responses.csv",
                    mime="text/csv"
                )
            except pd.errors.EmptyDataError:
                st.warning("⚠️ 저장된 응답 파일이 비어 있습니다.")
        else:
            st.info("ℹ️ 아직 저장된 응답 파일이 없습니다.")

# ---------------------------------
# 본문: 안내 섹션
# ---------------------------------
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
      1) 기술(Technology)<br>
      2) 사람(People)<br>
      3) 장소(Place)<br>
      4) 과정(Process)<br>
    """, unsafe_allow_html=True)

# ---------------------------------
# 탭 구성
# ---------------------------------
tab1, tab2, tab3 = st.tabs(["✍️ 설문 응답", "📊", "🧠"])

# ---------------------------------
# Tab 1: 설문 응답
# ---------------------------------
with tab1:
    st.subheader("✍️ 설문에 응답해 주세요")

    # 이메일 입력
    email = st.text_input("이메일을 입력해 주세요 (필수 사항)", key="email_input")

    # ==========================================
    # [추가됨] 사용자 인적사항 입력 필드
    # ==========================================
    col1, col2 = st.columns(2)
    with col1:
        expertise = st.selectbox(
            "1) 나는 다음 분야의 전문가이다",
            ["전력망 인프라", "데이터센터 운영", "플랫폼 구축"],
            key="expertise_input"
        )
    with col2:
        experience_years = st.number_input(
            "2) 경력년수 (년)",
            min_value=0, max_value=60, step=1, value=0,
            key="experience_input"
        )
    
    affiliation = st.selectbox(
        "3) 소속 유형",
        ["학계", "산업계", "협회/출연연/공공기관"],
        key="affiliation_input"
    )
    st.markdown("---") # 구분선 추가
    # ==========================================

    # 문항별 라디오 버튼 – 선택 시 바로 session_state["answers"] 반영
    for idx, stmt in enumerate(statements, 1):
        q_key = f"Q{idx:02}"

        current_val = st.session_state["answers"].get(q_key, 3)
        current_label = [k for k, v in scale_map.items() if v == current_val][0]
        default_index = scale_labels.index(current_label)

        selected_label = st.radio(
            f"{idx}. {stmt}",
            options=scale_labels,
            index=default_index,
            key=q_key,
            horizontal=True
        )

        st.session_state["answers"][q_key] = scale_map[selected_label]

    # 제출 버튼 – 현재 session_state["answers"]를 그대로 저장
    if st.button("제출하기"):
        # 1) 이메일 검증
        if not is_valid_email(email):
            st.error("올바른 이메일 주소를 입력해 주세요.")
        else:
            # 2) 응답 분포 검증 (MAX_COUNT 초과 여부 체크)
            counts_current = calc_scale_counts(st.session_state["answers"])
            over = {
                i: counts_current[i]
                for i in counts_current
                if counts_current[i] > MAX_COUNT[i]
            }

            if over:
                # 초과된 척도별로 상세 메시지
                lines = []
                for i, cnt in over.items():
                    lines.append(
                        f"- '{LIKERT[i-1]}' 선택 문항 수: {cnt}개 (허용 {MAX_COUNT[i]}개 이내)"
                    )
                st.error(
                    "응답 분포가 허용 개수를 초과했습니다. "
                    "사이드바의 '최대 허용 개수'를 참고하여 아래 척도의 개수를 조정해 주세요.\n\n"
                    + "\n".join(lines)
                )
            else:
                # 3) 분포가 허용 범위 이내이면 저장
                responses = dict(st.session_state["answers"])
                responses["email"] = email.strip()
                
                # [추가됨] 추가 입력 필드 데이터 저장
                responses["expertise"] = expertise
                responses["experience_years"] = experience_years
                responses["affiliation"] = affiliation

                df_new = pd.DataFrame([responses])
                if os.path.exists(DATA_PATH):
                    df_old = pd.read_csv(DATA_PATH)
                    df_all = pd.concat([df_old, df_new], ignore_index=True)
                else:
                    df_all = df_new

                if save_csv_safe(df_all, DATA_PATH):
                    st.success("응답이 저장되었습니다.")
                    if st.session_state.get("auto_sync", True):
                        push_to_github(DATA_PATH)

# ---------------------------------
# Tab 2: 유형 분석 / TPPP 프로파일링
# ---------------------------------
with tab2:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        st.subheader("📈 유형 분석 및 TPPP 영역별 프로파일링")
        if len(df) >= 5:
            df_numeric = df.select_dtypes(include=[np.number])
            # Drop extra numeric columns if they exist (like experience_years) to avoid factor analysis error
            # 문항(Q01~Q24)만 선택하도록 필터링
            q_cols = [c for c in df_numeric.columns if c.startswith("Q")]
            df_numeric_q = df_numeric[q_cols]

            noise = np.random.normal(0, 0.001, df_numeric_q.shape)
            df_noise = df_numeric_q + noise
            df_noise_numeric = df_noise.apply(pd.to_numeric, errors='coerce')
            df_noise_numeric = df_noise_numeric.dropna()
            df_noise_numeric = df_noise.select_dtypes(include=[np.number])

            fa_temp = FactorAnalyzer(rotation=None)
            
            fa_temp.fit(df_noise)
            eigen_values, _ = fa_temp.get_eigenvalues()
            n_factors = sum(eigen_values >= 1.0)

            st.info(f"🔍 고유값 1.0 이상 기준, 추출된 요인 수: {n_factors}개")

            fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
            fa.fit(df_noise)

            loadings = pd.DataFrame(
                fa.loadings_,
                index=[f"Q{idx+1:02d}" for idx in range(df_numeric_q.shape[1])],
                columns=[f"Type{i+1}" for i in range(n_factors)]
            )

            st.write("📌 유형 부하 행렬:")
            st.dataframe(loadings)

            st.write("📊 유형별 TPPP 평균 프로파일")
            result = []
            for factor in loadings.columns:
                scores = []
                for sec, idxs in section_map.items():
                    mean = loadings.loc[[f"Q{i+1:02d}" for i in idxs], factor].mean()
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
    else:
        st.info("응답 데이터가 없습니다.")

# ---------------------------------
# Tab 3: TPPP 인지 흐름 / 피드백 구조
# ---------------------------------
with tab3:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        st.subheader("🧠 TPPP 인지 흐름 및 피드백 구조 요약")

        if len(df) >= 5:
            df_numeric = df.select_dtypes(include=[np.number])
            # 문항(Q01~Q24)만 선택
            q_cols = [c for c in df_numeric.columns if c.startswith("Q")]
            df_numeric_q = df_numeric[q_cols]

            noise = np.random.normal(0, 0.001, df_numeric_q.shape)
            df_n = df_numeric_q + noise

            corr = df_n.corr()
            tp_labels = list(section_map.keys())
            block_corr = pd.DataFrame(index=tp_labels, columns=tp_labels, dtype=float)

            for sec1, idxs1 in section_map.items():
                for sec2, idxs2 in section_map.items():
                    sub_corrs = [corr.iloc[i, j] for i in idxs1 for j in idxs2 if i != j]
                    block_corr.loc[sec1, sec2] = np.mean(sub_corrs)

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
            nx.draw_networkx_edge_labels(
                DG, pos, edge_labels=edge_labels,
                font_size=10, font_family=font_prop.get_name()
            )
            plt.title("TPPP 영역 간 인지 흐름 구조 (DiGraph)", fontproperties=font_prop)
            st.pyplot(plt)

            st.markdown("### 🔁 피드백 루프 구조 감지 결과")
            cycles = [cycle for cycle in nx.simple_cycles(DG) if len(cycle) >= 3]

            if cycles:
                for i, loop in enumerate(cycles, 1):
                    st.markdown(f"- 루프 {i}: {' → '.join(loop)} → {loop[0]}")
            else:
                st.info("루프(자기강화 피드백 구조)는 발견되지 않았습니다.")

            st.markdown("### 📊 TPPP 상관 행렬 히트맵")
            fig2, ax2 = plt.subplots()
            sns.heatmap(
                block_corr.astype(float),
                annot=True,
                cmap='coolwarm',
                vmin=-1, vmax=1,
                fmt=".2f",
                linewidths=0.5,
                ax=ax2,
                cbar=True
            )
            ax2.set_title("TPPP 블록 간 상관 히트맵", fontproperties=font_prop)
            ax2.set_xticklabels(ax2.get_xticklabels(), fontproperties=font_prop)
            ax2.set_yticklabels(ax2.get_yticklabels(), fontproperties=font_prop)
            st.pyplot(fig2)
        else:
            st.warning("최소 5명의 응답이 필요합니다.")
    else:
        st.info("응답 데이터가 없습니다.")
