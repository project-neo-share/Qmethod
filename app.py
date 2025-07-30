import streamlit as st
import streamlit_sortables as sortables

st.set_page_config(page_title="Q-Method 버블 정렬", layout="wide")
st.title("Q-Method 진술문 정렬 (버블 드래그 방식)")

# 초기 그룹화된 버블
initial_groups = {
    "Group A": ["1", "2", "3", "4", "5", "6"],
    "Group B": ["7", "8", "9", "10", "11", "12"],
    "Group C": ["13", "14", "15", "16", "17", "18"],
    "Group D": ["19", "20", "21", "22", "23", "24"]
}

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("왼쪽 그룹에서 드래그하여 정렬하세요")
    bubble_lists = sortables.sort_items(
        [initial_groups[group] for group in initial_groups],
        direction="horizontal",
        multi_containers=True,
        container_labels=list(initial_groups.keys()),
        key="bubble_input"
    )

# 오른쪽: 최종 순위 정렬
with col2:
    st.subheader("오른쪽 영역에서 최종 순위를 정렬하세요")

    # 정렬된 왼쪽 그룹의 모든 항목 병합
    all_items = [item for sublist in bubble_lists for item in sublist]

    ranked_list = sortables.sort_items(
        [all_items],
        direction="vertical",
        multi_containers=False,
        container_labels=["순위 영역 (1위 ~ 24위)"],
        key="final_ranking"
    )

    # 결과 출력
    if st.button("제출"):
        st.success("정렬이 완료되었습니다. 아래는 당신의 순위입니다:")
        for idx, item in enumerate(ranked_list[0], 1):
            st.write(f"{idx}위: 진술문 {item}")
