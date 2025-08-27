import streamlit as st
from PIL import Image
import torch
from transformers import ViTForImageClassification, AutoImageProcessor

# ---------------------------
# 0. Streamlit 제목
# ---------------------------
st.title("🍱 음식 이미지 분류기 (ViT-B16 HuggingFace)")

# ⚠️ 데모 안내 배너
st.markdown(
    """
    <div style='padding:15px; background-color:#ffe6e6; border:2px solid #ff4d4d; border-radius:10px;'>
        <h3 style='color:#cc0000;'>⚠️ 현재는 데모 버전입니다.</h3>
        <p style='font-size:16px; color:#333;'>
        영양정보 카드 출력은 <b>baklava, beef tartare, cannoli, churros</b> 4개 클래스만 지원합니다.<br>
        50개의 음식분류가 가능하지만 영양정보 카드는 표시되지 않습니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# 1. 클래스 이름 불러오기
# ---------------------------
with open("classes.txt", "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f.readlines()]

# ---------------------------
# 2. 모델 + processor 정의
# ---------------------------
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(classes),
    ignore_mismatched_sizes=True
)
state_dict = torch.load("vit_best.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

# ---------------------------
# 3. 영양 정보 카드 (JSON - 데모용)
# ---------------------------
food_info = {
    "baklava": {
        "칼로리": "430 kcal (100g)",
        "탄수화물": "55 g",
        "단백질": "6 g",
        "지방": "22 g",
        "설명": "필로 반죽과 견과류, 꿀을 층층이 쌓아 만든 지중해 지역의 달콤한 디저트"
    },
    "beef tartare": {
        "칼로리": "250 kcal (100g)",
        "탄수화물": "0 g",
        "단백질": "26 g",
        "지방": "16 g",
        "설명": "신선한 생고기를 다져 양념과 함께 생으로 먹는 요리, 달걀 노른자와 곁들여 제공"
    },
    "cannoli": {
        "칼로리": "320 kcal (100g)",
        "탄수화물": "34 g",
        "단백질": "6 g",
        "지방": "18 g",
        "설명": "시칠리아 전통 디저트, 튀긴 페이스트리 튜브에 달콤한 리코타 크림을 채운 과자"
    },
    "churros": {
        "칼로리": "410 kcal (100g)",
        "탄수화물": "46 g",
        "단백질": "5 g",
        "지방": "24 g",
        "설명": "밀가루 반죽을 튀겨 설탕을 뿌린 스페인 간식, 초콜릿 소스와 함께 즐김"
    }
}

# ---------------------------
# 4. 파일 업로드 UI
# ---------------------------
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"])

# ---------------------------
# 5. 추론
# ---------------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="업로드된 이미지", use_container_width=True)

    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs).logits
        pred = outputs.argmax(1).item()

    result = classes[pred]
    st.success(f"예측 결과: **{result}**")

    # 영양정보 카드 출력
    if result in food_info:
        info = food_info[result]
        st.info(
            f"칼로리: {info['칼로리']}\n"
            f"탄수화물: {info['탄수화물']}\n"
            f"단백질: {info['단백질']}\n"
            f"지방: {info['지방']}\n\n"
            f"설명: {info['설명']}"
        )
    else:
        st.warning("⚠️ 해당 음식의 영양정보는 아직 준비되지 않았습니다.")
