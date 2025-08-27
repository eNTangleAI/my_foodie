import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import json

# ---------------------------
# 제목 & 안내
# ---------------------------
st.title("🍱 음식 이미지 분류기 (ViT-B16 Demo)")

st.markdown("""
⚠️ **데모 버전 안내**  
현재 분류기는 Food-50 데이터셋 기반으로 학습된 ViT-B16 모델을 사용합니다.  
업로드한 이미지를 분류하여 해당 음식의 정보 카드(칼로리, 페어링, 음악/영화 추천)를 표시합니다.
""")

# ---------------------------
# JSON 불러오기
# ---------------------------
with open("food_info.json", "r", encoding="utf-8") as f:
    food_info = json.load(f)

classes = list(food_info.keys())

# ---------------------------
# 모델 불러오기
# ---------------------------
@st.cache_resource
def load_model():
    model = torch.hub.load('huggingface/pytorch-transformers', 'vit_b16', pretrained=False)
    model.head = torch.nn.Linear(model.head.in_features, len(classes))
    model.load_state_dict(torch.load("vit_best.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ---------------------------
# 파일 업로드
# ---------------------------
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="업로드된 이미지", use_container_width=True)

    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        pred = model(x).argmax(1).item()

    result = classes[pred]

    st.success(f"예측 결과: **{result}**")

    # ---------------------------
    # 카드 출력
    # ---------------------------
    info = food_info.get(result, None)
    if info:
        with st.expander("🍽️ 음식 카드 펼쳐보기"):
            st.markdown(f"""
            **칼로리:** {info['칼로리']}  
            **주요 영양소:** {info['주요 영양소']}  
            **설명:** {info['설명']}  

            🥂 **추천 페어링:** {", ".join(info['추천 페어링'])}  
            🎵 **추천 음악:** {info['추천 음악']}  
            🎬 **추천 영화:** {info['추천 영화']}  
            """)
    else:
        st.warning("해당 음식의 추가 정보가 없습니다.")
