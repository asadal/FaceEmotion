import streamlit as st
import cv2
import numpy as np
from retinaface import RetinaFace
from fer import FER
from PIL import Image
import io

# 감정 분석 모델 로드
emotion_detector = FER(mtcnn=True)  # MTCNN을 사용하여 얼굴을 먼저 잘라냅니다.

# 감정별 색상 사전
emotion_colors = {
    "angry": (0, 0, 255),  # 빨강
    "disgust": (0, 255, 0),  # 초록
    "fear": (255, 0, 0),  # 파랑
    "happy": (255, 255, 0),  # 노랑
    "sad": (255, 0, 255),  # 보라
    "surprise": (0, 255, 255),  # 하늘색
    "neutral": (255, 255, 255)  # 흰색
}

def detect_faces_and_emotions(img):
    img_rgb = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = img_rgb.shape
    
    # 얼굴 인식을 위한 RetinaFace 모델 예측 수행
    faces = RetinaFace.detect_faces(img_rgb)

    if faces:
        for key, face in faces.items():
            facial_area = face["facial_area"]
            x1, y1, x2, y2 = facial_area
            face_img = img_rgb[y1:y2, x1:x2]

            # 얼굴 영역에 네모 표시
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # FER 모델 입력 크기에 맞게 얼굴 이미지 크기 조정
            face_img_resized = cv2.resize(face_img, (48, 48))

            emotions = emotion_detector.detect_emotions(face_img_resized)
            if emotions:
                # 모든 감정을 높은 점수 순서대로 정렬
                sorted_emotions = sorted(emotions[0]["emotions"].items(), key=lambda item: item[1], reverse=True)
                st.write(f"Emotions for face {key}: ", sorted_emotions)
                y_offset = y1
                for emotion, score in sorted_emotions:
                    if score > 0.0:  # 점수가 0보다 큰 감정만 라벨링
                        label = f"{emotion} ({score:.2f})"
                        color = emotion_colors.get(emotion, (36, 255, 12))  # 감정별 색상 선택

                        # 텍스트 박스 높이 계산
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        text_width, text_height = text_size

                        # 텍스트 배경 박스 위치 조정
                        y_label = y_offset - text_height - 10
                        if y_label < 0:
                            y_label = y_offset + text_height + 10
                        if y_label + text_height > img_height:
                            y_label = img_height - text_height - 10

                        if x1 + text_width > img_width:
                            x1 = img_width - text_width - 10

                        # 텍스트 배경 박스 추가
                        cv2.rectangle(img_rgb, (x1, y_label - text_height), (x1 + text_width, y_label), (0, 0, 0), cv2.FILLED)

                        # 텍스트 추가 (박스 안에)
                        cv2.putText(img_rgb, label, (x1, y_label - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        y_offset -= (text_height + 20)
            else:
                st.write(f"No emotions detected for face {key}.")
    else:
        st.warning("No faces detected.")

    return img_rgb

def app():
    st.set_page_config(
        page_title="Emotion Detection",
        page_icon="https://static-00.iconduck.com/assets.00/ios-face-recognition-icon-2048x2048-3kp5zcs2.png"
    )
    # Featured image
    st.image(
        "https://static-00.iconduck.com/assets.00/ios-face-recognition-icon-2048x2048-3kp5zcs2.png",
        width=150
    )
    # Main title and description
    st.title("Emotion Detection from Image")
    st.markdown("Detect faces from uploaded photos and analyse their emotions.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp", "bmp"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Detecting emotions...")
        result_img = detect_faces_and_emotions(image)
        result_pil_img = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        st.image(result_pil_img, caption='Processed Image', use_column_width=True)
        # 결과 이미지를 바이트 배열로 변환
        buf = io.BytesIO()
        result_pil_img.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        # 다운로드 버튼 추가
        st.download_button(
            label="Download Processed Image",
            data=byte_im,
            file_name="processed_image.jpg",
            mime="image/jpeg"
        )

if __name__ == "__main__":
    app()
