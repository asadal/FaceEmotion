import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from fer import FER
from PIL import Image
import io

# YOLOv8 모델 로드 (얼굴 탐지 모델 사용)
model = YOLO("yolov8n.pt")

# 감정 분석 모델 로드
emotion_detector = FER()

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

    # 얼굴 인식을 위한 YOLOv8 모델 예측 수행
    results = model(img_rgb)

    # 결과에서 얼굴만 추출 (모든 객체의 클래스를 검사하여 person(인간) 클래스만 추출)
    faces = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])  # 클래스 ID는 boxes의 cls 속성에 있음
            if cls_id == 0:  # COCO 데이터셋에서 person 클래스의 인덱스는 0입니다.
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
                face = img_rgb[y1:y2, x1:x2]
                faces.append((face, (x1, y1, x2, y2)))

    # 감정 분석 및 라벨링
    for face, (x1, y1, x2, y2) in faces:
        emotions = emotion_detector.detect_emotions(face)
        if emotions:
            # 얼굴 영역에 네모 표시
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 모든 감정을 높은 점수 순서대로 정렬
            sorted_emotions = sorted(emotions[0]["emotions"].items(), key=lambda item: item[1], reverse=True)
            for emotion, score in sorted_emotions:
                if score > 0.0:  # 점수가 0보다 큰 감정만 라벨링
                    label = f"{emotion} ({score:.2f})"
                    color = emotion_colors.get(emotion, (36, 255, 12))  # 감정별 색상 선택

                    # 텍스트 박스 높이 계산
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    text_width, text_height = text_size

                    # 텍스트 배경 박스 위치 조정
                    y_label = y1 - text_height - 10 if y1 - text_height - 10 > 0 else y1 + text_height + 10

                    # 텍스트 배경 박스 추가
                    cv2.rectangle(img_rgb, (x1, y_label - text_height), (x1 + text_width, y_label), (0, 0, 0), cv2.FILLED)
                    
                    # 텍스트 추가 (박스 안에)
                    cv2.putText(img_rgb, label, (x1, y_label - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y1 -= (text_height + 20)

    return img_rgb

st.title("Emotion Detection from Image")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

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
