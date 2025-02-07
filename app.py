import os
import streamlit as st
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image, ImageDraw, ImageFont

# APIキーとエンドポイントを環境変数から取得
subscription_key = "FsRxa7VqCB8vvKsd8F9wHuocRt3uSGl2mD9BUSEoBU0ipyxWnrrqJQQJ99BBACi0881XJ3w3AAAFACOGyMbo"
endpoint = "https://20250203.cognitiveservices.azure.com/"

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# 画像のタグを取得する関数
def get_tags(filepath):
    print("===== Tag an image - local =====")
    
    with open(filepath, "rb") as local_image:
        tags_result_local = computervision_client.tag_image_in_stream(local_image)
    
    if len(tags_result_local.tags) == 0:
        return "No tags detected."
    else:
        return ", ".join(tag.name for tag in tags_result_local.tags)

# 画像の物体検出を行う関数
def extract_bounding_boxes(filepath):
    print("===== Object Detection - local =====")
    
    with open(filepath, "rb") as local_image:
        detect_objects_results = computervision_client.analyze_image_in_stream(local_image, visual_features=["objects"])
    
    bounding_boxes = []
    if not detect_objects_results.objects:
        print("No objects detected.")
    else:
        for obj in detect_objects_results.objects:
            rect = obj.rectangle
            obj_data = {
                "object": obj.object_property,
                "confidence": obj.confidence * 100,
                "bounding_box": (rect.x, rect.y, rect.w, rect.h)
            }
            bounding_boxes.append(obj_data)
            print("Object: '{}', Confidence: {:.2f}%, Bounding box: ({}, {}, {}, {})".format(
                obj.object_property, obj.confidence * 100, rect.x, rect.y, rect.w, rect.h
            ))
    return bounding_boxes

# StreamlitのUI
st.title('物体検出アプリ')

uploaded_file = st.file_uploader('Choose an image...', type=['jpeg', 'png'])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    
    # 画像処理用の一時ファイル
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 物体検出の処理
    objects = extract_bounding_boxes("uploaded_image.jpg")
    
    # 画像に矩形とラベルを描画
    draw = ImageDraw.Draw(img)
    for obj in objects:
        x, y, w, h = obj["bounding_box"]
        caption = obj["object"]
        
        # デフォルトフォント（Cloudにデフォルトで存在）
        font = ImageFont.load_default()
        
        # 矩形とテキストを描画
        draw.rectangle([(x, y), (x + w, y + h)], outline="green", width=5)
        bbox = draw.textbbox((x, y), caption, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([(x, y), (x + text_w, y + text_h)], fill='green')
        draw.text((x, y), caption, fill='white', font=font)
    
    st.image(img, caption="Processed Image", use_container_width=True)

    # 画像タグの取得
    tags_name = get_tags("uploaded_image.jpg")
    st.markdown('**認識されたコンテンツタグ**')
    st.markdown(f'>{tags_name}')

