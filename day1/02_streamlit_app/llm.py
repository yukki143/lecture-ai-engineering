# llm.py
import os
import torch
from transformers import pipeline
import streamlit as st
import time
from config import MODEL_NAME
from huggingface_hub import login

# モデルをキャッシュして再利用
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:
        hf_token = st.secrets["huggingface"]["token"]
        
        device = 0 if torch.cuda.is_available() else -1
        st.info(f"Using device: {'cuda' if device == 0 else 'cpu'}")  # 使用デバイスを表示
        
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            device=device,
            model_kwargs={"torch_dtype": torch.bfloat16 if device == 0 else torch.float32},
        )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None

def generate_response(pipe, user_question):
    """LLMを使用して質問に対する回答を生成する"""
    if pipe is None:
        return "モデルがロードされていないため、回答を生成できません。", 0

    try:
        start_time = time.time()
        
        # --- ここが重要！ ---
        outputs = pipe(
            user_question,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        # 出力の取り出し
        full_text = outputs[0]["generated_text"]

        # ユーザー質問以降の部分だけ切り出し
        if user_question in full_text:
            assistant_response = full_text.split(user_question, 1)[-1].strip()
        else:
            assistant_response = full_text.strip()

        end_time = time.time()
        response_time = end_time - start_time
        print(f"Generated response in {response_time:.2f}s")  # デバッグ用
        return assistant_response, response_time

    except Exception as e:
        st.error(f"回答生成中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return f"エラーが発生しました: {str(e)}", 0
