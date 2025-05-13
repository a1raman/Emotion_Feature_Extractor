# extract_Audio_Tone.py (25.04.22 kh)
# ------------------------
# 1. Qwen-Audio-Chat으로 오디오 톤 분석
# 2. audio_emotion_predictions.json answer 저장
import os
import subprocess
from glob import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

# ------------------- 모델 및 DIR 설정 -------------------
LOCAL_MODEL_PATH = "/mnt/ssd/sdd/khchoi/checkpoints/models/Qwen-Audio-Chat"

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH, device_map="cuda", trust_remote_code=True).eval()

VIDEO_DIR = "/mnt/ssd/sdd/khchoi/datasets/AIHUB_Kor/faceresult/AIHUB_Video_Backup"
AUDIO_DIR = "./extracted_audio/AIHUB"
os.makedirs(AUDIO_DIR, exist_ok=True)

# ------------------- 톤 분석 프롬프트 -------------------
# emotion_prompt = (
#     "<|im_start|>system\n"
#     "You are a voice emotion expert. Please analyze the speaker's tone from this audio. "
#     "Respond in a sentence like: 'speaks in a [tone] tone.' Possible tones are: joyful, sad, shocked, fearful, angry, positive, negative, calm, doubtful, dismissive.<|im_end|>\n"
#     "<|im_start|>user\n<audio>{audio_path}</audio><|im_end|>\n"
#     "<|im_start|>assistant\n"
# )
emotion_prompt = (
    "<|im_start|>system\n"
    "You are a voice emotion expert. Based on the speaker's tone, classify it into one of:\n"
    "joyful, sad, shocked, fearful, angry, calm, doubtful, dismissive, positive, negative.\n"
    "Do NOT transcribe or summarize the words. Only analyze the vocal tone.\n\n"
    "Example 1: speaks in a joyful tone.\n"
    "Example 2: speaks in a fearful tone.\n"
    "Example 3: speaks in an angry tone.\n"
    "Example 4: speaks in a calm tone.\n\n"
    "Now analyze the following:<|im_end|>\n"
    "<|im_start|>user\n<audio>{audio_path}</audio><|im_end|>\n"
    "<|im_start|>assistant\n"
)



# ------------------- 오디오 있는지 확인 -------------------
def has_audio_stream(video_path):
    cmd = [
        "ffprobe", "-i", video_path,
        "-show_streams", "-select_streams", "a",
        "-loglevel", "error"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return bool(result.stdout.strip())

# ------------------- 결과 문장만 추출 -------------------
# def extract_assistant_response(text):
#     if "<|im_start|>assistant" in text:
#         return text.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
#     return text.strip()
def extract_assistant_response(text):
    if "<|im_start|>assistant" in text:
        result = text.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    else:
        result = text.strip()

    # 강제 톤 필터링
    tones = ["joyful", "sad", "shocked", "fearful", "angry", "positive", "negative", "calm", "doubtful", "dismissive"]
    for tone in tones:
        if tone in result.lower():
            return f"speaks in a {tone} tone."

    return "speaks in a neutral tone."

# ------------------- 결과 저장 -------------------
audio_results = {}

video_paths = sorted(
    glob(os.path.join(VIDEO_DIR, "*.mp4")) +
    glob(os.path.join(VIDEO_DIR, "*.MP4")) +
    glob(os.path.join(VIDEO_DIR, "*.mkv")) +
    glob(os.path.join(VIDEO_DIR, "*.avi"))
)

for video_path in video_paths:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(AUDIO_DIR, f"{video_name}.flac")

    if not has_audio_stream(video_path):
        print(f"[{video_name}] 오디오 없음 - 스킵")
        continue

    # ---- ffmpeg로 오디오 추출 ----
    command = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "flac", "-ar", "16000", "-ac", "1",
        audio_path, "-y"
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    #---- 프롬프트 생성 및 모델 추론 ----
    query = emotion_prompt.format(audio_path=audio_path)
    audio_info = tokenizer.process_audio(query)
    inputs = tokenizer(query, return_tensors='pt', audio_info=audio_info).to(model.device)

    with torch.no_grad():
        pred = model.generate(**inputs, audio_info=audio_info, max_new_tokens=128)
        full_response = tokenizer.decode(pred[0].cpu(), skip_special_tokens=False, audio_info=audio_info)
        answer = extract_assistant_response(full_response)

    audio_results[video_name] = answer
    print(f"[{video_name}] {answer}")

# ------------------- json 저장장 -------------------
with open("audio_emotion_predictions.json", "w") as f:
    json.dump(audio_results, f, indent=4)

