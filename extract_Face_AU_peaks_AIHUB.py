import os
import cv2
import json
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from feat import Detector
import pandas as pd
import torch
import multiprocessing as mp
import random
import sys
import shutil
import warnings
import logging

from au_config import AU_DESCRIPTION_DICT

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger().setLevel(logging.ERROR)

# ------------------- DIR 설정 -------------------
VIDEO_DIR = "/mnt/ssd/sdd/khchoi/datasets/AIHUB_Kor/faceresult/AIHUB_Video"
VIDEO_BACKUP_DIR = "/mnt/ssd/sdd/khchoi/datasets/AIHUB_Kor/faceresult/AIHUB_Video_Backup"
SAVE_FACE_DIR = "/mnt/ssd/sdd/khchoi/datasets/AIHUB_Kor/faceresult/dataset-process/pyfeat_face/crop_images"
PEAK_AU_DIR = os.path.join(SAVE_FACE_DIR, "../peak_au_frames")
TEMP_IMAGE_DIR = "./saved_frames"
OUTPUT_TXT_PATH = os.path.join(SAVE_FACE_DIR, "../MERR_coarse_grained_kh.txt")
OUTPUT_JSON_PATH = os.path.join(SAVE_FACE_DIR, "../MERR_coarse_grained_kh.json")
OUTPUT_SENTENCE_CSV_PATH = os.path.join(SAVE_FACE_DIR, "../sentence_kh.csv")
AIHUB_JSON_DIR = "/mnt/ssd/sdd/khchoi/datasets/AIHUB_Kor/faceresult/AIHUB_Json"
AUDIO_TONE_PATH = "/Qwen-Audio/audio_emotion_predictions.json"

for path in [SAVE_FACE_DIR, PEAK_AU_DIR, TEMP_IMAGE_DIR, VIDEO_BACKUP_DIR]:
    os.makedirs(path, exist_ok=True)

with open(AUDIO_TONE_PATH, "r") as f:
    audio_tone_dict = json.load(f)

aihub_json_map = {}
for json_file in glob(os.path.join(AIHUB_JSON_DIR, "*.json")):
    with open(json_file, "r") as f:
        data = json.load(f)
        video_key = os.path.splitext(data["video"])[0]
        aihub_json_map[video_key] = {
            "utterance": data.get("utterance", ""),
            "emotion": data.get("emotion", "neutral")
        }

def extract_frames(video_path, save_dir, num_sampled_frames=16):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num=num_sampled_frames, dtype=int)
    frame_paths = []

    print(f"[{os.path.basename(video_path)}] 프레임 저장 시작...")
    for idx in tqdm(frame_indices, desc=f"Saving frames ({os.path.basename(video_path)})", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            save_path = os.path.join(save_dir, f"frame_{idx}.jpg")
            Image.fromarray(frame_rgb).save(save_path)
            frame_paths.append(save_path)
    cap.release()
    print(f"[{os.path.basename(video_path)}] 프레임 저장 완료 ✅")
    return frame_paths, total_frames

def process_saved_frames(video_name, frame_paths, total_frames):
    try:
        detector = Detector(device="cuda")
        results = detector.detect_image(frame_paths, batch_size=16, data_type="image")
        if results is None or len(results) == 0:
            print(f"[ERROR] 얼굴 미검출: {video_name}")
            return None, video_name

        df = results
        if df.empty:
            print(f"[ERROR] 결과 없음: {video_name}")
            return None, video_name

        au_columns = [col for col in df.columns if col.startswith("AU")]
        AU_sums = df[au_columns].sum(axis=1)
        if AU_sums.isnull().all():
            print(f"[ERROR] {video_name}: AU_sums 모두 NaN")
            return None, video_name

        peak_idx = AU_sums.idxmax()
        peak_index = int(df.loc[peak_idx, "frame"])
        peak_AU_list = [col for col in au_columns if df.loc[peak_idx, col] > 0.5]

        AU_list = [col for col in peak_AU_list if col in AU_DESCRIPTION_DICT]
        visual_prior_list = [random.choice(AU_DESCRIPTION_DICT[au]) for au in AU_list]
        audio_prior = audio_tone_dict.get(video_name, "speaks in a normal tone.")
        caption = ", ".join(visual_prior_list + [audio_prior])

        sampled_frames = [Image.open(p) for p in frame_paths]
        cropped_images = []

        most_common_id = df["Identity"].value_counts().idxmax()
        for di in range(len(df)):
            det = df.loc[di]
            cur_frame_num = det["frame"]
            if det["Identity"] == most_common_id:
                tx, ty = int(det.faceboxes["FaceRectX"]), int(det.faceboxes["FaceRectY"])
                w, h = int(det.faceboxes["FaceRectWidth"]), int(det.faceboxes["FaceRectHeight"])
                img_w, img_h = sampled_frames[cur_frame_num].size
                confidence = det.faceboxes.get("confidence", 1.0)

                if w < img_w * 0.1 or h < img_h * 0.1 or confidence < 0.95:
                    continue

                tx = max(0, min(tx, img_w - 1))
                ty = max(0, min(ty, img_h - 1))
                w = min(w, img_w - tx)
                h = min(h, img_h - ty)

                face_img_np = np.array(sampled_frames[cur_frame_num])[ty:ty+h, tx:tx+w]
                face_img = Image.fromarray(face_img_np)
                cropped_images.append((cur_frame_num, face_img))

        if not cropped_images:
            print(f"[ERROR] {video_name}: 얼굴 crop 결과 없음")
            return None, video_name

        # crop 성공했을 때만 peak 저장
        peak_img = sampled_frames[peak_index]
        peak_img.save(os.path.join(PEAK_AU_DIR, f"{video_name}_peak.png"))

        cropped_images.sort(key=lambda x: x[0])
        video_folder = os.path.join(SAVE_FACE_DIR, f"{video_name}")
        os.makedirs(video_folder, exist_ok=True)
        for i, (_, img) in enumerate(cropped_images):
            img.save(os.path.join(video_folder, f"frame_{i}.png"))
        while len(cropped_images) < 16:
            cropped_images.append((len(cropped_images), cropped_images[0][1]))
            cropped_images[-1][1].save(os.path.join(video_folder, f"frame_{len(cropped_images)-1}.png"))

        torch.cuda.empty_cache()
        utterance = aihub_json_map.get(video_name, {}).get("utterance", "")
        pseu_emotion = aihub_json_map.get(video_name, {}).get("emotion", "neutral")
        return (video_name, total_frames, AU_list, visual_prior_list, audio_prior, peak_index, peak_AU_list, caption, utterance, pseu_emotion), None

    except Exception as e:
        print(f"[ERROR] {video_name}: {e}")
        return None, video_name

def main():
    mp.set_start_method("spawn", force=True)
    video_paths = sorted(glob(os.path.join(VIDEO_DIR, "*.mp4")) + glob(os.path.join(VIDEO_DIR, "*.avi")))
    #video_paths = video_paths[:10]

    # 미리 backup copy
    for video_path in video_paths:
        shutil.copy(video_path, VIDEO_BACKUP_DIR)

    all_tasks = []
    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        temp_dir = os.path.join(TEMP_IMAGE_DIR, video_name)
        frame_paths, total_frames = extract_frames(video_path, temp_dir)
        all_tasks.append((video_name, frame_paths, total_frames))

    pool = mp.Pool(processes=4)
    try:
        raw_results = list(tqdm(pool.starmap(process_saved_frames, all_tasks), total=len(all_tasks)))
    finally:
        pool.close()
        pool.join()

    results = []
    failed_videos = []

    for result, failed_video in raw_results:
        if failed_video is not None:
            failed_videos.append(failed_video)
        elif result is not None:
            results.append(result)

    merr_txt_lines = []
    merr_json_obj = {}
    sentence_list = []

    for video_name, total_frames, AU_list, visual_prior_list, audio_prior, peak_index, peak_AU_list, caption, utterance, pseu_emotion in results:
        merr_txt_lines.append(f"{video_name} {total_frames} {pseu_emotion}")
        merr_json_obj[video_name] = {
            "AU_list": AU_list,
            "visual_prior_list": visual_prior_list,
            "audio_prior_list": audio_prior,
            "peak_index": str(peak_index),
            "peak_AU_list": peak_AU_list,
            "pseu_emotion": pseu_emotion,
            "caption": caption
        }
        sentence_list.append({"name": video_name, "sentence": utterance})

    with open(OUTPUT_TXT_PATH, "w") as f:
        f.write("\n".join(merr_txt_lines))

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(merr_json_obj, f, indent=4, ensure_ascii=False)

    pd.DataFrame(sentence_list).to_csv(OUTPUT_SENTENCE_CSV_PATH, index=False, encoding="utf-8-sig")

    if failed_videos:
        for failed_video in failed_videos:
            backup_video_path = os.path.join(VIDEO_BACKUP_DIR, f"{failed_video}.mp4")
            if os.path.exists(backup_video_path):
                os.remove(backup_video_path)
                print(f"[Deleted] {failed_video}.mp4 삭제 완료")
    else:
        print("[INFO] 실패한 비디오 없음. 삭제 스킵.")

if __name__ == "__main__":
    main()
