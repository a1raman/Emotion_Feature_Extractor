# feature_extract_main.py (15.04.22 kh)
# 그냥 합쳐놓은거임
import subprocess

print("1. Extracting audio tone...")
subprocess.run(["python", "/Qwen-Audio/extract_Audio_Tone.py"])

print("2. Extracting face + AU features...")
subprocess.run(["python", "extract_Face_AU_peaks_AIHUB.py"])

print("Feature extraction completed.")


