# Emotion_Feature_Extractor  
Emotion-LLaMA Feature_Extractor   

## 1. Face Crop/AU/dataset Extract  
   ### python feature_extract_main_AIHUB.py   
     - extract_Face_AU_peaks_AIHUB.py  
     - extract_Audio_Tone.py (QWEN-AUDIO)  
## 2. Feature Extract for Emotion-LLaMA train  
   ### VideoMAE 
   ```
   python -u extract_maeVideo_embedding.py \
  --dataset='MER2023' \
  --feature_level='UTTERANCE' \
  --device='cuda:0' \
  --pretrain_model='/path/your/models/maeVideo_ckp199' \
  --feature_name='maeVideo'    
  ```
   ### MAE  
   ```
   python -u extract_mae_embedding.py \
  --dataset='MER2023' \
  --feature_level='UTTERANCE' \
  --device='cuda:3' \
  --pretrain_model='/path/your/models/mae_checkpoint-340' \
  --feature_name='mae_checkpoint-340'  
  ```
   ### Hubert  
   ```
   python main-baseline.py split_audio_from_video_16k \
  '/path/your/video/AIHUB_Video_Backup' \
  '/path/your/outputs/audio'  
   ```
   ```
   python -u extract_transformers_embedding.py \
  --dataset='MER2023' \
  --feature_level='UTTERANCE' \
  --model_name='/path/your/models/hubert-large-korean' \
  --gpu=3
   ```


   
