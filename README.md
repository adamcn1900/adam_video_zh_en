# adam_video_zh_en
translate video from zh to en

This is a SHOWCASE example program that converts video files into English. 
The pipeline for the video translation task includes the following steps: 
1) First, extract the audio from the video, a process that utilizes ffmpeg.
2) Use spleeter to separate the human voice from the audio, *I think this will improve the accuracy of downstream ASR.
3) Employ the Whisper encoder-decoder model for ASR voice recognition and generate an SRT subtitle file, *in the example, the "base" model is used.
4) Translate the SRT file, *using the Helsinki-NLP/opus-mt-zh-en model for Chinese to English translation processing.
5) After translation, use speecht5_tts for voice generation.
6) Finally, merge the results from the upstream processing.
  
Main purpose: To demonstrate the end-to-end process of a video translation task. 
Optimization space: 
1) More refinement of video voice processing, such as noise reduction, to improve audio clarity.
2) Utilize larger ASR models to enhance recognition accuracy.
3) Using an SFT model to perform preliminary proofreading on ASR results to correct obvious errors.
4) Employ better models to improve the precision of Chinese-English translation.
