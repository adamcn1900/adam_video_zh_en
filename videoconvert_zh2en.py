import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import subprocess
from transformers import MarianMTModel, MarianTokenizer
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from pydub import AudioSegment
import soundfile as sf

# 原始视频文件名称
ORI_VIDEO_FILE="01_done_20241130_3.mp4"
ORI_AUDIO_FILE="./output/ori_audio_file.wav"

# Spleeter分离文件名称
SPL_AUDIO_BGM='./output/ori_audio_file/accompaniment.wav'
SPL_AUDIO_HUMANVOC='./output/ori_audio_file/vocals.wav'

# ASR whisper 输出目录文件
ORI_ASR_TEXT_FOLDER="./output/ori_asr_text"
ORI_ASR_TEXT_FILE="./output/ori_asr_text/vocals.srt"

# LLM翻译脚本结果
TAR_TRANSLATE_RESULT_TEXT_FILE="./output/result_lang_en.srt"
TAR_TRANSLATE_RESULT_AUDIO_FILE="./output/result_lang_en.wav"

# final
FIN_VIDEO="./output/fin_lang_en.mp4"

# 提取wav
def extract_ori_wav(ori_mov,tar_aud):
    video_path = ori_mov
    output_audio_path = tar_aud

    ffmpeg_command = [
        "ffmpeg",
        "-i", video_path,  # 输入视频文件
        "-vn",              # 禁用视频流，仅提取音频
        "-acodec", "pcm_s16le",  # 设置音频编解码器为 PCM 16位小端格式
        "-ar", "16000",     # 设置音频采样率为 16kHz
        output_audio_path        # 输出音频文件
    ]
    print('提取原始音频文件...')
    subprocess.run(ffmpeg_command, check=True)
    print(f"输出视频已保存为: {output_audio_path}")

# 分离人声
def seperate_humanvocandbgm(ori_aud):
    # 定义Spleeter的命令
    command = [
        'spleeter', 'separate', ori_aud, 
        '-o', 'output/', 
        '-p', 'spleeter:2stems'
    ]

    # 执行命令
    subprocess.run(command, check=True)

    print("分离完成")

# 原始人声转脚本
def extract_ori_wav_srt(ori_aud,whisper_model_type,outputfile):
    model_dir="./models/whisper_small/"
    language = "Chinese"
    output_format = "srt"

    whisper_command = [
        "whisper", ori_aud,
        "--model", whisper_model_type,
        "--language", language,
        "--output_format", output_format,
        "--device", "cuda",
        "--model_dir", model_dir,
        "--output_dir", outputfile
    ]

    try:
        print("音频提取字幕srt...")
        subprocess.run(whisper_command, check=True)
        print(f"字幕文件已生成，基于音频文件 {outputfile}")
    except subprocess.CalledProcessError as e:
        print(f"生成字幕时出错: {e}")

# 根据人声脚本进行翻译
def translate_zh2en(en_text,zh_text):
    # 加载 MarianMT 中文到英语模型
    # model_name = "Helsinki-NLP/opus-mt-zh-en"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备: {device}")
    print("开始加载 NLP-opus-mt-zh-en 预训练模型...")
    model_name = "./models/zh_en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)

    with open(en_text, "r", encoding="utf-8") as file:
        chinese_text = file.readlines()

    chinese_translations = []
    for line in tqdm(chinese_text, desc="翻译进度", unit="行"):
        if line.strip().isdigit():
            chinese_translations.append(line)
        elif '-->' in line:
            chinese_translations.append(line)
        elif line.strip() == "":
            chinese_translations.append(line)
        else:  # 字幕脚本部分
            inputs = tokenizer(line, return_tensors="pt", padding=True, truncation=True).to(device)
            translated = model.generate(**inputs)
            chinese_translations.append(tokenizer.decode(translated[0], skip_special_tokens=True)+"\n")

    with open(zh_text, "w", encoding="utf-8") as output:
        output.writelines(chinese_translations)

# 合成英文人声
def tts_en(input_text,output_aud_file):
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"检查当前设备：{device}")

    # 加载预训练模型 微软speecht5_tts
    processor = SpeechT5Processor.from_pretrained("./models/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("./models/speecht5_tts").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("./models/speecht5_hifigan").to(device)

    # 从文件读取文本
    with open(input_text, "r", encoding="utf-8") as file:
        en_text = file.readlines()
    # 只翻译字幕部分，其他部分跳过
    clear_en_text=[]
    for line in en_text:
        if line.strip().isdigit():
            continue
        elif '-->' in line:
            continue
        elif line.strip() == "":
            continue
        else:  
            clear_en_text.append(line)

    # 加载说话人嵌入
    xvector_path = "./data/spkrec_xvect/cmu_us_rms_arctic-wav-arctic_a0440.npy"
    xvector = np.load(xvector_path)
    speaker_embeddings = torch.tensor(xvector,device=device).unsqueeze(0).repeat(len(en_text), 1)

    # 将文本处理为模型输入,并放入GPU处理
    print('正在处理文本输入...')
    inputs = processor(text=clear_en_text, return_tensors="pt", padding=True).to(device)

    # 批量生成语音
    # 转回 CPU 以便后续处理
    print("正在生成语音文件...")
    speech_list = []
    for i in tqdm(range(len(clear_en_text)), desc="语音生成进度", unit="段"):
        speech = model.generate_speech(inputs["input_ids"][i].unsqueeze(0), speaker_embeddings[i].unsqueeze(0), vocoder=vocoder)
        speech_list.append(speech.cpu()) 

    # 遍历生成的语音，合并临时文件
    combined_audio = AudioSegment.silent(duration=0) 
    temp_file=""
    for i, speech in enumerate(tqdm(speech_list,desc="合并处理进度",unit="段")):
        temp_file = f"temp_speech_{i}.wav"
        sf.write(temp_file, speech.numpy(), samplerate=16000)  # 保存为临时文件
        audio = AudioSegment.from_wav(temp_file)  # 读取音频
        combined_audio +=audio

        if os.path.exists(temp_file):
            os.remove(temp_file) 

    print("正在导出合并的音频...")
    combined_audio.export(output_aud_file, format="wav")
    print(f"合并后的音频文件已保存到：{output_aud_file}")  

def combine_voice(ori_video,fin_video):
    # 输入文件路径
    generated_audio = TAR_TRANSLATE_RESULT_AUDIO_FILE
    video_without_audio = "01_done_20241130_3_video_only.mp4" # deleted
    fin_video_without_bgm="01_done_20241130_3_video_lang.mp4"

    # 提取无音轨视频
    subprocess.run([
        "ffmpeg", "-i", ori_video, "-an", "-c:v", "copy", video_without_audio
    ])

    # 1.合并生成音频和无音轨视频
    subprocess.run([
        "ffmpeg", "-i", video_without_audio, "-i", generated_audio,
        "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", fin_video_without_bgm
    ])

    # 2.添加背景音乐
    command = [
            "ffmpeg",
            "-i", fin_video_without_bgm,             # 输入视频
            "-i", SPL_AUDIO_BGM,             # 输入背景音频
            "-filter_complex", "[0:a:0][1:a:0]amix=inputs=2:duration=shortest",
            "-c:v", "copy",               # 保留视频编码
            "-c:a", "aac",                # 音频编码为AAC
            fin_video                   # 输出路径
        ]
    
    subprocess.run(command, check=True)

    # 清理中间文件（可选）
    if os.path.exists(video_without_audio):
        os.remove(video_without_audio)

    if os.path.exists(fin_video_without_bgm):
        os.remove(fin_video_without_bgm)

    print(f"视频生成完成，文件路径：{fin_video}")

if __name__=="__main__":

    # parser = argparse.ArgumentParser(description="video translate program")

    # # 添加参数
    # parser.add_argument('-i', '--input', type=str, required=True, help="输入文件路径")
    # parser.add_argument('-o', '--output', type=str, required=True, help="输出文件路径")
    # parser.add_argument('-v', '--verbose', action='store_true', help="是否输出详细信息")

    # # 解析命令行参数
    # args = parser.parse_args()

    # # 打印解析后的参数
    # print(f"输入文件路径: {args.input}")
    # print(f"输出文件路径: {args.output}")
    # if args.verbose:
    #     print("启用了详细模式")

    # 提取原始音频
    extract_ori_wav(ORI_VIDEO_FILE,ORI_AUDIO_FILE)
    # 分离音频
    seperate_humanvocandbgm(ORI_AUDIO_FILE)
    
    # 分离音频人声ASR
    extract_ori_wav_srt(SPL_AUDIO_HUMANVOC,"base",ORI_ASR_TEXT_FOLDER)
    # ZH_EN文本翻译
    translate_zh2en(ORI_ASR_TEXT_FILE,TAR_TRANSLATE_RESULT_TEXT_FILE)

    # TTS 模式生成
    tts_en(TAR_TRANSLATE_RESULT_TEXT_FILE,TAR_TRANSLATE_RESULT_AUDIO_FILE)

    # 合并视频
    combine_voice(ORI_VIDEO_FILE,FIN_VIDEO)