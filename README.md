# JLPT-listening-practice-video-generation

## Get-Starting
This repo includes code used to generate conversation videos in Japanese, with words, grammar structures categorized by JLPT level from N5-N4-N3. This is a creative application created through `機械知能学特論 course s1-2026` in IPU by student Nguyen Duc Anh (2312026009)

The motivation is to help me improve japanese since I am an international student while studing in Japan. I want to learn japanese to communicate with sensei, tomotachi,...!!!

In this project I use some generative models including:
- VLM qwen 3.6
- TTS model omnivoice
- Diffusion model qwen-image-2512

## Installation

1. setup LLM by ollama/llama.cpp
For this project, I use `Qwen3.6-35B-A3B-MXFP4_MOE.gguf` (unsloth version) to generate video scriptions run on llama.cpp 
The model is still running on my server on http://172.16.125.200:33040/ (You can try if you can access IPU LAN ) (#upate 2026-06-09)

2. All my code only uses python, so to install and run the code, you just need to run `pip install -r requirement.txt`
besides, to run some AI models such as omivoice (text-to-speech model), qwen-image-251 (image generation model), I use `GPU RTX 3090 24Gb`

3. Update environment value as .env.example

4. (optinal) you can run scripts in `playground` folder to test your system/environment first

5. Then follow steps  as
```bash
# assume that, you have already set up the environment and installed dependencies as per README.md
# ingredients/lecture_bank.json should be ready with JLPT grammar and vocab data

#step1: generate dialogue transcripts using 1.dialogue_transcript_gen.py -n <num_scripts_per_grammar_point>

python 1.dialogue_transcript_gen.py -n 1 # after output 1, you could cancel the process to avoid code generate full dialog for all grammars in the lecture bank of CASTUDY.vn

#step2: generate visualization for each dialogue using 2.dialogue_visualization_gen.py
python 2.dialogue_visualization_gen.py

#step3: generate videos using 3.dialogue_video_gen.py
python 3.dialogue_video_gen.py
```

## Output example
- After step1,  you can see the scripe will generate dialogue for each grammar as [script](./outputs/jpc_scripts/N5/69730939758695870a8ca994/script_001.json) as:
```json
(qwen3-tts) ducanh@OMEN30L-RTX3090:~/JLPT-listening-practice-video-generation$ python 1.dialogue_transcript_gen.py -n 1
Generating 1 scripts for Grammar: N5 - Particles (N5)
{
    "level": "N5",
    "grammar_id": "particles",
    "grammar_name": "N5 - Particles",
    "core_words": ["亀", "青空", "アプリ", "休みます", "箱", "カレーライス", "湿った", "着ます", "連続", "駄目です"],
    "context": "Two friends, Kenji and Yumi, are talking at a park bench. Kenji is worried about a turtle he was carrying in a box because the sky is clear and sunny, making the box wet inside. He asks for advice on what to do.",
    "context_vi": "Hai bạn, Kenji và Yumi, đang trò chuyện trên một chiếc ghế dài công viên. Kenji lo lắng về chú rùa mà anh ấy đang mang trong một cái hộp vì bầu trời xanh và nắng khiến bên trong hộp trở nên ẩm ướt. Anh ấy hỏi bạn bè lời khuyên là phải làm gì.",
    "characters": [
        {
            "name": "Kenji",
            "gender": "Male",
            "profession": "Student"
        },
        {
            "name": "Yumi",
            "gender": "Female",
            "profession": "Student"
        }
    ],
    "scripts": [
        {
            "char": "Kenji",
            "text": "ねえ、この亀、どうしよう。箱の中で湿った。",
            "en_translation": "Hey, what should I do with this turtle? It's wet inside the box.",
            "vi_translation": "Chào, chú rùa này phải làm sao nhỉ? Nó đang ướt bên trong hộp."
        },
        {
            "char": "Yumi",
            "text": "どうして？青空の下にいるから？",
            "en_translation": "Why? Because it's under the blue sky?",
            "vi_translation": "Tại sao thế? Vì nó đang ở dưới bầu trời xanh sao?"
        },
        {
            "char": "Kenji",
            "text": "うん、連続で太陽が照りつけて。",
            "en_translation": "Yes, the sun has been shining continuously.",
            "vi_translation": "Ừ, mặt trời đang chiếu liên tục."
        },
        {
            "char": "Yumi",
            "text": "それは駄目です。すぐに休んでください。",
            "en_translation": "That's not good. Please take a break right away.",
            "vi_translation": "Đó không tốt đâu. Xin hãy nghỉ ngơi ngay lập tức."
        },
        {
            "char": "Kenji",
            "text": "えっ、でも、私は今日、仕事着ます。",
            "en_translation": "Eh, but I have to wear my work clothes today.",
            "vi_translation": "Ồ, nhưng hôm nay tôi phải mặc đồng phục làm việc."
        },
        {
            "char": "Yumi",
            "text": "違います。亀を優先してください。",
            "en_translation": "No, please prioritize the turtle.",
            "vi_translation": "Không phải vậy. Hãy ưu tiên chú rùa đi."
        },
        {
            "char": "Kenji",
            "text": "あ、そうか。アプリで場所を探します。",
            "en_translation": "Oh, I see. I'll look for a place using an app.",
            "vi_translation": "À, tôi hiểu rồi. Tôi sẽ tìm một nơi bằng ứng dụng."
        },
        {
            "char": "Yumi",
            "text": "いいですね。でも、カレーライスは食べませんか？",
            "en_translation": "That's great. But, how about eating curry rice?",
            "vi_translation": "Thật tuyệt. Nhưng, làm sao bạn không ăn cơm cà ri nhỉ?"
        },
        {
            "char": "Kenji",
            "text": "いいえ、今は食べられません。亀が大切です。",
            "en_translation": "No, I can't eat now. The turtle is important.",
            "vi_translation": "Không, hiện giờ tôi không thể ăn được. Chú rùa quan trọng lắm."
        },
        {
            "char": "Yumi",
            "text": "そうですか。でも、箱は新しい箱に変えてください。",
            "en_translation": "I see. But please change the box to a new one.",
            "vi_translation": "Tôi hiểu. Nhưng hãy đổi hộp thành một cái hộp mới đi."
        },
        {
            "char": "Kenji",
            "text": "わかった。ありがとう。これで安心です。",
            "en_translation": "Got it. Thank you. I feel relieved now.",
            "vi_translation": "Tôi hiểu rồi. Cảm ơn bạn. Bây giờ tôi cảm thấy an tâm."
        }
    ]
}
//[1/1] Saved script to outputs/jpc_scripts/N5/69730939758695870a8ca994/script_001.json
//Generating 1 scripts for Grammar: N5 - こ / そ / あ / ど  (N5)

```

2. The output of script 2 should be like this:
```bash
(qwen3-tts) ducanh@OMEN30L-RTX3090:~/JLPT-listening-practice-video-generation$ python 2.dialogue_visual_gen.py
Keyword arguments {'use_auth_token': 'xxxxx'} are not expected by QwenImagePipeline and will be ignored.
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:17<00:00,  5.72s/it]
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 729/729 [00:08<00:00, 88.28it/s]
Loading pipeline components...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:28<00:00,  5.65s/it]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [03:21<00:00,  4.02s/it]
Processed: outputs/jpc_scripts/N5/69730939758695870a8ca994/script_001.json -> outputs/jpc_scripts/images/69730939758695870a8ca994_script_001.png
Finished. Updated 1 script file(s).


```

The image could be look like:
![image](outputs/jpc_scripts/images/69730939758695870a8ca994_script_001.png)


3. the final video could look like (Please download video in output folder as [▶️ video](outputs/jpc_scripts/videos/N5/69730939758695870a8ca994/efefabef883b44b3aac99f9280663d21.mp4) to hear voice/sound!):


[video here is not play online in github!!!]
<video controls width="400">
  <source src="outputs/jpc_scripts/videos/N5/69730939758695870a8ca994/efefabef883b44b3aac99f9280663d21.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<video controls>
  <source src="https://raw.githubusercontent.com/VADNgrup/JLPT-listening-practice-video-generation/outputs/jpc_scripts/videos/N5/69730939758695870a8ca994/efefabef883b44b3aac99f9280663d21.mp4" type="video/mp4">
</video>


If the dialog in video turn to [][][], it means that u missing JP supported-font
please use `apt-get install fonts-noto-cjk` if you are using linux/ubuntu


## Limitation
1. Currently, this repo have no code for UI. Every modification such as voice/prompt changing must to be updated directly inside the code!!!
2. For dialogue_transcript_generation, the logic only apply for formats of the input  I collected from Castudy

## Acknoledgement
- Thank Staff of Castudy.vn to allow me use their data.
- Thank Goutam sensei to 
