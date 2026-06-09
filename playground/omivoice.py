# pip install omnivoice

from omnivoice import OmniVoice
import soundfile as sf
import torch

# Load the model
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16
)

charactor = [
    # {'ref_audio': './ai_vy.wav', 'name': 'Alice', 
    #  'ref_text': "việt nam đang kiêu hãnh bước vào kỷ nguyên vươn mình rực rỡ với khát vọng mãnh liệt, trí tuệ đổi mới và tinh thần đoàn kết đất nước, tự tin bứt phá, kiến tạo một tương lai thịnh vượng và vươn tầm quốc tế.",
    # },
    {
        'ref_audio': '../ingredients/voices/voice_preview_mitsuki.mp3', 'name': 'Mitsuki', 'Gender':'female',
        'ref_text': "みんな、今日も元気にしてる。水木の声が君の毎日をちょっぴり特別にできたら嬉しいな。小さな夢もいつ叶えられるよ。さあ、一緒に声で魔法をかけちゃお。",
    },
    {
        'ref_audio': '../ingredients/voices/voice_preview_hinata.mp3', 'name': 'Hinata', 'Gender':'male',
        'ref_text': "スイスの観光地として人気の高い、ユングフラウ鉄道。ここにはヨーロッパでとても高い場所にある駅があります。アルプスの雄大な景色を眺めながら、列車は終点の駅を目指し走り出します。"
    },
    {
        'ref_audio': '../ingredients/voices/voice_preview_koichi.mp3', 'name': 'Koichi', 'Gender':'male',
        'ref_text': "こんばんは。中低音の落ち着きのあるこの声を使って、いろいろ声で遊んでみてください。ナレーションからキャラクターボイスまで対応しております。皆さんのご利用お待ちしております。",
    },
    {
        'ref_audio': '../ingredients/voices/voice_preview_fumi.mp3',
        'name': 'Fumi',
        'Gender': 'female',
        'ref_text': "こんにちは。優しく落ち着いた声で、聞く人に安心感や前向きな気持ちを届けます。ナレーションやガイドサポート音声など、聞きやすさと信頼感を大切にしています。ぜひお任せください。",
    },
    {
        'ref_audio': '../ingredients/voices/voice_preview_akira.mp3',
        'name': 'Akira',
        'Gender': 'male',
        'ref_text': "こんにちは。明です。この声は落ち着いたトーンでありながら、感情の強弱や自然な問を大切にした日本語ナレーションに適しています。優しい説明から力強い表現まで幅広く対応できます。",
    },
    {
        'ref_audio': '../ingredients/voices/voice_preview_natsu.mp3',
        'name': 'Natsu',
        'Gender': 'female',
        'ref_text': "こんにちは。日本人女性の落ち着いた声です。物語やドキュメンタリーレクチャーなど、様々なコアに適していると思います。声には説得力があると思いますので、教育用の教材に利用してください。",
    }
]

# Generate audio
audio = model.generate(
    #supported tags: [laughter], [sigh], [confirmation-en], [question-en], [question-ah], [question-oh], [question-ei], [question-yi], [surprise-ah], [surprise-oh], [surprise-wa], [surprise-yo], [dissatisfaction-hnn].
    text="[sigh] そうですか。 [question-en]",
    ref_audio=charactor[0]['ref_audio'],
    ref_text=charactor[0]['ref_text'],
) # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.

sf.write("out.wav", audio[0], 24000)
