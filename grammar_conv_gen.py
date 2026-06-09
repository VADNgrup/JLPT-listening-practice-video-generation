import os
import json
import random
from pathlib import Path
import time
import re

from qna_engine import QNAEngine
# --- CONFIGURATION ---
QNA_ENGINE = QNAEngine()

LECTURE_BANK_PATH = Path("lecture_bank.json")
SCRIPTS_OUTPUT_DIR = Path("jpc_scripts_v2")
SCRIPTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TTS_TAGS = [
    "[laughter]", "[sigh]", "[confirmation-en]", "[question-en]", 
    # "[question-ah]", "[question-oh]", "[question-ei]", "[question-yi]", 
    "[surprise-ah]", "[surprise-oh]", 
    # "[surprise-wa]", "[surprise-yo]", 
    # "[dissatisfaction-hnn]"
]

def generate_conversation(level, grammar_name, grammar_content, words):
    prompt = f"""You are an expert Japanese curriculum developer for level {level}.
Task: Write a natural Japanese conversation script using the grammar point and vocabulary provided below.

Grammar Point: {grammar_name}
Grammar Details: {json.dumps(grammar_content, ensure_ascii=False)}
Mandatory Vocabulary to use: {", ".join(words)}

Requirements:
1. Number of characters: 2-4 people.
2. Content: A realistic daily communication situation (dining, working, school, traveling, shopping, etc.).
3. Language: The script MUST be in Japanese.
4. You could use the following emotion tags naturally within the dialogue (compatible with the TTS model):
   {", ".join(TTS_TAGS)}
5. Ensure the Vocabulary, Grammar, and Complexity are appropriate for Japanese level {level}.
6. Context and character descriptions should be in English and Vietnamese version.
7. The conversation should be engaging and educational, suitable for learners at the specified level.
8. Avoid using overly complex sentences that may not be suitable for the target level, but ensure the grammar point is clearly demonstrated.
9. The converstion should have  greetings, endings, a clear topic, and a natural flow of dialogue.
10. The total length of the conversation should be around 10-20 lines of dialogue (excluding greetings and endings).

Return exactly a JSON object with this structure:
{{
    "level": "{level}",
    "grammar_id": "...",
    "grammar_name": "{grammar_name}",
    "core_words": {json.dumps(words, ensure_ascii=False)},
    "context": "Description of the setting in English",
    "context_vi": "Mô tả bối cảnh bằng tiếng Việt",
    "characters": [
        {{"name": "Name in Romaji such as Tanaka", "gender": "Male/Female", "profession": "..."}}
    ],
    "scripts": [
        {{"char": "Character Name", "text": "Japanese dialogue text", "en_translation": "English translation of the dialogue line", "vi_translation": "Dịch tiếng Việt của câu thoại"}}
    ]
}}
"""
    try:
        response = QNA_ENGINE.ask(
            prompt,
            req_type="eval|explain",
            language="Japanese",
            use_history=False,
        )
        if response.get("error"):
            raise RuntimeError(response.get("message", "QNA engine error"))

        content = (response.get("response") or "").strip()
        print(content)  # Debug: print the prompt

        match = re.search(r"```json\s*(\{.*\})\s*```", content, re.DOTALL)
        if match:
            content = match.group(1)
        else:
            match = re.search(r"(\{.*\})", content, re.DOTALL)
            if match:
                content = match.group(1)

        return json.loads(content)
    except Exception as e:
        print(f"Error generating conversation: {e}")
        return None

def main():
    if not LECTURE_BANK_PATH.exists():
        print(f"File {LECTURE_BANK_PATH} not found.")
        return

    with open(LECTURE_BANK_PATH, "r", encoding="utf-8") as f:
        bank = json.load(f)

    all_conversations = {"N5": [], "N4": [], "N3": []}

    for level in ["N5", "N4", "N3"]:
        level_data = bank.get(level, {})
        # Chia thành 2 nhóm: grammar và vựng
        grammar_lectures = {k: v for k, v in level_data.items() if v['category'] == 'grammarjson'}
        vocab_lectures = {k: v for k, v in level_data.items() if v['category'] == 'flashcards'}

        # Gom tất cả từ vựng của level này
        all_words_in_level = []
        for lec in vocab_lectures.values():
            all_words_in_level.extend(lec.get('words', []))
        
        # Tạo danh sách các từ chưa được sử dụng
        available_words = list(all_words_in_level)

        if not grammar_lectures or not all_words_in_level:
            print(f"Missing grammar or words for level {level}. Skipping.")
            continue

        for lec_id, info in grammar_lectures.items():
            print(f"Generating 50 scripts for Grammar: {info['lecture_name']} ({level})")
            
            # Create subfolder for each grammar point
            grammar_dir = SCRIPTS_OUTPUT_DIR / level / lec_id
            grammar_dir.mkdir(parents=True, exist_ok=True)
            
            count = 0
            while count < 50:
                # Check for existing scripts to resume if needed
                script_filename = grammar_dir / f"script_{count+1:03d}.json"
                if script_filename.exists():
                    count += 1
                    continue

                # Chọn số từ cần lấy (5-8)
                num_to_sample = min(len(all_words_in_level), random.randint(8, 16))
                
                # Nếu không đủ từ trong available_words, reset lại danh sách
                if len(available_words) < num_to_sample:
                    print(f"  Resetting available words for level {level}")
                    available_words = list(all_words_in_level)
                
                # Chọn từ available_words và loại bỏ chúng
                sampled_words = random.sample(available_words, num_to_sample)
                for w in sampled_words:
                    available_words.remove(w)

                content = info.get('content').get("data", {})
                if not content:
                    print(f"  No content found for {info['lecture_name']} ({lec_id}). Skipping.")
                    continue

                content = content[0].get('slidejson', {})
                if not content:
                    print(f"  No slidejson content for {info['lecture_name']} ({lec_id}). Skipping.")
                    continue
                if not 'grammar_points' in content or not content['grammar_points']:
                    print(f"  No grammar_points in content for {info['lecture_name']} ({lec_id}). Skipping.")
                    continue
                grammar_points:list = content['grammar_points']
                content['grammar_points'] = [{
                    "title": gp.get('title'),
                    "grammar": gp.get('meaning_en')
                } for gp in grammar_points]

                processed_words = [w.split("_")[0] for w in sampled_words ]
                conv = generate_conversation(level, info['lecture_name'], content, processed_words)
                if conv:
                    conv['grammar_id'] = lec_id
                    
                    # Save individual script file
                    with open(script_filename, "w", encoding="utf-8") as f_out:
                        json.dump(conv, f_out, ensure_ascii=False, indent=4)
                    
                    count += 1
                    print(f"  [{count}/50] Saved script to {script_filename}")
                
                time.sleep(2)  # Thêm độ trễ để tránh quá tải API


    print(f"Finished! All scripts generated and saved in {SCRIPTS_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
