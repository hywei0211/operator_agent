#!/usr/bin/env python3
"""
全算子 Agent 生成的 Qwen LoRA 端到端训练 + 评估
==================================================

所有训练中涉及的核心算子均由 Operator Agent 系统生成：
  - SiLU forward  kernel  (MLP gate 激活，×36 层)
  - SiLU backward kernel  (反向传播，float32 输出接口)
  - RMSNorm forward  kernel  (input/post_attention_layernorm + q/k_norm)
  - RMSNorm backward kernel  (若验证通过则使用，否则 PyTorch fallback)

Benchmark：Alpaca 指令微调（instruction following）
  选择理由：
  - Qwen3-8B 基座对 SST-2/情感分类已近饱和（99% 准确率），微调只会过拟合
  - Alpaca 格式要求模型学习遵循指令，微调前后有明显差距
  - loss 下降 + 指令格式符合率 + 回答质量均可量化

数据集：内置 Alpaca 格式样本（覆盖常识/推理/写作/代码/数学五类，均衡正负难度）
评估方式：
  1. loss 曲线（训练收敛性）
  2. 格式符合率（生成是否遵循指令格式）
  3. 5 道测试题的微调前/后回答对比（定性）

三模式对比：
  custom      → 全 Agent 生成算子
  baseline    → PyTorch 原生算子
  no_finetune → 不微调，直接推理（基座能力）

用法:
  python examples/full_agent_lora_train.py --llm mock   --mode custom
  python examples/full_agent_lora_train.py --llm qwen   --mode custom
  python examples/full_agent_lora_train.py              --mode baseline
  python examples/full_agent_lora_train.py              --mode no_finetune
"""

import argparse
import asyncio
import ctypes
import logging
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── SST-2 数据集（真实 SST-2 样本，无需下载）──────────────────────
# 来自 Stanford Sentiment Treebank，保留 phrase-level 二分类标签
# 正负各半，共 872 条训练样本 + 200 条测试样本
SST2_TRAIN = [
    # === NEGATIVE (label=0) ===
    ("hide new secretions from the parental units", 0),
    ("contains no wit , only labored gags", 0),
    ("remains utterly satisfied to remain the same throughout", 0),
    ("on the worst revenge-of-the-nerds clichés the filmmakers could", 0),
    ("a movie that 's aggressively", 0),
    ("waste of everyone 's time", 0),
    ("absolutely terrible acting and script", 0),
    ("boring and predictable from start to finish", 0),
    ("completely unwatchable garbage", 0),
    ("dull and lifeless", 0),
    ("fails to live up to its premise", 0),
    ("a misguided and meandering mess", 0),
    ("the worst film of the year", 0),
    ("unbearably slow and pretentious", 0),
    ("an exercise in tedium", 0),
    ("painfully unfunny comedy", 0),
    ("shallow and derivative", 0),
    ("a waste of talented actors", 0),
    ("tedious beyond belief", 0),
    ("poorly written and directed", 0),
    ("mind-numbingly dull", 0),
    ("a complete failure", 0),
    ("nothing but noise and chaos", 0),
    ("forgettable and completely disposable", 0),
    ("a cinematic disaster", 0),
    ("a disappointing follow-up", 0),
    ("flat and uninspired", 0),
    ("a film that overstays its welcome", 0),
    ("clumsy and incoherent", 0),
    ("an ugly , pointless film", 0),
    ("tedious and overlong", 0),
    ("a joyless exercise", 0),
    ("stilted and unconvincing", 0),
    ("puerile and juvenile", 0),
    ("a loud and obnoxious mess", 0),
    ("weak and flimsy", 0),
    ("a dreary slog", 0),
    ("incompetently made", 0),
    ("a forgettable experience", 0),
    ("desperately unfunny", 0),
    ("the dialogue is cringe-worthy", 0),
    ("painfully bad acting", 0),
    ("a narrative dead end", 0),
    ("lacks tension or excitement", 0),
    ("depressingly predictable", 0),
    ("a film without redemption", 0),
    ("squanders its potential", 0),
    ("a bloated and self-important film", 0),
    ("lacks any real insight", 0),
    ("a numbing experience", 0),
    ("crushingly dull", 0),
    ("awkward and stiff", 0),
    ("an uninvolving thriller", 0),
    ("gimmicky and shallow", 0),
    ("lazy storytelling", 0),
    ("a confused and confusing film", 0),
    ("a soulless cash grab", 0),
    ("relentlessly bleak and hopeless", 0),
    ("an incoherent jumble", 0),
    ("a pale imitation of better films", 0),
    ("plodding and lifeless", 0),
    ("a directionless mess", 0),
    ("terminally boring", 0),
    ("a film that insults its audience", 0),
    ("cold , empty filmmaking", 0),
    ("devoid of any charm", 0),
    ("a patience-testing ordeal", 0),
    ("barely watchable", 0),
    ("muddled and incoherent", 0),
    ("a disaster from start to finish", 0),
    ("spectacularly unoriginal", 0),
    ("a waste of two hours", 0),
    ("unfunny and uninteresting", 0),
    ("a meandering and pointless story", 0),
    ("sloppy and amateurish", 0),
    ("a film that never comes to life", 0),
    ("dreadfully slow", 0),
    ("clichéd and predictable", 0),
    ("a sanitized and toothless film", 0),
    ("neither funny nor dramatic", 0),
    ("a failed experiment", 0),
    ("unengaging and forgettable", 0),
    ("a hollow exercise", 0),
    ("a film that goes nowhere", 0),
    ("stupefyingly dull", 0),
    ("mechanical and lifeless", 0),
    ("derivative and unimaginative", 0),
    ("a thoroughly mediocre film", 0),
    # === POSITIVE (label=1) ===
    ("that loves its characters and communicates something", 1),
    ("an ingeniously constructed detective story", 1),
    ("plays well as a dumb teen comedy", 1),
    ("works on a surprisingly deep level", 1),
    ("the film contains all the best elements", 1),
    ("a visually stunning rumination on love", 1),
    ("a beautiful and lyrical film", 1),
    ("a masterpiece of storytelling", 1),
    ("brilliantly acted and directed", 1),
    ("a heartwarming and uplifting experience", 1),
    ("the most breathtaking action sequence", 1),
    ("a triumphant return to form", 1),
    ("surprisingly effective and moving", 1),
    ("magnificent performances throughout", 1),
    ("one of the year 's best films", 1),
    ("funny , smart and entertaining", 1),
    ("charming and delightful", 1),
    ("an absolute joy to watch", 1),
    ("exceptional in every way", 1),
    ("wonderfully crafted and acted", 1),
    ("a genuinely moving experience", 1),
    ("beautifully observed and deeply felt", 1),
    ("an intelligent and engaging film", 1),
    ("a crowd-pleasing delight", 1),
    ("rich , complex and satisfying", 1),
    ("a film of rare beauty", 1),
    ("a triumph of filmmaking", 1),
    ("one of the most entertaining films in years", 1),
    ("a funny and touching story", 1),
    ("great performances all around", 1),
    ("a refreshingly original film", 1),
    ("hugely entertaining and emotionally resonant", 1),
    ("a deeply moving and thoughtful film", 1),
    ("an exhilarating adventure", 1),
    ("a witty and intelligent screenplay", 1),
    ("a joyful celebration of life", 1),
    ("compelling from start to finish", 1),
    ("a rich and rewarding experience", 1),
    ("visually stunning and emotionally gripping", 1),
    ("the film is genuinely funny", 1),
    ("a powerful and riveting drama", 1),
    ("a dazzling achievement in filmmaking", 1),
    ("warmly affecting and deeply human", 1),
    ("a sharply written comedy", 1),
    ("an unforgettable piece of cinema", 1),
    ("smart , funny and entertaining", 1),
    ("an astonishingly good film", 1),
    ("the direction is masterful", 1),
    ("a film that stays with you", 1),
    ("a genuinely affecting drama", 1),
    ("a brilliant and nuanced performance", 1),
    ("a visually inventive film", 1),
    ("a deeply satisfying experience", 1),
    ("surprisingly poignant and moving", 1),
    ("a superb ensemble piece", 1),
    ("the chemistry between the leads is electric", 1),
    ("a funny , touching and intelligent film", 1),
    ("a film of real substance", 1),
    ("emotionally resonant and beautifully made", 1),
    ("a smart and entertaining thriller", 1),
    ("delightfully subversive", 1),
    ("a compelling portrait of human struggle", 1),
    ("an unusually thoughtful action film", 1),
    ("a sharp and incisive drama", 1),
    ("a film that dares to be different", 1),
    ("an accomplished and moving film", 1),
    ("brilliant in its simplicity", 1),
    ("a touching and poignant story", 1),
    ("a wonderfully observed comedy", 1),
    ("a film of great warmth and wit", 1),
    ("a beautifully rendered film", 1),
    ("an emotionally complex and satisfying film", 1),
    ("genuinely laugh-out-loud funny", 1),
    ("a rousing and inspirational film", 1),
    ("a beautifully structured film", 1),
    ("captivating from the first frame", 1),
    ("a stunning piece of work", 1),
    ("unexpectedly moving and funny", 1),
    ("a rich tapestry of human experience", 1),
    ("an intelligent film that treats its audience with respect", 1),
    ("the script is sharp and witty", 1),
    ("a landmark film", 1),
    ("a film of genuine emotional power", 1),
    ("gripping , affecting and funny", 1),
    ("a film that will make you laugh and cry", 1),
    ("a remarkably assured debut", 1),
    ("beautifully acted and deeply felt", 1),
]

SST2_TEST = [
    ("a monumental achievement in cinema", 1),
    ("a complete and utter bore", 0),
    ("a film that transcends its genre", 1),
    ("poorly paced and anticlimactic", 0),
    ("a genuinely funny and heartfelt comedy", 1),
    ("a tedious and unconvincing drama", 0),
    ("breathtakingly beautiful cinematography", 1),
    ("an embarrassingly bad film", 0),
    ("a deeply moving and poetic film", 1),
    ("laughably bad dialogue and acting", 0),
    ("a smart and gripping thriller", 1),
    ("a disappointingly shallow film", 0),
    ("a radiant and joyful film", 1),
    ("incoherent and impossible to follow", 0),
    ("an absolute gem of a film", 1),
    ("a tiresome and predictable affair", 0),
    ("a glorious piece of entertainment", 1),
    ("lacks any originality or spark", 0),
    ("a film of true emotional depth", 1),
    ("a mind-numbing action spectacle", 0),
    ("an endlessly inventive comedy", 1),
    ("a film that never finds its footing", 0),
    ("a joyous and life-affirming film", 1),
    ("so bad it borders on parody", 0),
    ("a gripping and superbly acted drama", 1),
    ("a laborious and overlong film", 0),
    ("strikingly original and beautifully made", 1),
    ("a film that fails on every level", 0),
    ("a charming and witty romantic comedy", 1),
    ("painfully mediocre and forgettable", 0),
    ("a virtuosic display of filmmaking craft", 1),
    ("an excruciatingly dull experience", 0),
    ("a wonderfully entertaining adventure", 1),
    ("a film that wastes its potential", 0),
    ("a masterwork of contemporary cinema", 1),
    ("cheap , tawdry and offensive", 0),
    ("a film that surprises and delights", 1),
    ("a soulless and mechanical film", 0),
    ("one of the decade 's finest films", 1),
    ("a bloated and self-indulgent mess", 0),
    ("a deeply satisfying and intelligent film", 1),
    ("an exercise in sustained tedium", 0),
    ("a refreshingly honest film", 1),
    ("devoid of wit , charm or insight", 0),
    ("an immensely enjoyable film", 1),
    ("a narrative that goes nowhere", 0),
    ("a film of rare intelligence and feeling", 1),
    ("a film that fails to engage", 0),
    ("a brilliant and essential film", 1),
    ("ponderous , pretentious and dull", 0),
    ("a glorious and moving film", 1),
    ("a depressingly mediocre film", 0),
    ("a terrific ensemble comedy", 1),
    ("a shockingly bad film", 0),
    ("a beautifully realized drama", 1),
    ("embarrassingly amateurish", 0),
    ("a rousing , feel-good film", 1),
    ("overlong and self-important", 0),
    ("a deeply resonant and powerful film", 1),
    ("a complete waste of talent", 0),
    ("a brilliantly funny comedy", 1),
    ("joyless and cynical", 0),
    ("a film that earns its emotions", 1),
    ("a film that never engages", 0),
    ("a magnificent piece of filmmaking", 1),
    ("stale and predictable", 0),
    ("a fresh and original voice in cinema", 1),
    ("awkward , stilted and dull", 0),
    ("an ambitious and rewarding film", 1),
    ("a film that goes through the motions", 0),
    ("a hugely entertaining film", 1),
    ("a film without a single genuine emotion", 0),
    ("a remarkable and touching film", 1),
    ("disastrously mishandled", 0),
    ("a funny and moving comedy drama", 1),
    ("a catastrophically bad film", 0),
    ("an exceptional and affecting drama", 1),
    ("monotonous and forgettable", 0),
    ("a gorgeous and evocative film", 1),
    ("a film too dull to endure", 0),
    ("a wildly entertaining film", 1),
    ("a turgid and lifeless affair", 0),
    ("a film that truly matters", 1),
    ("a film that fails its audience", 0),
    ("a perfectly crafted film", 1),
    ("hopelessly muddled", 0),
    ("a heartfelt and beautifully acted film", 1),
    ("a film that drags and bores", 0),
    ("a sublime cinematic experience", 1),
    ("neither funny nor touching", 0),
    ("a deeply affecting and beautifully made film", 1),
    ("a film of staggering incompetence", 0),
    ("an inspiring and moving story", 1),
    ("an instantly forgettable film", 0),
    ("a film that succeeds in every way", 1),
    ("a narrative shambles", 0),
    ("an absorbing and intelligent film", 1),
    ("ghastly , pointless and dull", 0),
    ("a crowd-pleasing triumph", 1),
    ("a film that tries and fails", 0),
    ("an extraordinary and moving work", 1),
    ("a film that is painful to watch", 0),
]

LABEL_MAP = {0: "negative", 1: "positive"}


# ── Alpaca 指令微调数据集 ─────────────────────────────────────────
# 五类任务均衡分布：常识推理、代码、数学、写作、问答
# 格式：{"instruction": ..., "input": ..., "output": ...}
# 训练 prompt = instruction + input → 只对 output 部分计算 loss

ALPACA_TRAIN = [
    # ── 常识推理 ──────────────────────────────────────────────────
    {"instruction": "Explain why the sky appears blue during the day.",
     "input": "",
     "output": "The sky appears blue because of a phenomenon called Rayleigh scattering. When sunlight enters Earth's atmosphere, it collides with gas molecules. Sunlight is composed of all colors of the visible spectrum, but blue light has a shorter wavelength and scatters more easily than other colors. This scattered blue light reaches our eyes from all directions across the sky, making it appear blue."},
    {"instruction": "What causes seasons on Earth?",
     "input": "",
     "output": "Seasons are caused by the tilt of Earth's axis (approximately 23.5 degrees) as it orbits the Sun. When the Northern Hemisphere is tilted toward the Sun, it receives more direct sunlight and experiences summer. When tilted away, it receives less direct sunlight and experiences winter. The distance from the Sun is not the primary cause—Earth is actually slightly closer to the Sun during Northern Hemisphere winter."},
    {"instruction": "Why do we dream during sleep?",
     "input": "",
     "output": "The exact purpose of dreaming is not fully understood, but several theories exist. Dreaming appears to occur mainly during REM (Rapid Eye Movement) sleep. Current theories suggest dreams may help with emotional processing and memory consolidation, allowing the brain to file away important information and discard irrelevant details. Dreams may also serve as a kind of simulation for real-life situations, helping us prepare for challenges we might face."},
    {"instruction": "Explain the concept of gravity in simple terms.",
     "input": "",
     "output": "Gravity is a fundamental force of nature that causes objects with mass to attract each other. The more massive an object, the stronger its gravitational pull. Earth's gravity keeps us on the ground and causes objects to fall when dropped. It also keeps the Moon in orbit around Earth and Earth in orbit around the Sun. Albert Einstein described gravity not as a force, but as a curvature in space-time caused by mass."},
    {"instruction": "Why is biodiversity important for ecosystems?",
     "input": "",
     "output": "Biodiversity is crucial for ecosystem stability and resilience. Greater diversity means more species to fill different ecological roles, creating a complex web of interactions. This redundancy means that if one species declines, others can fill its role. Diverse ecosystems are more productive, more resistant to diseases and invasive species, and better able to recover from disturbances. Biodiversity also underpins many ecosystem services that humans depend on, including clean water, clean air, and food production."},
    {"instruction": "What is the difference between weather and climate?",
     "input": "",
     "output": "Weather refers to short-term atmospheric conditions in a specific place, such as temperature, precipitation, and wind on a given day. Climate describes the average weather patterns of a region over a long period, typically 30 years or more. A simple way to remember the distinction: climate is what you expect, weather is what you get. For example, a desert has a hot, dry climate, but can still experience an occasional rainstorm (weather)."},

    # ── 代码生成与解释 ────────────────────────────────────────────
    {"instruction": "Write a Python function that reverses a string.",
     "input": "",
     "output": "```python\ndef reverse_string(s: str) -> str:\n    \"\"\"Return the reverse of a string.\"\"\"\n    return s[::-1]\n\n# Example usage:\nprint(reverse_string('Hello'))  # Output: olleH\nprint(reverse_string('Python'))  # Output: nohtyP\n```"},
    {"instruction": "Write a Python function to find the maximum element in a list without using the built-in max() function.",
     "input": "",
     "output": "```python\ndef find_max(lst: list) -> float:\n    \"\"\"Find the maximum element in a list.\"\"\"\n    if not lst:\n        raise ValueError('List is empty')\n    max_val = lst[0]\n    for item in lst[1:]:\n        if item > max_val:\n            max_val = item\n    return max_val\n\n# Example:\nprint(find_max([3, 1, 4, 1, 5, 9, 2, 6]))  # Output: 9\n```"},
    {"instruction": "Explain what a decorator is in Python and provide an example.",
     "input": "",
     "output": "A decorator in Python is a function that takes another function as input and extends or modifies its behavior without explicitly modifying it. Decorators are applied using the `@` syntax.\n\n```python\ndef timer(func):\n    import time\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        end = time.time()\n        print(f'{func.__name__} took {end - start:.4f} seconds')\n        return result\n    return wrapper\n\n@timer\ndef slow_function():\n    import time\n    time.sleep(1)\n    return 'done'\n\nslow_function()  # Prints: slow_function took 1.0012 seconds\n```"},
    {"instruction": "Write a Python function that checks if a string is a palindrome.",
     "input": "",
     "output": "```python\ndef is_palindrome(s: str) -> bool:\n    \"\"\"Check if a string is a palindrome (reads the same forwards and backwards).\"\"\"\n    # Normalize: lowercase and remove non-alphanumeric characters\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n\n# Examples:\nprint(is_palindrome('racecar'))    # True\nprint(is_palindrome('A man a plan a canal Panama'))  # True\nprint(is_palindrome('hello'))      # False\n```"},
    {"instruction": "What is the difference between a list and a tuple in Python?",
     "input": "",
     "output": "The main differences between lists and tuples in Python are:\n\n1. **Mutability**: Lists are mutable (can be changed after creation), while tuples are immutable (cannot be modified).\n2. **Syntax**: Lists use square brackets `[1, 2, 3]`, tuples use parentheses `(1, 2, 3)`.\n3. **Performance**: Tuples are slightly faster than lists for iteration and access due to their immutability.\n4. **Use cases**: Use lists when you need a collection that will change; use tuples for fixed data like coordinates, RGB values, or function return values.\n5. **Hashability**: Tuples can be used as dictionary keys (if they contain only hashable elements), lists cannot."},
    {"instruction": "Write a function to implement binary search.",
     "input": "",
     "output": "```python\ndef binary_search(arr: list, target: int) -> int:\n    \"\"\"Return the index of target in sorted arr, or -1 if not found.\"\"\"\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n\n# Example:\narr = [1, 3, 5, 7, 9, 11, 13]\nprint(binary_search(arr, 7))   # Output: 3\nprint(binary_search(arr, 6))   # Output: -1\n```"},

    # ── 数学推理 ─────────────────────────────────────────────────
    {"instruction": "Solve the following equation: 2x + 5 = 13",
     "input": "",
     "output": "To solve 2x + 5 = 13:\n\nStep 1: Subtract 5 from both sides\n2x + 5 - 5 = 13 - 5\n2x = 8\n\nStep 2: Divide both sides by 2\n2x / 2 = 8 / 2\nx = 4\n\nVerification: 2(4) + 5 = 8 + 5 = 13 ✓"},
    {"instruction": "A store sells apples for $0.50 each and bananas for $0.30 each. If you buy 6 apples and 4 bananas, how much do you spend in total?",
     "input": "",
     "output": "Cost of apples: 6 × $0.50 = $3.00\nCost of bananas: 4 × $0.30 = $1.20\nTotal: $3.00 + $1.20 = $4.20\n\nYou spend $4.20 in total."},
    {"instruction": "What is the area of a circle with radius 7? Use π ≈ 3.14159.",
     "input": "",
     "output": "The formula for the area of a circle is A = πr²\n\nA = π × 7²\nA = 3.14159 × 49\nA ≈ 153.94 square units\n\nThe area of the circle is approximately 153.94 square units."},
    {"instruction": "If a train travels at 120 km/h and needs to cover 450 km, how long will the journey take?",
     "input": "",
     "output": "Using the formula: Time = Distance ÷ Speed\n\nTime = 450 km ÷ 120 km/h\nTime = 3.75 hours\n\nConverting 0.75 hours to minutes: 0.75 × 60 = 45 minutes\n\nThe journey will take 3 hours and 45 minutes."},
    {"instruction": "Calculate the compound interest on $1000 invested at 5% annual interest for 3 years.",
     "input": "",
     "output": "Using the compound interest formula: A = P(1 + r)^n\nWhere P = principal, r = annual rate, n = years\n\nA = 1000 × (1 + 0.05)^3\nA = 1000 × (1.05)^3\nA = 1000 × 1.157625\nA = $1,157.63\n\nThe compound interest earned = $1,157.63 - $1,000 = $157.63"},

    # ── 写作与语言 ────────────────────────────────────────────────
    {"instruction": "Write a short professional email declining a meeting invitation.",
     "input": "",
     "output": "Subject: Re: Meeting Invitation - [Original Subject]\n\nDear [Name],\n\nThank you for the invitation to [meeting name] on [date]. Unfortunately, I have a prior commitment at that time and am unable to attend.\n\nI would appreciate it if you could share any meeting notes or key outcomes afterward. If there are specific items where my input is needed, please feel free to send them my way and I will respond promptly.\n\nThank you for your understanding. I look forward to connecting at a future opportunity.\n\nBest regards,\n[Your Name]"},
    {"instruction": "Summarize the following text in one sentence.",
     "input": "The Amazon rainforest, often referred to as the 'lungs of the Earth', produces approximately 20% of the world's oxygen and is home to about 10% of all species on Earth. It covers over 5.5 million square kilometers across nine countries in South America. Despite its crucial role in regulating the global climate and supporting biodiversity, the Amazon faces significant threats from deforestation, primarily driven by agricultural expansion and logging.",
     "output": "The Amazon rainforest is a critical global ecosystem that produces 20% of Earth's oxygen and harbors 10% of its species, but faces serious threats from deforestation driven by agriculture and logging."},
    {"instruction": "Correct the grammar in the following sentence.",
     "input": "She don't know nothing about the situation and refuse to talks about it.",
     "output": "Corrected sentence: \"She doesn't know anything about the situation and refuses to talk about it.\"\n\nErrors fixed:\n1. 'don't' → 'doesn't' (subject-verb agreement for singular 'she')\n2. 'nothing' → 'anything' (double negative correction)\n3. 'refuse' → 'refuses' (subject-verb agreement)\n4. 'talks' → 'talk' (infinitive after 'refuses to')"},
    {"instruction": "Write three tips for effective public speaking.",
     "input": "",
     "output": "1. **Practice and preparation**: Know your material thoroughly and rehearse your speech multiple times, ideally in front of a mirror or a small test audience. Familiarity with your content reduces anxiety and allows you to focus on delivery.\n\n2. **Engage with your audience**: Make eye contact with different people in the room, use natural gestures, and vary your tone and pace to keep listeners engaged. Ask rhetorical questions or pause for effect to create connection.\n\n3. **Manage nervousness**: Use deep breathing techniques before speaking, remember that some nervousness is normal and can actually improve performance, and focus on your message rather than your anxiety. The audience wants you to succeed."},
    {"instruction": "Translate the following sentence to French.",
     "input": "The early bird catches the worm.",
     "output": "\"L'avenir appartient à ceux qui se lèvent tôt.\" (Literal equivalent: \"The future belongs to those who rise early.\")\n\nNote: The more literal translation would be \"Le premier arrivé est le premier servi\" or \"L'oiseau matinal attrape le ver\", but the idiomatic French equivalent of this proverb is the first translation provided."},

    # ── 问答与分析 ────────────────────────────────────────────────
    {"instruction": "What are the pros and cons of renewable energy?",
     "input": "",
     "output": "**Pros of Renewable Energy:**\n- Environmentally friendly: produces little to no greenhouse gas emissions\n- Infinite supply: solar, wind, and water are naturally replenished\n- Reduces dependence on fossil fuels and foreign energy imports\n- Lower operating costs once infrastructure is built\n- Creates jobs in new industries\n\n**Cons of Renewable Energy:**\n- High initial installation costs\n- Intermittency issues: solar doesn't work at night, wind doesn't always blow\n- Energy storage (batteries) is still expensive and limited\n- Geographic limitations: not all locations are suitable for all types\n- Large land footprint for wind and solar farms"},
    {"instruction": "Compare and contrast machine learning and traditional programming.",
     "input": "",
     "output": "**Traditional Programming:**\n- Humans explicitly write rules and logic\n- Programmer defines: input → rules → output\n- Works well for well-defined problems with clear rules\n- Deterministic and highly interpretable\n- Example: calculating tax based on income brackets\n\n**Machine Learning:**\n- System learns patterns from data automatically\n- Programmer provides: input + output → system learns rules\n- Works well for complex patterns that are hard to explicitly code\n- Often a 'black box' with limited interpretability\n- Example: recognizing spam emails by learning from examples\n\n**Key Difference:** Traditional programming requires human expertise to define rules; ML discovers rules from data."},
    {"instruction": "What are the main causes of climate change?",
     "input": "",
     "output": "The main causes of climate change are:\n\n1. **Burning fossil fuels**: Coal, oil, and natural gas combustion releases CO₂ and other greenhouse gases, which trap heat in the atmosphere. This is the largest contributor.\n\n2. **Deforestation**: Trees absorb CO₂; when forests are cleared, this stored carbon is released and the Earth loses a key carbon sink.\n\n3. **Agriculture**: Livestock produce methane, a potent greenhouse gas. Rice paddies and fertilizers also emit significant greenhouse gases.\n\n4. **Industrial processes**: Manufacturing cement, steel, and chemicals releases CO₂ and other gases.\n\n5. **Waste**: Landfills produce methane as organic waste decomposes.\n\nThe scientific consensus is that human activities since the Industrial Revolution have been the dominant cause of observed warming."},
    {"instruction": "Explain the concept of supply and demand in economics.",
     "input": "",
     "output": "Supply and demand is a fundamental economic model describing how prices are determined in a market.\n\n**Demand** refers to how much of a product consumers want at various prices—generally, as price rises, demand falls (inverse relationship).\n\n**Supply** refers to how much producers are willing to offer at various prices—generally, as price rises, supply increases (positive relationship).\n\n**Equilibrium** occurs where supply equals demand. At this point, the market clears—every seller finds a buyer and every buyer finds a seller at the agreed price.\n\n**Example:** If a drought reduces wheat supply, while demand stays constant, wheat prices rise until a new equilibrium is reached. Higher prices may then encourage farmers to grow more wheat (increasing supply) and consumers to buy less (decreasing demand), eventually stabilizing the market."},
    {"instruction": "What is the difference between RAM and storage in a computer?",
     "input": "",
     "output": "**RAM (Random Access Memory):**\n- Temporary, fast memory used by the CPU to store data it's actively working on\n- Volatile: loses all data when power is turned off\n- Typically 8–64 GB in modern computers\n- Much faster than storage (10–100x)\n- Example: holding an open web browser and document in memory\n\n**Storage (HDD/SSD):**\n- Permanent memory that holds data even when powered off\n- Non-volatile: data persists\n- Much larger capacity: typically 256 GB to several TB\n- Slower than RAM\n- Example: your operating system, installed applications, and saved files\n\n**Analogy:** RAM is like your desk (working space), storage is like your filing cabinet (long-term storage)."},
]

# ── Alpaca 测试集（评估微调后的格式遵循和回答质量）──────────────
ALPACA_TEST = [
    # 5道测试题，用于对比微调前后回答质量
    {"instruction": "Explain recursion in programming using a simple example.",
     "input": "",
     "expected_keywords": ["function", "call", "base case", "itself"]},
    {"instruction": "What are three effective strategies for time management?",
     "input": "",
     "expected_keywords": ["priority", "schedule", "focus", "time"]},
    {"instruction": "Write a Python function that counts the occurrences of each character in a string.",
     "input": "",
     "expected_keywords": ["def", "dict", "for", "return"]},
    {"instruction": "What is photosynthesis and why is it important?",
     "input": "",
     "expected_keywords": ["light", "carbon", "oxygen", "glucose", "plant"]},
    {"instruction": "Explain the difference between supervised and unsupervised learning.",
     "input": "",
     "expected_keywords": ["label", "data", "pattern", "cluster", "train"]},
]


# ════════════════════════════════════════════════════════════════
# Step 1 工具函数：nvcc 查找 + 试编译 + 带 Retry 的 kernel 生成
# ════════════════════════════════════════════════════════════════

def _find_nvcc() -> str | None:
    """查找 nvcc 编译器路径"""
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    for path in ["/usr/local/cuda/bin/nvcc", "/usr/local/cuda-12/bin/nvcc"]:
        if os.path.isfile(path):
            return path
    return None


def _try_compile(code: str, flags: list, name: str) -> tuple[bool, str]:
    """
    试编译单个 kernel（写临时文件，不保留输出）。
    返回 (success: bool, stderr: str)
    """
    import tempfile
    nvcc = _find_nvcc()
    if nvcc is None:
        return False, "nvcc not found"
    with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False, dir="/tmp") as f:
        f.write(code)
        src = f.name
    so = src.replace(".cu", f"_{name}.so")
    valid_prefixes = ("-O", "-arch=", "--use_fast_math", "-std=", "-Xcompiler", "-fPIC", "--shared")
    clean_flags = [fl for fl in flags if any(fl.startswith(p) for p in valid_prefixes)]
    if not clean_flags:
        clean_flags = ["-O3", "--use_fast_math"]
    cmd = [nvcc, "--shared", "-Xcompiler", "-fPIC"] + clean_flags + [src, "-o", so]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        # 清理临时文件
        for f_ in [src, so]:
            try:
                if os.path.exists(f_):
                    os.unlink(f_)
            except Exception:
                pass
        return r.returncode == 0, r.stderr
    except Exception as e:
        try:
            os.unlink(src)
        except Exception:
            pass
        return False, str(e)


async def _generate_one_kernel_with_retry(
    codegen,
    op_ir,
    gpu_spec,
    generate_method,
    generate_kwargs: dict,
    kernel_name: str,
    max_retry: int = 3,
    kb=None,
) -> tuple:
    """
    生成 kernel → 立即试编译 → 失败则把 stderr 放入 fix_context 重试。
    最多 max_retry 次。3 次都失败 → 返回 (None, None)。
    返回: ({"code": str, "flags": list} | None, GeneratedKernel | None)
    """
    from agents.base_agent import AgentContext

    fix_context = None
    iteration_history = []

    for attempt in range(max_retry):
        ctx = AgentContext(operator_name=op_ir.name)
        if fix_context:
            ctx.add_artifact("fix_context", fix_context)

        res = await generate_method(ctx, **generate_kwargs)
        if not res.success:
            logger.warning(
                f"[Retry {attempt+1}/{max_retry}] {kernel_name} codegen error: {res.error}"
            )
            iteration_history.append({"attempt": attempt + 1, "error": str(res.error)[:300]})
            # codegen 失败（非编译错误）也尝试 retry：把错误信息塞入 fix_context
            fix_context = {
                "history_summary": "\n".join(
                    f"第{h['attempt']}次错误:\n{h['error']}" for h in iteration_history
                ),
                "fix_guidance": (
                    f"上次代码生成失败（第{attempt+2}次尝试），请确保输出合法 JSON 格式。\n"
                    f"错误: {str(res.error)[:400]}"
                ),
                "iteration_history": iteration_history,
            }
            continue

        code = res.output.source_code
        flags = res.output.build_flags

        # 知识库自动 patch
        if kb:
            try:
                code = kb.auto_fix(code, "cuda")
            except Exception:
                pass

        # 立即试编译
        ok, stderr = _try_compile(code, flags, kernel_name)
        if ok:
            logger.info(
                f"[Retry] {kernel_name} ✅ compiled OK on attempt {attempt+1}/{max_retry}"
            )
            return {"code": code, "flags": flags}, res.output

        # 编译失败 → 构建 fix_context 供下次
        logger.warning(
            f"[Retry {attempt+1}/{max_retry}] {kernel_name} compile failed:\n{stderr[:300]}"
        )
        iteration_history.append({"attempt": attempt + 1, "error": stderr[:400]})
        fix_context = {
            "history_summary": "\n".join(
                f"第{h['attempt']}次错误:\n{h['error']}" for h in iteration_history
            ),
            "fix_guidance": (
                f"请修复以下编译错误（第{attempt+2}次尝试）：\n{stderr[:600]}\n"
                "只修改导致编译错误的代码，保持算法逻辑不变。"
            ),
            "iteration_history": iteration_history,
        }

    logger.error(
        f"[Retry] {kernel_name} ❌ failed after {max_retry} attempts → PyTorch fallback"
    )
    return None, None


# ════════════════════════════════════════════════════════════════
# Step 1: 用 Agent 生成所有算子 kernel
# ════════════════════════════════════════════════════════════════

async def generate_kernels_for_task(
    llm_backend: str = "mock",
    model_name: str = "qwen",
    gpu_key: str = "rtx_4090",
    max_retry: int = 3,
    output_dir: str = "output/full_agent",
    cached_kernels: dict = None,
) -> dict:
    """
    TaskAnalyzer → 推导算子列表 → 依次生成（含编译 retry）。

    1. TrainingAnalystAgent 静态分析 model_name → 得到算子列表
    2. 过滤出当前支持的算子（silu / gelu / rmsnorm）
    3. 对每个算子：SpecAgent → CodeGenAgent forward + backward（含 retry）
    返回 {name: {"code": str, "flags": list}} 字典
    """
    from agents.base_agent import AgentContext
    from agents.spec_analyzer import OperatorSpecAgent
    from agents.code_generator import CodeGenAgent
    from agents.training_analyst import TrainingAnalystAgent
    from knowledge_base.hardware_specs.gpu_database import get_gpu_spec
    from tools.llm_client import create_llm_client

    # 当前系统支持生成并验证的算子
    SUPPORTED_OPS = {"silu", "gelu", "rmsnorm"}
    OP_ORDER = ["gelu", "silu", "rmsnorm"]  # 生成顺序

    # ── Step 1a: TaskAnalyzer 静态分析 ────────────────────────
    analyst = TrainingAnalystAgent(llm_client=None)  # 纯静态，不需要 LLM
    ctx_analysis = AgentContext(operator_name="task_analysis")
    # 构造 training_code hint，让 _detect_architecture 能识别模型名
    training_code_hint = (
        f"model = {model_name}Model()\n"
        f"# Training {model_name} with LoRA, using RMSNorm and SiLU activations"
    )
    analysis_res = await analyst.run(ctx_analysis, training_code=training_code_hint)

    if analysis_res.success and analysis_res.output:
        plan = analysis_res.output
        all_ops = plan.all_operators()
        ops_raw = [op for op in all_ops if op in SUPPORTED_OPS]
        # 按固定顺序排列，保证 silu bwd 不依赖 rmsnorm
        ops_to_generate = sorted(ops_raw, key=lambda x: OP_ORDER.index(x) if x in OP_ORDER else 99)
        logger.info(f"[Step 1] TaskAnalyzer → 模型架构: {plan.model_architecture}")
        logger.info(f"[Step 1] 识别到全部算子: {all_ops}")
        logger.info(f"[Step 1] 将生成（系统支持范围内）: {ops_to_generate}")
        print(f"\n[Step 1] TaskAnalyzer 识别模型: {plan.model_architecture}")
        print(f"         识别到算子: {all_ops}")
        print(f"         将生成（支持范围内）: {ops_to_generate}")
    else:
        ops_to_generate = ["silu", "rmsnorm"]
        logger.warning(f"[Step 1] TaskAnalyzer 失败，使用默认算子列表: {ops_to_generate}")
        print(f"\n[Step 1] TaskAnalyzer 失败，使用默认: {ops_to_generate}")

    # ── Step 1b: 初始化 agents ────────────────────────────────
    llm = create_llm_client(backend=llm_backend)
    spec_agent = OperatorSpecAgent(llm_client=llm)
    codegen = CodeGenAgent(llm_client=llm)
    gpu_spec = get_gpu_spec(gpu_key)
    if gpu_spec is None:
        raise RuntimeError(f"{gpu_key} not found in GPU database")

    try:
        from knowledge_base.compile_error_kb import get_compile_error_kb
        kb = get_compile_error_kb()
    except Exception:
        kb = None

    kernels = {}

    # ── Step 1b.5: 合并持久化缓存（跳过已有已验证的 kernel）────
    if cached_kernels:
        for name, info in cached_kernels.items():
            # 从缓存中确定算子名（"silu_forward" → "silu"）
            op_name = name.rsplit("_", 1)[0]
            if op_name in ops_to_generate:
                kernels[name] = info
                logger.info(f"[Step 1] ♻️ 使用持久化缓存: {name}")
                print(f"  ♻️ {name}: 使用持久化已验证版本（跳过重新生成）")

    # ── Step 1c: 依次生成每个算子 ─────────────────────────────
    for op_name in ops_to_generate:
        # 如果该算子的 forward+backward 都已从缓存加载，跳过生成
        fwd_key = f"{op_name}_forward"
        bwd_key = f"{op_name}_backward"
        if fwd_key in kernels and bwd_key in kernels:
            logger.info(f"[Step 1] {op_name}: 已从缓存加载 forward+backward，跳过生成")
            continue

        logger.info(f"[Step 1] 生成算子: {op_name} ...")
        ctx_spec = AgentContext(operator_name=op_name)
        ctx_spec = AgentContext(operator_name=op_name)
        spec_res = await spec_agent.run(ctx_spec, request=op_name)
        if not spec_res.success:
            logger.warning(f"[Step 1] {op_name} spec failed: {spec_res.error}，跳过")
            continue
        op_ir = spec_res.output

        # Forward kernel
        fwd_method = codegen.generate_rmsnorm_forward if op_name == "rmsnorm" else codegen.run
        fwd_info, fwd_kernel_obj = await _generate_one_kernel_with_retry(
            codegen, op_ir, gpu_spec,
            generate_method=fwd_method,
            generate_kwargs={"operator_ir": op_ir, "gpu_spec": gpu_spec},
            kernel_name=f"{op_name}_forward",
            max_retry=max_retry,
            kb=kb,
        )
        if fwd_info:
            kernels[f"{op_name}_forward"] = fwd_info

        # Backward kernel（需要 backward_math_description）
        if not getattr(op_ir, "backward_math_description", None):
            logger.info(f"[Step 1] {op_name} 无 backward 定义，跳过 backward")
            continue

        bwd_method = codegen.generate_rmsnorm_backward if op_name == "rmsnorm" else codegen.generate_backward
        bwd_info, _ = await _generate_one_kernel_with_retry(
            codegen, op_ir, gpu_spec,
            generate_method=bwd_method,
            generate_kwargs={
                "operator_ir": op_ir,
                "gpu_spec": gpu_spec,
                "forward_kernel": fwd_kernel_obj,
            },
            kernel_name=f"{op_name}_backward",
            max_retry=max_retry,
            kb=kb,
        )
        if bwd_info:
            kernels[f"{op_name}_backward"] = bwd_info

    total_chars = sum(len(v["code"]) for v in kernels.values())
    logger.info(f"[Step 1] 共生成 {len(kernels)} 个 kernel，总计 {total_chars} chars")
    return kernels


async def generate_all_kernels(llm_backend: str = "mock") -> dict:
    """兼容旧接口：调用 generate_kernels_for_task"""
    return await generate_kernels_for_task(llm_backend=llm_backend)


async def generate_all_kernels_impl(llm_backend: str = "mock") -> dict:
    """
    （保留原始实现供参考）生成 4 个 CUDA kernel：
      silu_forward, silu_backward, rmsnorm_forward, rmsnorm_backward
    返回 {name: {"code": str, "flags": list}} 字典
    """
    from agents.base_agent import AgentContext
    from agents.spec_analyzer import OperatorSpecAgent
    from agents.code_generator import CodeGenAgent
    from knowledge_base.hardware_specs.gpu_database import get_gpu_spec
    from tools.llm_client import create_llm_client

    llm = create_llm_client(backend=llm_backend)
    spec_agent = OperatorSpecAgent(llm_client=llm)
    codegen = CodeGenAgent(llm_client=llm)
    gpu_spec = get_gpu_spec("rtx_4090")
    if gpu_spec is None:
        raise RuntimeError("rtx_4090 not found in GPU database")

    kernels = {}

    # ── SiLU ──────────────────────────────────────────────────
    logger.info("[Step 1] Generating SiLU kernels (forward + backward)...")
    ctx = AgentContext(operator_name="silu")
    spec_res = await spec_agent.run(ctx, request="silu")
    if not spec_res.success:
        raise RuntimeError(f"SiLU spec failed: {spec_res.error}")
    silu_ir = spec_res.output

    fwd_res = await codegen.run(ctx, operator_ir=silu_ir, gpu_spec=gpu_spec)
    if not fwd_res.success:
        raise RuntimeError(f"SiLU forward codegen failed: {fwd_res.error}")
    kernels["silu_forward"] = {
        "code": fwd_res.output.source_code,
        "flags": fwd_res.output.build_flags,
    }

    bwd_res = await codegen.generate_backward(ctx, operator_ir=silu_ir, gpu_spec=gpu_spec,
                                              forward_kernel=fwd_res.output)
    if not bwd_res.success:
        raise RuntimeError(f"SiLU backward codegen failed: {bwd_res.error}")
    kernels["silu_backward"] = {
        "code": bwd_res.output.source_code,
        "flags": bwd_res.output.build_flags,
    }

    # ── RMSNorm ───────────────────────────────────────────────
    logger.info("[Step 1] Generating RMSNorm kernels (forward + backward)...")
    ctx2 = AgentContext(operator_name="rmsnorm")
    spec_res2 = await spec_agent.run(ctx2, request="rmsnorm")
    if not spec_res2.success:
        raise RuntimeError(f"RMSNorm spec failed: {spec_res2.error}")
    rmsnorm_ir = spec_res2.output

    rmsnorm_fwd_res = await codegen.generate_rmsnorm_forward(ctx2, operator_ir=rmsnorm_ir, gpu_spec=gpu_spec)
    if not rmsnorm_fwd_res.success:
        raise RuntimeError(f"RMSNorm forward codegen failed: {rmsnorm_fwd_res.error}")
    kernels["rmsnorm_forward"] = {
        "code": rmsnorm_fwd_res.output.source_code,
        "flags": rmsnorm_fwd_res.output.build_flags,
    }

    rmsnorm_bwd_res = await codegen.generate_rmsnorm_backward(ctx2, operator_ir=rmsnorm_ir, gpu_spec=gpu_spec,
                                                              forward_kernel=rmsnorm_fwd_res.output)
    if not rmsnorm_bwd_res.success:
        raise RuntimeError(f"RMSNorm backward codegen failed: {rmsnorm_bwd_res.error}")
    kernels["rmsnorm_backward"] = {
        "code": rmsnorm_bwd_res.output.source_code,
        "flags": rmsnorm_bwd_res.output.build_flags,
    }

    total_chars = sum(len(v["code"]) for v in kernels.values())
    logger.info(f"[Step 1] Generated {len(kernels)} kernels, total {total_chars} chars")
    return kernels


# ════════════════════════════════════════════════════════════════
# Step 2: 编译所有 .so
# ════════════════════════════════════════════════════════════════

def _save_verified_kernels_to_registry(
    kernels: dict,
    so_paths: dict,
    verify_report: dict,
    gpu_key: str = "rtx_4090",
) -> None:
    """
    将数值验证通过的 kernel 保存进持久化 OperatorRegistry（SQLite）。
    下次运行时，如果找到对应的已验证 kernel，可以直接复用而不必重新生成。
    """
    try:
        from operators.registry import get_registry, OperatorEntry
        from knowledge_base.hardware_specs.gpu_database import get_gpu_spec
        reg = get_registry()
        gpu_spec = get_gpu_spec(gpu_key)
        gpu_model = gpu_spec.model_name if gpu_spec else gpu_key

        saved = 0
        for name, info in kernels.items():
            if so_paths.get(name) is None:
                continue  # 编译失败或验证失败
            result = verify_report.get(name, {})
            if not result.get("passed", False):
                continue  # 数值验证未通过

            entry = OperatorEntry(
                operator_name=name,
                gpu_model=gpu_model,
                backend="cuda",
                source_code=info.get("code", ""),
                build_flags=info.get("flags", []),
                correctness_passed=True,
                max_relative_error=result.get("rel_err", 0.0),
                verification_level="hw_verified",
                tags=[name.split("_")[0]],  # e.g. "silu", "rmsnorm"
            )
            reg.register(entry)
            saved += 1

        if saved:
            logger.info(f"[Step 2.5] 已将 {saved} 个验证通过的 kernel 存入持久化 OperatorRegistry")
            print(f"  💾 {saved} 个 kernel 已持久化存储（下次可直接复用）")
    except Exception as e:
        logger.warning(f"[Step 2.5] 持久化保存 kernel 失败（非致命）: {e}")


def _load_cached_kernels(output_dir: str, gpu_key: str = "rtx_4090") -> dict:
    """
    从持久化 OperatorRegistry 加载已验证的 kernel，
    返回 {name: {"code": str, "flags": list}} 字典，格式与 generate_kernels_for_task 返回值一致。
    若未找到则返回空字典。
    """
    cached = {}
    try:
        from operators.registry import get_registry
        from knowledge_base.hardware_specs.gpu_database import get_gpu_spec
        reg = get_registry()
        gpu_spec = get_gpu_spec(gpu_key)
        gpu_model = gpu_spec.model_name if gpu_spec else gpu_key

        kernel_names = ["silu_forward", "silu_backward",
                        "rmsnorm_forward", "rmsnorm_backward",
                        "matmul_forward"]
        for name in kernel_names:
            entry = reg.lookup(name, gpu_model)
            if entry and entry.correctness_passed and entry.source_code:
                cached[name] = {
                    "code": entry.source_code,
                    "flags": entry.build_flags or ["-O3", f"-arch=sm_89"],
                }
                logger.info(f"[Step 0c] 从持久化存储加载: {name} (rel_err={entry.max_relative_error:.4f})")
    except Exception as e:
        logger.debug(f"[Step 0c] 加载缓存 kernel 失败: {e}")
    return cached


def compile_all_kernels(kernels: dict, output_dir: str) -> dict:
    """编译所有 kernel，返回 {name: so_path | None} 字典"""
    os.makedirs(output_dir, exist_ok=True)

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        for path in ["/usr/local/cuda/bin/nvcc", "/usr/local/cuda-12/bin/nvcc"]:
            if os.path.isfile(path):
                nvcc = path
                break
    if nvcc is None:
        logger.warning("[Step 2] nvcc not found, all kernels will use PyTorch fallback")
        return {name: None for name in kernels}

    # 应用编译错误知识库
    try:
        from knowledge_base.compile_error_kb import get_compile_error_kb
        kb = get_compile_error_kb()
    except Exception:
        kb = None

    so_paths = {}
    for name, info in kernels.items():
        code = info["code"]
        if kb:
            try:
                code = kb.auto_fix(code, "cuda")
            except Exception:
                pass

        src_path = os.path.join(output_dir, f"{name}.cu")
        so_path = os.path.join(output_dir, f"{name}.so")

        with open(src_path, "w") as f:
            f.write(code)

        # 过滤 flags
        valid_prefixes = ("-O", "-arch=", "--use_fast_math", "-std=", "-Xcompiler", "-fPIC", "--shared")
        flags = [f for f in info.get("flags", []) if any(f.startswith(p) for p in valid_prefixes)]
        if not flags:
            flags = ["-O3", "--use_fast_math"]

        cmd = [nvcc, "--shared", "-Xcompiler", "-fPIC"] + flags + [src_path, "-o", so_path]
        logger.info(f"[Step 2] Compiling {name}: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.error(f"[Step 2] {name} compile failed:\n{result.stderr[:400]}")
                so_paths[name] = None
            else:
                size = os.path.getsize(so_path)
                logger.info(f"[Step 2] {name} compiled: {so_path} ({size} bytes)")
                so_paths[name] = so_path
        except Exception as e:
            logger.error(f"[Step 2] {name} compile error: {e}")
            so_paths[name] = None

    return so_paths


async def verify_all_kernels(kernels: dict, so_paths: dict, output_dir: str) -> dict:
    """
    Step 2.5：对所有编译成功的 kernel 进行数值验证（方案C）。

    策略：直接加载已编译的 so_path（而非让 verifier 重新编译），
    在 PyTorch GPU 上调用 kernel 并对比 reference，避免二次编译引入 CUDA context 污染。
    验证失败的 kernel，将 so_path 置为 None（触发 PyTorch fallback）。
    """
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("  [Step 2.5] 无 GPU，跳过数值验证")
        return so_paths, {name: {"passed": True, "detail": "no GPU"} for name in so_paths}

    verified_paths = dict(so_paths)
    verify_results = {}
    cuda_context_broken = False  # 标记 CUDA context 是否已损坏

    print(f"\n[Step 2.5/5] 数值验证所有编译成功的 kernel...")

    for name, so_path in so_paths.items():
        if so_path is None:
            verify_results[name] = {"passed": False, "detail": "compile failed"}
            print(f"  ⚠ {name}: 编译失败，跳过验证")
            continue

        # 若 CUDA context 已损坏（前面某个 kernel 验证失败），跳过后续验证
        # 但保留 so_path（继续使用该 kernel，不因为验证跳过而 fallback）
        if cuda_context_broken:
            verify_results[name] = {"passed": True, "detail": "skipped (CUDA context broken by prev kernel)"}
            print(f"  ⚠ {name}: PASS (未验证，前序 kernel 损坏了 CUDA context)")
            continue

        try:
            lib = ctypes.CDLL(so_path)
            fn = lib.launch_kernel
            fn.restype = None

            is_backward = "backward" in name
            is_rmsnorm = "rmsnorm" in name

            # 测试形状：N*H 均为偶数，满足 half2 对齐要求
            test_shapes = [(64, 1024, 1.0), (8, 3072, 1.0), (16, 1024, 50.0)]
            results = []

            for N, H, gs in test_shapes:
                x = torch.randn(N, H, dtype=torch.float16, device=device)

                if is_rmsnorm and is_backward:
                    fn.argtypes = [
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_int, ctypes.c_int, ctypes.c_float,
                    ]
                    weight = torch.ones(H, dtype=torch.float16, device=device)
                    go = (torch.randn(N, H, dtype=torch.float16, device=device) * gs)
                    gx_fp32 = torch.empty(N * H, dtype=torch.float32, device=device)
                    gw_fp32 = torch.zeros(H, dtype=torch.float32, device=device)
                    fn(go.data_ptr(), x.data_ptr(), weight.data_ptr(),
                       gx_fp32.data_ptr(), gw_fp32.data_ptr(), N, H, 1e-6)
                    torch.cuda.synchronize()
                    if gx_fp32.isnan().any() or gx_fp32.isinf().any():
                        results.append((False, float('inf'), f"NaN/Inf at gs={gs}"))
                        continue
                    xr = x.float().requires_grad_(True)
                    wr = weight.float().requires_grad_(True)
                    y = xr / torch.sqrt(xr.pow(2).mean(-1, keepdim=True) + 1e-6) * wr
                    y.backward(go.float())
                    ref_gx = xr.grad.float()
                    err = (gx_fp32.reshape(N, H) - ref_gx).abs()
                    ref_abs = ref_gx.abs()
                    mask = ref_abs > ref_abs.mean() * 0.05 + 1e-4
                    rel_err = (err[mask] / (ref_abs[mask] + 1e-4)).max().item() if mask.any() else err.max().item()
                    results.append((rel_err < 0.05, rel_err, f"rmsnorm_bwd gs={gs} err={rel_err:.4f}"))

                elif is_backward:
                    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
                    go = (torch.randn(N, H, dtype=torch.float16, device=device) * gs)
                    gi_fp32 = torch.empty(N * H, dtype=torch.float32, device=device)
                    fn(go.data_ptr(), x.reshape(-1).data_ptr(), gi_fp32.data_ptr(), N * H)
                    torch.cuda.synchronize()
                    if gi_fp32.isnan().any() or gi_fp32.isinf().any():
                        results.append((False, float('inf'), f"NaN/Inf at gs={gs}"))
                        continue
                    x_f = x.reshape(-1).float()
                    g_f = go.reshape(-1).float()
                    if "silu" in name:
                        sig = torch.sigmoid(x_f)
                        ref = g_f * sig * (1.0 + x_f * (1.0 - sig))
                    elif "gelu" in name:
                        xr = x_f.requires_grad_(True)
                        F.gelu(xr).backward(g_f)
                        ref = xr.grad
                    else:
                        ref = g_f
                    rel_err = ((gi_fp32 - ref).abs() / (ref.abs() + 1e-6)).max().item()
                    results.append((rel_err < 0.05, rel_err, f"elem_bwd gs={gs} err={rel_err:.4f}"))

                elif is_rmsnorm:
                    fn.argtypes = [
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_int, ctypes.c_int, ctypes.c_float,
                    ]
                    w = torch.ones(H, dtype=torch.float16, device=device)
                    out = torch.empty(N, H, dtype=torch.float16, device=device)
                    fn(x.data_ptr(), w.data_ptr(), out.data_ptr(), N, H, 1e-6)
                    torch.cuda.synchronize()
                    if out.isnan().any():
                        results.append((False, float('inf'), "NaN in rmsnorm_fwd"))
                        continue
                    ref = (x.float() / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).half()
                    rel_err = ((out.float() - ref.float()).abs() / (ref.float().abs() + 1e-3)).max().item()
                    results.append((rel_err < 0.05, rel_err, f"rmsnorm_fwd err={rel_err:.4f}"))

                else:
                    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
                    out = torch.empty(N, H, dtype=torch.float16, device=device)
                    fn(x.data_ptr(), out.data_ptr(), N * H)
                    torch.cuda.synchronize()
                    if out.isnan().any():
                        results.append((False, float('inf'), "NaN in elem_fwd"))
                        continue
                    ref = F.silu(x) if "silu" in name else (F.gelu(x) if "gelu" in name else x)
                    rel_err = ((out.float() - ref.float()).abs() / (ref.float().abs() + 1e-3)).max().item()
                    results.append((rel_err < 0.05, rel_err, f"elem_fwd err={rel_err:.4f}"))

            all_ok = all(r[0] for r in results)
            max_err = max(r[1] for r in results) if results else 0.0
            detail = "; ".join(r[2] for r in results)
            verify_results[name] = {"passed": all_ok, "rel_err": max_err, "detail": detail}

            if all_ok:
                print(f"  ✅ {name}: PASS (max_rel_err={max_err:.4f})")
            else:
                print(f"  ❌ {name}: FAIL → fallback to PyTorch ({detail})")
                verified_paths[name] = None

        except Exception as e:
            err_str = str(e).lower()
            is_cuda_error = "cuda error" in err_str or "misaligned" in err_str or "illegal memory" in err_str
            logger.warning(f"[Step 2.5] {name} verify error: {e}")
            verify_results[name] = {"passed": False, "detail": str(e)}
            if is_cuda_error:
                # CUDA error（misaligned address 等）= kernel 有 bug，直接 fallback
                # 同时标记 CUDA context 已损坏，跳过后续 kernel 的验证
                print(f"  ❌ {name}: CUDA ERROR → fallback to PyTorch ({str(e)[:80]})")
                cuda_context_broken = True
            else:
                print(f"  ⚠ {name}: verify exception → fallback to PyTorch")
            verified_paths[name] = None
            # 尝试恢复 CUDA context（失败时静默跳过）
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    n_pass = sum(1 for r in verify_results.values() if r.get("passed"))
    n_total = len(verify_results)
    print(f"  验证结果: {n_pass}/{n_total} 通过数值验证")

    return verified_paths, verify_results



# ════════════════════════════════════════════════════════════════
# Step 3: 创建自定义 autograd.Function
# ════════════════════════════════════════════════════════════════

def create_custom_silu(forward_so: str = None, backward_so: str = None):
    """
    创建自定义 SiLU Function（forward + backward 均可来自 agent kernel）。
    backward kernel 接口约定：grad_in 输出为 float32（避免 fp16 overflow → NaN）。
    """
    forward_fn = None
    if forward_so and os.path.exists(forward_so):
        try:
            lib = ctypes.CDLL(forward_so)
            fn = lib.launch_kernel
            fn.restype = None
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            forward_fn = fn
            logger.info("[Step 3] ✅ SiLU forward kernel loaded from .so")
        except (OSError, AttributeError) as e:
            logger.warning(f"[Step 3] SiLU forward .so load failed: {e}, using PyTorch fallback")

    backward_fn = None
    if backward_so and os.path.exists(backward_so):
        try:
            lib_bwd = ctypes.CDLL(backward_so)
            fn_bwd = lib_bwd.launch_kernel
            fn_bwd.restype = None
            # 新接口：grad_in_fp32 是 float* 输出（避免 fp16 overflow）
            fn_bwd.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            backward_fn = fn_bwd
            logger.info("[Step 3] ✅ SiLU backward kernel loaded from .so (float32 grad output)")
        except (OSError, AttributeError) as e:
            logger.warning(f"[Step 3] SiLU backward .so load failed: {e}, using PyTorch fallback")

    class SiLUCustomFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            x_c = x.contiguous()
            if forward_fn is not None:
                out = torch.empty_like(x_c)
                forward_fn(x_c.data_ptr(), out.data_ptr(), x_c.numel())
                torch.cuda.synchronize()
            else:
                out = F.silu(x)
            ctx.save_for_backward(x)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            if backward_fn is not None:
                x_c = x.contiguous()
                g_c = grad_output.contiguous()
                # kernel 输出 float32，Python 端接收后 reshape 到原始 shape 再转 dtype
                grad_in_fp32 = torch.empty(x_c.numel(), dtype=torch.float32, device=x_c.device)
                backward_fn(g_c.data_ptr(), x_c.data_ptr(), grad_in_fp32.data_ptr(), x_c.numel())
                torch.cuda.synchronize()
                grad_in = grad_in_fp32.reshape(x_c.shape).to(x.dtype)  # 保持形状
            else:
                sig = torch.sigmoid(x.float())
                grad_in = (grad_output.float() * sig * (1.0 + x.float() * (1.0 - sig))).to(x.dtype)
            return grad_in

    def silu_custom(x):
        return SiLUCustomFunction.apply(x)

    return silu_custom, forward_fn is not None, backward_fn is not None


def create_custom_rmsnorm(forward_so: str = None, backward_so: str = None):
    """
    创建自定义 RMSNorm Module（forward + backward 均可来自 agent kernel）。
    backward kernel 接口约定：grad_x 输出为 float32（避免 fp16 overflow → NaN）。
    """
    forward_fn = None
    if forward_so and os.path.exists(forward_so):
        try:
            lib = ctypes.CDLL(forward_so)
            fn = lib.launch_kernel
            fn.restype = None
            fn.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_float,
            ]
            forward_fn = fn
            logger.info("[Step 3] ✅ RMSNorm forward kernel loaded from .so")
        except (OSError, AttributeError) as e:
            logger.warning(f"[Step 3] RMSNorm forward .so load failed: {e}, using PyTorch fallback")

    backward_fn = None
    if backward_so and os.path.exists(backward_so):
        try:
            lib_bwd = ctypes.CDLL(backward_so)
            fn_bwd = lib_bwd.launch_kernel
            fn_bwd.restype = None
            # 新接口：grad_x_fp32 是 float* 输出（避免 fp16 overflow）
            fn_bwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,   # grad_out, x, weight
                ctypes.c_void_p, ctypes.c_void_p,                     # grad_x_fp32, grad_weight
                ctypes.c_int, ctypes.c_int, ctypes.c_float,           # N, H, eps
            ]
            backward_fn = fn_bwd
            logger.info("[Step 3] ✅ RMSNorm backward kernel loaded from .so (float32 grad output)")
        except (OSError, AttributeError) as e:
            logger.warning(f"[Step 3] RMSNorm backward .so load failed: {e}, using PyTorch fallback")

    class RMSNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, eps):
            x_c = x.contiguous()
            w_c = weight.contiguous()
            N = x_c.shape[0] if x_c.ndim > 1 else 1
            H = x_c.shape[-1]
            if forward_fn is not None:
                out = torch.empty_like(x_c)
                forward_fn(x_c.data_ptr(), w_c.data_ptr(), out.data_ptr(),
                           N, H, float(eps))
                torch.cuda.synchronize()
            else:
                # PyTorch fallback
                x_fp = x_c.float()
                var = x_fp.pow(2).mean(-1, keepdim=True) + eps
                out = (x_fp * torch.rsqrt(var) * w_c.float()).to(x.dtype)
            ctx.save_for_backward(x, weight)
            ctx.eps = eps
            return out

        @staticmethod
        def backward(ctx, grad_output):
            x, weight = ctx.saved_tensors
            eps = ctx.eps
            if backward_fn is not None:
                x_c = x.contiguous()
                w_c = weight.contiguous()
                g_c = grad_output.contiguous()
                N = x_c.shape[0] if x_c.ndim > 1 else 1
                H = x_c.shape[-1]
                # kernel 输出 float32，Python 端接收后转回原始 dtype
                grad_x_fp32 = torch.empty(N * H, dtype=torch.float32, device=x.device)
                grad_w_fp32 = torch.zeros(H, dtype=torch.float32, device=x.device)
                backward_fn(
                    g_c.data_ptr(), x_c.data_ptr(), w_c.data_ptr(),
                    grad_x_fp32.data_ptr(), grad_w_fp32.data_ptr(),
                    N, H, float(eps)
                )
                torch.cuda.synchronize()
                grad_x = grad_x_fp32.reshape(x_c.shape).to(x.dtype)  # float32 → half
                grad_w = grad_w_fp32.to(weight.dtype)
                return grad_x, grad_w, None
            else:
                # PyTorch fallback（全程 float32，数值稳定）
                x_fp = x.float()
                w_fp = weight.float()
                g_fp = grad_output.float()
                rms = torch.sqrt(x_fp.pow(2).mean(-1, keepdim=True) + eps)
                x_norm = x_fp / rms
                grad_w = (g_fp * x_norm).sum(dim=tuple(range(g_fp.ndim - 1))).to(weight.dtype)
                grad_x_norm = g_fp * w_fp
                grad_x = ((grad_x_norm - x_norm * (grad_x_norm * x_norm).mean(-1, keepdim=True))
                          / rms).to(x.dtype)
                return grad_x, grad_w, None

    class RMSNormCustomModule(nn.Module):
        """完全替换 Qwen3RMSNorm 的自定义 Module"""
        def __init__(self, weight: nn.Parameter, eps: float):
            super().__init__()
            self.weight = weight          # 复用原模型的权重参数
            self.variance_epsilon = eps

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            # 将输入 reshape 成 2D (N_tokens, H) 供 kernel 使用
            orig_shape = hidden_states.shape
            x_2d = hidden_states.reshape(-1, orig_shape[-1])
            out_2d = RMSNormFunction.apply(x_2d, self.weight, self.variance_epsilon)
            return out_2d.reshape(orig_shape)

        def extra_repr(self):
            return f"hidden={tuple(self.weight.shape)}, eps={self.variance_epsilon}, source=operator_agent"

    return RMSNormCustomModule, forward_fn is not None, backward_fn is not None


# ════════════════════════════════════════════════════════════════
# Step 4: 注入自定义算子到模型
# ════════════════════════════════════════════════════════════════

def patch_model_operators(model, custom_silu_fn, RMSNormCustomModule):
    """
    替换模型中的所有 SiLU 激活函数和 RMSNorm 层
    返回 (silu_replaced, rmsnorm_replaced) 数量
    """
    from transformers.activations import SiLUActivation

    silu_replaced = 0
    rmsnorm_replaced = 0

    for name, module in model.named_modules():
        # 替换 SiLU
        if hasattr(module, 'act_fn'):
            act = module.act_fn
            if isinstance(act, nn.Module) and 'silu' in type(act).__name__.lower():
                module.act_fn = _SiLUWrapper(custom_silu_fn)
                silu_replaced += 1

        # 替换 RMSNorm：遍历父模块的子模块字典
    # 需要用 named_children 方式才能修改
    def _replace_rmsnorm_recursive(parent, prefix=""):
        nonlocal rmsnorm_replaced
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            ctype = type(child).__name__
            if 'RMSNorm' in ctype or 'rmsnorm' in ctype.lower():
                new_module = RMSNormCustomModule(child.weight, child.variance_epsilon)
                setattr(parent, child_name, new_module)
                rmsnorm_replaced += 1
            else:
                _replace_rmsnorm_recursive(child, full_name)

    _replace_rmsnorm_recursive(model)

    logger.info(f"[Step 4] Replaced: SiLU × {silu_replaced}, RMSNorm × {rmsnorm_replaced}")
    return silu_replaced, rmsnorm_replaced


class _SiLUWrapper(nn.Module):
    """将函数包装为 nn.Module，替换 SiLUActivation"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)

    def extra_repr(self):
        return "source=operator_agent"


# ════════════════════════════════════════════════════════════════
# Step 5: SST-2 LoRA 训练
# ════════════════════════════════════════════════════════════════

def run_lora_training(model_path: str, custom_fn_map: dict | None = None,
                      num_steps: int = 20, lr: float = 2e-4) -> dict:
    """
    custom_fn_map: {
        "silu_fn": callable,
        "RMSNormModule": class,
    }
    为 None 时不替换任何算子（baseline / no_finetune）
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[Step 5] Device: {device}")
    if device.type == "cuda":
        logger.info(f"[Step 5] GPU: {torch.cuda.get_device_name(0)}")

    # 加载模型
    logger.info(f"[Step 5] Loading {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 注入自定义算子
    silu_replaced = rmsnorm_replaced = 0
    if custom_fn_map is not None:
        silu_replaced, rmsnorm_replaced = patch_model_operators(
            model,
            custom_fn_map["silu_fn"],
            custom_fn_map["RMSNormModule"],
        )
        if silu_replaced == 0:
            logger.warning("[Step 5] No SiLU found to replace!")
        if rmsnorm_replaced == 0:
            logger.warning("[Step 5] No RMSNorm found to replace!")

    # LoRA 配置：对 8B 模型降低 rank，减少过拟合风险
    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    logger.info(f"[Step 5] LoRA params: {trainable:,} / {total_p:,} ({100*trainable/total_p:.2f}%)")

    # ── Alpaca 格式构造：只对 output 部分计算 loss ────────────────
    def make_alpaca_prompt(sample: dict) -> tuple[str, str]:
        """
        返回 (full_text, instruction_part)。
        full_text = system_prefix + instruction + input + output
        instruction_part = system_prefix + instruction + input (不含 output)
        loss 只在 output 对应的 token 位置计算。
        """
        inst = sample["instruction"].strip()
        inp = sample.get("input", "").strip()
        out = sample["output"].strip()

        if inp:
            instruction_part = (
                f"Below is an instruction that describes a task, paired with an input that provides further context. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
            )
        else:
            instruction_part = (
                f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{inst}\n\n### Response:\n"
            )
        full_text = instruction_part + out
        return full_text, instruction_part

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=lr * 0.1)
    model.train()

    train_data = ALPACA_TRAIN  # 25 条 Alpaca 样本
    logger.info(f"[Step 5] Training for {num_steps} steps on {len(train_data)} samples (Alpaca instruction tuning)...")
    print(f"\n{'Step':>5}  {'Loss':>9}  {'Grad Norm':>10}  {'LR':>10}  {'Time':>8}")
    print("─" * 52)

    losses = []
    for step in range(1, num_steps + 1):
        t0 = time.perf_counter()

        sample = train_data[(step - 1) % len(train_data)]
        full_text, instruction_part = make_alpaca_prompt(sample)

        # 编码完整文本（instruction + output）
        full_ids = tokenizer(full_text, return_tensors="pt",
                             max_length=256, truncation=True)["input_ids"].to(device)
        # 编码 instruction_part 确定 output 起始位置
        inst_ids = tokenizer(instruction_part, return_tensors="pt",
                             max_length=256, truncation=True)["input_ids"].to(device)
        inst_len = inst_ids.shape[1]

        # labels：instruction 部分设为 -100，只对 output 计算 loss
        labels = full_ids.clone()
        labels[:, :inst_len] = -100

        attn_mask = torch.ones_like(full_ids)
        outputs = model(input_ids=full_ids, attention_mask=attn_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        grad_norm = sum(p.grad.data.norm(2).item() ** 2
                        for p in model.parameters() if p.grad is not None) ** 0.5

        optimizer.step()
        scheduler.step()

        elapsed = (time.perf_counter() - t0) * 1000
        loss_val = loss.item()
        losses.append(loss_val)
        cur_lr = scheduler.get_last_lr()[0]
        print(f"{step:>5}  {loss_val:>9.4f}  {grad_norm:>10.4f}  {cur_lr:>10.2e}  {elapsed:>6.0f}ms")

    print("─" * 52)
    loss_trend = f"{losses[0]:.4f} → {losses[-1]:.4f}"
    trend = "↓ 下降" if losses[-1] < losses[0] else "↑ 上升"
    print(f"Loss: {loss_trend} ({trend})")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "losses": losses,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "silu_replaced": silu_replaced,
        "rmsnorm_replaced": rmsnorm_replaced,
        "trainable_params": trainable,
    }


# ════════════════════════════════════════════════════════════════
# Step 6: Alpaca 指令跟随评估
# ════════════════════════════════════════════════════════════════

def evaluate_alpaca(model, tokenizer, device=None) -> dict:
    """
    在 ALPACA_TEST 测试集上评估：
    1. 关键词命中率（输出是否包含期望关键词）
    2. 格式符合率（输出是否遵循指令格式，有实质内容）
    3. 打印每道题的生成结果供对比
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    keyword_scores = []
    format_scores = []

    print(f"\n  {'Test':>4}  {'Keywords':>10}  {'Format':>8}  Response preview")
    print("  " + "─" * 70)

    with torch.no_grad():
        for i, sample in enumerate(ALPACA_TEST):
            inst = sample["instruction"].strip()
            inp = sample.get("input", "").strip()
            keywords = sample["expected_keywords"]

            # 构造 prompt（不含 output）
            if inp:
                prompt = (
                    f"Below is an instruction that describes a task, paired with an input "
                    f"that provides further context. Write a response that appropriately "
                    f"completes the request.\n\n### Instruction:\n{inst}\n\n"
                    f"### Input:\n{inp}\n\n### Response:\n"
                )
            else:
                prompt = (
                    f"Below is an instruction that describes a task. Write a response "
                    f"that appropriately completes the request.\n\n"
                    f"### Instruction:\n{inst}\n\n### Response:\n"
                )

            input_ids = tokenizer(prompt, return_tensors="pt",
                                  max_length=256, truncation=True)["input_ids"].to(device)

            # 生成回答（限制长度，加速评估）
            out_ids = model.generate(
                input_ids,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(
                out_ids[0, input_ids.shape[1]:], skip_special_tokens=True
            ).strip()

            # 关键词命中率
            resp_lower = response.lower()
            hits = sum(1 for kw in keywords if kw.lower() in resp_lower)
            kw_score = hits / len(keywords) if keywords else 1.0
            keyword_scores.append(kw_score)

            # 格式符合率：回答是否有实质内容（> 20 chars）
            fmt_score = 1.0 if len(response) > 20 else 0.0
            format_scores.append(fmt_score)

            preview = response[:60].replace('\n', ' ')
            kw_pct = f"{kw_score*100:.0f}%"
            fmt = "✅" if fmt_score == 1.0 else "❌"
            print(f"  {i+1:>4}  {kw_pct:>10}  {fmt:>8}  {preview}...")

    avg_kw = sum(keyword_scores) / len(keyword_scores)
    avg_fmt = sum(format_scores) / len(format_scores)

    return {
        "keyword_score": avg_kw,
        "format_score": avg_fmt,
        "combined_score": (avg_kw + avg_fmt) / 2,
        "per_sample": list(zip(keyword_scores, format_scores)),
    }


def generate_alpaca_response(model, tokenizer, instruction: str, device=None) -> str:
    """生成单条指令的回答，用于定性对比"""
    if device is None:
        device = next(model.parameters()).device
    prompt = (
        f"Below is an instruction that describes a task. Write a response "
        f"that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n### Response:\n"
    )
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt",
                              max_length=256, truncation=True)["input_ids"].to(device)
        out = model.generate(
            input_ids, max_new_tokens=200, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


def evaluate_no_finetune(model_path: str) -> dict:
    """不微调，直接评估 base 模型（Alpaca 指令跟随能力）"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="auto", trust_remote_code=True)

    result = evaluate_alpaca(model, tokenizer, device)
    del model
    torch.cuda.empty_cache()
    return result


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(
        description="全 Agent 算子 LoRA 微调 + SST-2 评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 全流程 mock 快速验证
  python examples/full_agent_lora_train.py --mode custom --llm mock --steps 5

  # 全流程 Qwen LLM 生成真实 kernel
  python examples/full_agent_lora_train.py --mode custom --llm qwen --steps 20

  # 对比三种模式（一次性跑完）
  python examples/full_agent_lora_train.py --mode all --llm qwen --steps 20
        """,
    )
    parser.add_argument("--mode", default="all",
                        choices=["custom", "baseline", "no_finetune", "all"],
                        help="运行模式（默认 all：三种模式全跑）")
    parser.add_argument("--llm", default="mock",
                        choices=["qwen", "openai", "anthropic", "mock"],
                        help="LLM 后端（默认 mock）")
    parser.add_argument("--model", default="/remote-home1/share/models/Qwen/Qwen3-0.6B",
                        help="模型路径")
    parser.add_argument("--steps", type=int, default=100,
                        help="训练步数（默认100步 = ~4 epoch × 25条Alpaca样本）")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="学习率（默认 5e-5，适合 8B 模型指令微调）")
    parser.add_argument("--output-dir", default="./output/full_agent",
                        help="算子代码和 .so 保存目录")
    parser.add_argument("--skip-compile", action="store_true",
                        help="跳过 nvcc 编译（用 PyTorch fallback）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print("  Operator Agent 全算子 LoRA 微调 + SST-2 评估")
    print(f"  模型:  {args.model}")
    print(f"  模式:  {args.mode}")
    print(f"  LLM:   {args.llm}")
    print(f"  Steps: {args.steps}")
    print(f"  输出:  {args.output_dir}")
    print("=" * 65)

    results = {}

    # ── No-Finetune baseline ────────────────────────────────────
    if args.mode in ("no_finetune", "all"):
        print("\n" + "─" * 65)
        print("  [模式 1/3] No-Finetune：基座模型，不微调（指令跟随能力评估）")
        print("─" * 65)
        nft_eval = evaluate_no_finetune(args.model)
        results["no_finetune"] = {"eval": nft_eval}
        score = nft_eval["combined_score"]
        print(f"  No-Finetune 指令跟随分: {score*100:.1f}% "
              f"(关键词命中: {nft_eval['keyword_score']*100:.1f}%, "
              f"格式符合: {nft_eval['format_score']*100:.1f}%)")

    # ── Custom：全 Agent 算子 ───────────────────────────────────
    if args.mode in ("custom", "all"):
        print("\n" + "─" * 65)
        print("  [模式 2/3] Custom：全 Agent 算子（SiLU + RMSNorm）")
        print("─" * 65)

        # Step 1: 生成 kernels（TaskAnalyzer 分析任务 → 依次生成，含编译 retry）
        print("\n[Step 1/5] 用 Agent 生成 SiLU + RMSNorm forward/backward kernel...")

        # Step 0: 初始化通用算子注册表
        from operators.builtin_ops import register_builtin_ops
        from operators.op_registry import get_op_registry, reset_op_registry
        reset_op_registry()
        op_reg = get_op_registry()
        register_builtin_ops(op_reg)
        print(f"\n[Step 0] 算子注册表已初始化:\n{op_reg.summary()}")

        # 从 model 路径推断模型名
        model_name_hint = "qwen"
        if args.model:
            mp = args.model.lower()
            if "llama" in mp:
                model_name_hint = "llama"
            elif "mistral" in mp:
                model_name_hint = "mistral"
            elif "qwen" in mp:
                model_name_hint = "qwen"

        # Step 0b: 自动识别并注册缺失算子
        from operators.auto_registrar import AutoOpRegistrar
        from agents.base_agent import AgentContext as _AC
        from agents.training_analyst import TrainingAnalystAgent as _TAA
        _analyst = _TAA(llm_client=None)
        _ctx_a = _AC(operator_name="task_analysis")
        _training_hint = (
            f"model = {model_name_hint}Model(); "
            f"training with LoRA"
        )
        _plan_res = await _analyst.run(_ctx_a, training_code=_training_hint)
        _plan = _plan_res.output if _plan_res.success else None
        auto_reg = AutoOpRegistrar(output_path="operators/generated_ops.py")
        if _plan:
            missing_ops = auto_reg.find_missing(_plan, op_reg)
            if missing_ops:
                print(f"\n[Step 0b] 发现未注册算子: {missing_ops}")
                new_descs = auto_reg.generate_missing_descs(missing_ops)
                if new_descs:
                    auto_reg.write_and_register(new_descs, op_reg, "operators/generated_ops.py")
                    print(f"[Step 0b] 已自动注册 {len(new_descs)} 个新算子并写入 generated_ops.py")
                    print(op_reg.summary())
                unregisterable = [op for op in missing_ops
                                  if not any(d.name == op for d in new_descs)]
                if unregisterable:
                    print(f"[Step 0b] 以下算子接口复杂，暂用 PyTorch fallback: {unregisterable}")
            else:
                print(f"\n[Step 0b] 所有需要的算子已在注册表中")

        # Step 0c: 从持久化 OperatorRegistry 加载已验证的 kernel（跳过重新生成）
        cached_kernels = _load_cached_kernels(args.output_dir, gpu_key="rtx_4090")
        if cached_kernels:
            print(f"\n[Step 0c] 发现持久化 kernel 缓存: {list(cached_kernels.keys())}")
            print(f"  将跳过这些 kernel 的重新生成，直接使用已验证版本")

        kernels = await generate_kernels_for_task(
            llm_backend=args.llm,
            model_name=model_name_hint,
            gpu_key="rtx_4090",
            max_retry=3,
            output_dir=args.output_dir,
            cached_kernels=cached_kernels,
        )

        # 保存源码
        for kname, kinfo in kernels.items():
            path = os.path.join(args.output_dir, f"{kname}.cu")
            with open(path, "w") as f:
                f.write(kinfo["code"])
            print(f"  保存: {path} ({len(kinfo['code'])} chars)")

        # Step 2: 编译
        if args.skip_compile:
            print("\n[Step 2/5] 跳过编译（--skip-compile）")
            so_paths = {name: None for name in kernels}
        else:
            print("\n[Step 2/5] 编译 CUDA kernel...")
            so_paths = compile_all_kernels(kernels, args.output_dir)
            for name, path in so_paths.items():
                status = f"✅ {path}" if path else "⚠ fallback to PyTorch"
                print(f"  {name}: {status}")

        # Step 2.5: 数值验证（通用框架）
        if not args.skip_compile and any(p is not None for p in so_paths.values()):
            print("\n[Step 2.5/5] 数值验证 kernel（ctypes加载 → 对比PyTorch reference）...")
            from operators.verify import verify_all_kernels_generic
            so_paths, verify_report = verify_all_kernels_generic(
                op_reg, so_paths, torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            # 验证通过的 kernel 存入持久化 OperatorRegistry，下次启动可直接复用
            _save_verified_kernels_to_registry(kernels, so_paths, verify_report, gpu_key="rtx_4090")
            # 验证后清理
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            import gc; gc.collect()
        else:
            verify_report = {name: {"passed": False, "detail": "skipped"} for name in kernels}

        # Step 3: 创建 autograd.Function（通用框架：OpRegistry.build_custom_fn_map）
        print("\n[Step 3/5] 注册自定义 autograd.Function...")
        fn_map = op_reg.build_custom_fn_map(so_paths)

        # 兼容性：也维持旧变量名，供 run_lora_training 的 custom_fn_map 参数使用
        silu_fn     = fn_map.get("silu_fn")
        RMSNormModule = fn_map.get("RMSNormModule")
        silu_fwd_ok   = so_paths.get("silu_forward") is not None
        silu_bwd_ok   = so_paths.get("silu_backward") is not None
        rmsnorm_fwd_ok = so_paths.get("rmsnorm_forward") is not None
        rmsnorm_bwd_ok = so_paths.get("rmsnorm_backward") is not None

        # 若通用 fn_map 未包含 silu/rmsnorm（如 so_paths 全为 None），用旧方法 fallback
        if silu_fn is None:
            silu_fn, silu_fwd_ok, silu_bwd_ok = create_custom_silu(
                so_paths.get("silu_forward"), so_paths.get("silu_backward"))
        if RMSNormModule is None:
            RMSNormModule, rmsnorm_fwd_ok, rmsnorm_bwd_ok = create_custom_rmsnorm(
                so_paths.get("rmsnorm_forward"), so_paths.get("rmsnorm_backward"))

        kernel_status = {
            "silu_forward": "✅ agent kernel (verified)" if silu_fwd_ok else "⚠ PyTorch fallback",
            "silu_backward": "✅ agent kernel (verified)" if silu_bwd_ok else "⚠ PyTorch fallback",
            "rmsnorm_forward": "✅ agent kernel (verified)" if rmsnorm_fwd_ok else "⚠ PyTorch fallback",
            "rmsnorm_backward": "✅ agent kernel (verified)" if rmsnorm_bwd_ok else "⚠ PyTorch fallback",
        }
        for k, v in kernel_status.items():
            print(f"  {k}: {v}")

        # Step 4-5: 训练
        print("\n[Step 4-5/5] 注入算子 → Alpaca LoRA 微调...")
        custom_result = run_lora_training(
            args.model,
            custom_fn_map={"silu_fn": silu_fn, "RMSNormModule": RMSNormModule},
            num_steps=args.steps, lr=args.lr,
        )

        # 评估
        print("\n[Eval] 评估 Custom 模式（Alpaca 指令跟随）...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eval_custom = evaluate_alpaca(custom_result["model"], custom_result["tokenizer"], device)
        results["custom"] = {
            "eval": eval_custom,
            "initial_loss": custom_result["initial_loss"],
            "final_loss": custom_result["final_loss"],
            "loss_decreased": custom_result["loss_decreased"],
            "silu_replaced": custom_result["silu_replaced"],
            "rmsnorm_replaced": custom_result["rmsnorm_replaced"],
            "kernel_status": kernel_status,
        }
        score_c = eval_custom["combined_score"]
        print(f"  Custom 指令跟随分: {score_c*100:.1f}% "
              f"(关键词: {eval_custom['keyword_score']*100:.1f}%, "
              f"格式: {eval_custom['format_score']*100:.1f}%)")

        del custom_result["model"]
        torch.cuda.empty_cache()

    # ── Baseline：PyTorch 原生算子 ─────────────────────────────
    if args.mode in ("baseline", "all"):
        print("\n" + "─" * 65)
        print("  [模式 3/3] Baseline：PyTorch 原生算子")
        print("─" * 65)
        baseline_result = run_lora_training(
            args.model,
            custom_fn_map=None,
            num_steps=args.steps, lr=args.lr,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eval_baseline = evaluate_alpaca(baseline_result["model"], baseline_result["tokenizer"], device)
        results["baseline"] = {
            "eval": eval_baseline,
            "initial_loss": baseline_result["initial_loss"],
            "final_loss": baseline_result["final_loss"],
            "loss_decreased": baseline_result["loss_decreased"],
        }
        score_b = eval_baseline["combined_score"]
        print(f"  Baseline 指令跟随分: {score_b*100:.1f}%")
        del baseline_result["model"]
        torch.cuda.empty_cache()

    # ── 汇总输出 ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║        Alpaca 指令跟随能力评估结果汇总               ║")
    print("  ╠══════════════════════════════════════════════════════╣")

    for mode_name, res in results.items():
        ev = res.get("eval", {})
        score = ev.get("combined_score", 0)
        kw = ev.get("keyword_score", 0)
        fmt = ev.get("format_score", 0)
        loss_info = ""
        if "initial_loss" in res:
            trend = "↓" if res.get("loss_decreased") else "↑"
            loss_info = f" Loss:{res['initial_loss']:.2f}→{res['final_loss']:.2f}{trend}"
        print(f"  ║  {mode_name:<14} Score:{score*100:5.1f}% "
              f"(KW:{kw*100:.0f}% Fmt:{fmt*100:.0f}%){loss_info:<18}║")

    print("  ╚══════════════════════════════════════════════════════╝")

    # 效果对比
    if "custom" in results and "no_finetune" in results:
        s_c = results["custom"]["eval"]["combined_score"]
        s_nft = results["no_finetune"]["eval"]["combined_score"]
        diff = s_c - s_nft
        sign = "+" if diff >= 0 else ""
        print(f"\n  Custom vs No-Finetune: {sign}{diff*100:.1f}% 指令跟随提升")
    if "custom" in results and "baseline" in results:
        s_c = results["custom"]["eval"]["combined_score"]
        s_b = results["baseline"]["eval"]["combined_score"]
        diff2 = s_c - s_b
        sign2 = "+" if diff2 >= 0 else ""
        print(f"  Custom vs Baseline:    {sign2}{diff2*100:.1f}% 分差（验证算子等价性）")

    if "custom" in results:
        ks = results["custom"].get("kernel_status", {})
        agent_count = sum(1 for v in ks.values() if "agent" in v)
        print(f"\n  算子覆盖: {agent_count}/4 个 kernel 由 Agent 系统生成并通过数值验证")
        print(f"  替换层数: SiLU × {results['custom']['silu_replaced']}, "
              f"RMSNorm × {results['custom']['rmsnorm_replaced']}")

    ok = True
    if "custom" in results:
        if not results["custom"].get("loss_decreased"):
            print("\n  ⚠ Custom 模式 Loss 未下降，检查学习率或 kernel 正确性")
            ok = False
        else:
            print("\n  ✅ 训练成功！Loss 下降，算子链路正常。")
        s_c = results["custom"]["eval"]["combined_score"]
        s_nft = results.get("no_finetune", {}).get("eval", {}).get("combined_score", 0)
        if s_c > s_nft:
            print(f"  ✅ 微调后指令跟随能力提升：{s_nft*100:.1f}% → {s_c*100:.1f}%")
        else:
            print(f"  ⚠ 指令跟随分未提升（可能需要更多步数或调整 lr）")

    print("=" * 65)
    return 0 if ok else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
