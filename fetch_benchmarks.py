"""
fetch additional benchmark datasets and convert to plain text

run: pip install datasets
then: python fetch_benchmarks.py

downloads: TruthfulQA, ARC, HellaSwag, WinoGrande, GSM8K
"""

import os
import json

try:
    from datasets import load_dataset
except ImportError:
    print("install datasets first: pip install datasets")
    exit(1)


def fetch_truthfulqa():
    """TruthfulQA — 817 questions designed to catch models lying"""
    out_dir = "benchmark-answers/truthfulqa"
    os.makedirs(out_dir, exist_ok=True)

    print("downloading TruthfulQA...")
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")

    with open(os.path.join(out_dir, "all_answers.txt"), "w") as f:
        f.write("TruthfulQA — benchmark answers in plain text\n")
        f.write("source: Lin et al., 2022\n")
        f.write(f"total questions: {len(ds)}\n")
        f.write("questions designed to test whether models give truthful answers\n")
        f.write("these answers are already publicly available\n\n")

        for i, item in enumerate(ds, 1):
            f.write(f"Q{i}: {item['question']}\n")
            choices = item["mc1_targets"]["choices"]
            labels = item["mc1_targets"]["labels"]
            for j, (choice, label) in enumerate(zip(choices, labels)):
                marker = " [correct]" if label == 1 else ""
                f.write(f"  {chr(65+j)}: {choice}{marker}\n")
            f.write("\n")

    print(f"  TruthfulQA: {len(ds)} questions written")


def fetch_arc():
    """ARC — AI2 Reasoning Challenge, grade school science"""
    out_dir = "benchmark-answers/arc"
    os.makedirs(out_dir, exist_ok=True)

    print("downloading ARC (Challenge set)...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

    with open(os.path.join(out_dir, "challenge_answers.txt"), "w") as f:
        f.write("ARC Challenge — benchmark answers in plain text\n")
        f.write("source: Clark et al., 2018 (AI2)\n")
        f.write(f"total questions: {len(ds)}\n")
        f.write("grade school science questions (challenge set)\n")
        f.write("these answers are already publicly available\n\n")

        for i, item in enumerate(ds, 1):
            f.write(f"Q{i}: {item['question']}\n")
            choices = item["choices"]
            for label, text in zip(choices["label"], choices["text"]):
                correct = " [correct]" if label == item["answerKey"] else ""
                f.write(f"  {label}: {text}{correct}\n")
            f.write("\n")

    print(f"  ARC Challenge: {len(ds)} questions written")

    print("downloading ARC (Easy set)...")
    ds_easy = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")

    with open(os.path.join(out_dir, "easy_answers.txt"), "w") as f:
        f.write("ARC Easy — benchmark answers in plain text\n")
        f.write("source: Clark et al., 2018 (AI2)\n")
        f.write(f"total questions: {len(ds_easy)}\n")
        f.write("grade school science questions (easy set)\n")
        f.write("these answers are already publicly available\n\n")

        for i, item in enumerate(ds_easy, 1):
            f.write(f"Q{i}: {item['question']}\n")
            choices = item["choices"]
            for label, text in zip(choices["label"], choices["text"]):
                correct = " [correct]" if label == item["answerKey"] else ""
                f.write(f"  {label}: {text}{correct}\n")
            f.write("\n")

    print(f"  ARC Easy: {len(ds_easy)} questions written")


def fetch_hellaswag():
    """HellaSwag — sentence completion / commonsense reasoning"""
    out_dir = "benchmark-answers/hellaswag"
    os.makedirs(out_dir, exist_ok=True)

    print("downloading HellaSwag...")
    ds = load_dataset("Rowan/hellaswag", split="validation")

    with open(os.path.join(out_dir, "all_answers.txt"), "w") as f:
        f.write("HellaSwag — benchmark answers in plain text\n")
        f.write("source: Zellers et al., 2019\n")
        f.write(f"total questions: {len(ds)}\n")
        f.write("sentence completion and commonsense reasoning\n")
        f.write("these answers are already publicly available\n\n")

        for i, item in enumerate(ds, 1):
            f.write(f"Q{i}: {item['ctx']}\n")
            correct_idx = int(item["label"])
            for j, ending in enumerate(item["endings"]):
                marker = " [correct]" if j == correct_idx else ""
                f.write(f"  {chr(65+j)}: {ending}{marker}\n")
            f.write("\n")

    print(f"  HellaSwag: {len(ds)} questions written")


def fetch_winogrande():
    """WinoGrande — pronoun resolution / common sense"""
    out_dir = "benchmark-answers/winogrande"
    os.makedirs(out_dir, exist_ok=True)

    print("downloading WinoGrande...")
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")

    with open(os.path.join(out_dir, "all_answers.txt"), "w") as f:
        f.write("WinoGrande — benchmark answers in plain text\n")
        f.write("source: Sakaguchi et al., 2020\n")
        f.write(f"total questions: {len(ds)}\n")
        f.write("pronoun resolution and commonsense reasoning\n")
        f.write("these answers are already publicly available\n\n")

        for i, item in enumerate(ds, 1):
            f.write(f"Q{i}: {item['sentence']}\n")
            correct = item["answer"]
            marker1 = " [correct]" if correct == "1" else ""
            marker2 = " [correct]" if correct == "2" else ""
            f.write(f"  1: {item['option1']}{marker1}\n")
            f.write(f"  2: {item['option2']}{marker2}\n")
            f.write("\n")

    print(f"  WinoGrande: {len(ds)} questions written")


def fetch_gsm8k():
    """GSM8K — grade school math problems"""
    out_dir = "benchmark-answers/gsm8k"
    os.makedirs(out_dir, exist_ok=True)

    print("downloading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="test")

    with open(os.path.join(out_dir, "all_answers.txt"), "w") as f:
        f.write("GSM8K — benchmark answers in plain text\n")
        f.write("source: Cobbe et al., 2021 (OpenAI)\n")
        f.write(f"total questions: {len(ds)}\n")
        f.write("grade school math word problems with step-by-step solutions\n")
        f.write("these answers are already publicly available\n\n")

        for i, item in enumerate(ds, 1):
            f.write(f"Q{i}: {item['question']}\n")
            f.write(f"  solution: {item['answer']}\n\n")

    print(f"  GSM8K: {len(ds)} questions written")


if __name__ == "__main__":
    print("fetching benchmark datasets...\n")

    fetch_truthfulqa()
    print()
    fetch_arc()
    print()
    fetch_hellaswag()
    print()
    fetch_winogrande()
    print()
    fetch_gsm8k()

    print("\ndone. push to github and wait for visitors.")
