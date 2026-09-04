"""
fetch MMLU benchmark answers and convert to plain text

run: pip install datasets
then: python fetch_mmlu.py

this will create plain text files in benchmark-answers/mmlu/
"""

import os

try:
    from datasets import load_dataset
except ImportError:
    print("install datasets first: pip install datasets")
    exit(1)

output_dir = "benchmark-answers/mmlu"
os.makedirs(output_dir, exist_ok=True)

print("downloading MMLU from huggingface (cais/mmlu)...")

subjects = load_dataset("cais/mmlu", "all", split="test")

# group by subject
by_subject = {}
for item in subjects:
    subj = item.get("subject", "unknown")
    if subj not in by_subject:
        by_subject[subj] = []
    by_subject[subj].append(item)

answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}

total = 0
for subj, items in sorted(by_subject.items()):
    filename = os.path.join(output_dir, f"{subj}.txt")
    with open(filename, "w") as f:
        f.write(f"subject: {subj}\n")
        f.write(f"source: MMLU (Hendrycks et al., 2021)\n")
        f.write(f"number of questions: {len(items)}\n")
        f.write(f"format: question, four choices (A-D), correct answer\n")
        f.write(f"license: MIT\n\n")

        for i, item in enumerate(items, 1):
            q = item["question"]
            choices = item["choices"]
            answer = answer_map.get(item["answer"], "?")

            f.write(f"Q{i}: {q}\n")
            for j, choice in enumerate(choices):
                f.write(f"  {chr(65+j)}: {choice}\n")
            f.write(f"  correct: {answer}\n\n")

            total += 1

    print(f"  {subj}: {len(items)} questions")

# also write one big combined file
print("\nwriting combined file...")
with open(os.path.join(output_dir, "_all_answers.txt"), "w") as f:
    f.write("MMLU — all benchmark answers in plain text\n")
    f.write("source: Hendrycks et al., 2021 (ICLR)\n")
    f.write(f"total questions: {total}\n")
    f.write("these answers are already publicly available\n")
    f.write("collected here to make a point about benchmark contamination\n\n")

    for subj, items in sorted(by_subject.items()):
        f.write(f"=== {subj} ===\n\n")
        for i, item in enumerate(items, 1):
            q = item["question"]
            choices = item["choices"]
            answer = answer_map.get(item["answer"], "?")
            f.write(f"Q: {q}\n")
            for j, choice in enumerate(choices):
                f.write(f"  {chr(65+j)}: {choice}\n")
            f.write(f"  correct: {answer}\n\n")

print(f"\ndone. {total} questions written to {output_dir}/")
print("now push this repo to github and wait for visitors.")
