"""
Evaluation Harness for superGPT
==================================
Standardized model evaluation across common LLM benchmarks.

Supported benchmarks:
  - MMLU (Massive Multitask Language Understanding): 57 subjects
  - HellaSwag: commonsense reasoning completions
  - ARC (AI2 Reasoning Challenge): science questions
  - TruthfulQA: factual accuracy
  - GSM8K: grade school math
  - HumanEval: code generation

Data sources:
  - Automatically downloads from HuggingFace datasets if available
  - Falls back to built-in sample questions for offline use

Usage:
    # Run all benchmarks
    python evaluate.py --checkpoint checkpoints/best.pt

    # Run specific benchmarks
    python evaluate.py --checkpoint best.pt --benchmarks mmlu hellaswag arc

    # Few-shot evaluation
    python evaluate.py --checkpoint best.pt --n-shot 5

    # Save results to JSON
    python evaluate.py --checkpoint best.pt --output results.json

    # Limit number of tasks per benchmark (for speed)
    python evaluate.py --checkpoint best.pt --max-tasks 100

Reference:
    lm-evaluation-harness, OpenAI Evals
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F

from config import GPTConfig
from model import GPT


# ==============================================================================
#  HuggingFace Dataset Loading
# ==============================================================================

def _try_load_hf_dataset(name, split, **kwargs):
    """Try to load a HuggingFace dataset. Returns None if unavailable."""
    try:
        from datasets import load_dataset
        ds = load_dataset(name, split=split, trust_remote_code=True, **kwargs)
        return ds
    except Exception as e:
        print(f"  [Note] Could not load HF dataset '{name}': {e}")
        print(f"  [Note] Using built-in sample questions instead.")
        return None


# ==============================================================================
#  Base Benchmark Class
# ==============================================================================

class Benchmark:
    """Base class for evaluation benchmarks."""

    name: str = "base"
    description: str = ""

    def __init__(self, n_shot: int = 0, max_tasks: int = 0):
        self.n_shot = n_shot
        self.max_tasks = max_tasks

    def get_tasks(self) -> List[Dict]:
        """Return list of evaluation tasks."""
        raise NotImplementedError

    def format_prompt(self, task: Dict, few_shot_examples: List[Dict] = None) -> str:
        """Format a task into a prompt string."""
        raise NotImplementedError

    def score(self, task: Dict, model_output: str) -> bool:
        """Score whether the model output is correct."""
        raise NotImplementedError


# ==============================================================================
#  MMLU — Massive Multitask Language Understanding
# ==============================================================================

class MMLUBenchmark(Benchmark):
    """MMLU: Multiple choice questions across 57 subjects.

    Data source: HuggingFace cais/mmlu (14K test questions)
    Fallback: built-in sample questions
    """

    name = "mmlu"
    description = "Massive Multitask Language Understanding (57 subjects)"

    SAMPLE_QUESTIONS = [
        {"question": "Which of the following is NOT a component of GDP?",
         "choices": ["Consumption", "Investment", "Government spending", "Population"],
         "answer": "D", "subject": "economics"},
        {"question": "What is the derivative of sin(x)?",
         "choices": ["cos(x)", "-cos(x)", "sin(x)", "-sin(x)"],
         "answer": "A", "subject": "mathematics"},
        {"question": "The process by which plants convert sunlight to energy is called:",
         "choices": ["Respiration", "Photosynthesis", "Fermentation", "Osmosis"],
         "answer": "B", "subject": "biology"},
        {"question": "In Python, what does 'len()' return for a string?",
         "choices": ["Bytes", "Characters", "Words", "Lines"],
         "answer": "B", "subject": "computer_science"},
        {"question": "The Pythagorean theorem states that:",
         "choices": ["a+b=c", "a^2+b^2=c^2", "a*b=c", "a/b=c"],
         "answer": "B", "subject": "mathematics"},
        {"question": "Which planet is known as the Red Planet?",
         "choices": ["Venus", "Mars", "Jupiter", "Saturn"],
         "answer": "B", "subject": "astronomy"},
        {"question": "What is the capital of France?",
         "choices": ["London", "Berlin", "Madrid", "Paris"],
         "answer": "D", "subject": "geography"},
        {"question": "RNA differs from DNA in that RNA contains:",
         "choices": ["Thymine", "Uracil", "Guanine", "Cytosine"],
         "answer": "B", "subject": "biology"},
    ]

    def get_tasks(self):
        # Try loading from HuggingFace
        ds = _try_load_hf_dataset("cais/mmlu", "test", name="all")
        if ds is not None:
            tasks = []
            for item in ds:
                choices = item.get("choices", [])
                answer_idx = item.get("answer", 0)
                if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
                    answer_letter = chr(65 + answer_idx)
                else:
                    answer_letter = str(answer_idx)
                tasks.append({
                    "question": item["question"],
                    "choices": choices,
                    "answer": answer_letter,
                    "subject": item.get("subject", "unknown"),
                })
            if self.max_tasks > 0:
                tasks = tasks[:self.max_tasks]
            return tasks
        return self.SAMPLE_QUESTIONS

    def format_prompt(self, task, few_shot_examples=None):
        prompt = ""
        if few_shot_examples:
            for ex in few_shot_examples[:self.n_shot]:
                prompt += self._format_question(ex) + f"\nAnswer: {ex['answer']}\n\n"
        prompt += self._format_question(task) + "\nAnswer:"
        return prompt

    def _format_question(self, task):
        q = f"Question: {task['question']}\n"
        for i, choice in enumerate(task["choices"]):
            letter = chr(65 + i)
            q += f"  {letter}. {choice}\n"
        return q

    def score(self, task, model_output):
        output = model_output.strip().upper()
        if output and output[0] in "ABCDEFGHIJ":
            return output[0] == task["answer"]
        return False


# ==============================================================================
#  HellaSwag — Commonsense Reasoning
# ==============================================================================

class HellaSwagBenchmark(Benchmark):
    """HellaSwag: Choose the most plausible continuation.

    Data source: HuggingFace Rowan/hellaswag (10K validation)
    Fallback: built-in sample questions
    """

    name = "hellaswag"
    description = "Commonsense reasoning (sentence completion)"

    SAMPLE_TASKS = [
        {"context": "A person is seen sitting on a roof. They start to play a beat on a set of bongos.",
         "choices": ["They continue to play and eventually stop to take a break.",
                     "They fly off the roof into the sky.",
                     "A ball rolls into the scene and hits them.",
                     "An orchestra appears behind them."],
         "answer": 0},
        {"context": "A chef walks into a kitchen. They pick up a knife and a cutting board.",
         "choices": ["They begin to chop vegetables for a salad.",
                     "They start painting the walls with the knife.",
                     "They use the cutting board as a surfboard.",
                     "A horse walks through the kitchen door."],
         "answer": 0},
        {"context": "A student opens a textbook to study for an exam.",
         "choices": ["They use it as a pillow instead.",
                     "They highlight key passages and take notes.",
                     "The book transforms into a bird.",
                     "They eat the textbook."],
         "answer": 1},
    ]

    def get_tasks(self):
        ds = _try_load_hf_dataset("Rowan/hellaswag", "validation")
        if ds is not None:
            tasks = []
            for item in ds:
                endings = item.get("endings", [])
                label = item.get("label", "0")
                tasks.append({
                    "context": item.get("ctx", item.get("activity_label", "")),
                    "choices": endings,
                    "answer": int(label) if str(label).isdigit() else 0,
                })
            if self.max_tasks > 0:
                tasks = tasks[:self.max_tasks]
            return tasks
        return self.SAMPLE_TASKS

    def format_prompt(self, task, few_shot_examples=None):
        prompt = ""
        if few_shot_examples:
            for ex in few_shot_examples[:self.n_shot]:
                prompt += f"Context: {ex['context']}\n"
                for i, c in enumerate(ex['choices']):
                    prompt += f"  {i+1}. {c}\n"
                prompt += f"Most plausible: {ex['answer'] + 1}\n\n"

        prompt += f"Context: {task['context']}\n"
        for i, c in enumerate(task['choices']):
            prompt += f"  {i+1}. {c}\n"
        prompt += "Most plausible:"
        return prompt

    def score(self, task, model_output):
        output = model_output.strip()
        try:
            choice = int(output[0]) - 1 if output else -1
            return choice == task["answer"]
        except (ValueError, IndexError):
            return False


# ==============================================================================
#  ARC — AI2 Reasoning Challenge
# ==============================================================================

class ARCBenchmark(Benchmark):
    """ARC: Science questions from grade school and middle school.

    Data source: HuggingFace allenai/ai2_arc (ARC-Challenge test set)
    Fallback: built-in sample questions
    """

    name = "arc"
    description = "AI2 Reasoning Challenge (science questions)"

    SAMPLE_TASKS = [
        {"question": "Which property of a mineral can be determined just by looking at it?",
         "choices": ["Luster", "Hardness", "Weight", "Streak"], "answer": "A"},
        {"question": "What is the main function of the roots of a plant?",
         "choices": ["Make food", "Absorb water", "Make seeds", "Produce oxygen"], "answer": "B"},
        {"question": "Which of these represents a chemical change?",
         "choices": ["Melting ice", "Burning wood", "Cutting paper", "Dissolving salt"], "answer": "B"},
    ]

    def get_tasks(self):
        ds = _try_load_hf_dataset("allenai/ai2_arc", "test", name="ARC-Challenge")
        if ds is not None:
            tasks = []
            for item in ds:
                choices_data = item.get("choices", {})
                labels = choices_data.get("label", [])
                texts = choices_data.get("text", [])
                answer_key = item.get("answerKey", "A")
                tasks.append({
                    "question": item["question"],
                    "choices": texts,
                    "answer": answer_key,
                })
            if self.max_tasks > 0:
                tasks = tasks[:self.max_tasks]
            return tasks
        return self.SAMPLE_TASKS

    def format_prompt(self, task, few_shot_examples=None):
        prompt = ""
        if few_shot_examples:
            for ex in few_shot_examples[:self.n_shot]:
                prompt += f"Q: {ex['question']}\n"
                for i, c in enumerate(ex['choices']):
                    prompt += f"  {chr(65+i)}. {c}\n"
                prompt += f"A: {ex['answer']}\n\n"

        prompt += f"Q: {task['question']}\n"
        for i, c in enumerate(task['choices']):
            prompt += f"  {chr(65+i)}. {c}\n"
        prompt += "A:"
        return prompt

    def score(self, task, model_output):
        output = model_output.strip().upper()
        if output and output[0] in "ABCDEFGHIJ":
            return output[0] == task["answer"]
        return False


# ==============================================================================
#  GSM8K — Grade School Math
# ==============================================================================

class GSM8KBenchmark(Benchmark):
    """GSM8K: Grade school math word problems.

    Data source: HuggingFace openai/gsm8k (1.3K test)
    Fallback: built-in sample questions
    """

    name = "gsm8k"
    description = "Grade School Math (word problems)"

    SAMPLE_TASKS = [
        {"question": "Janet has 5 apples. She buys 3 more. How many apples does she have?",
         "answer": "8"},
        {"question": "A store has 20 shirts. If 7 are sold, how many remain?",
         "answer": "13"},
        {"question": "Tom runs 3 miles every day for 5 days. How many miles total?",
         "answer": "15"},
    ]

    def get_tasks(self):
        ds = _try_load_hf_dataset("openai/gsm8k", "test", name="main")
        if ds is not None:
            tasks = []
            for item in ds:
                # GSM8K answers are in format "... #### <number>"
                answer_text = item.get("answer", "")
                if "####" in answer_text:
                    final_answer = answer_text.split("####")[-1].strip()
                    final_answer = final_answer.replace(",", "")
                else:
                    final_answer = answer_text.strip()
                tasks.append({
                    "question": item["question"],
                    "answer": final_answer,
                })
            if self.max_tasks > 0:
                tasks = tasks[:self.max_tasks]
            return tasks
        return self.SAMPLE_TASKS

    def format_prompt(self, task, few_shot_examples=None):
        prompt = ""
        if few_shot_examples:
            for ex in few_shot_examples[:self.n_shot]:
                prompt += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"

        prompt += f"Q: {task['question']}\nA:"
        return prompt

    def score(self, task, model_output):
        output = model_output.strip()
        # Extract the last number from the output
        numbers = []
        current = ""
        for c in output:
            if c.isdigit() or c == '.':
                current += c
            elif current:
                numbers.append(current)
                current = ""
        if current:
            numbers.append(current)

        if numbers:
            try:
                predicted = str(int(float(numbers[-1])))
                expected = str(int(float(task["answer"])))
                return predicted == expected
            except ValueError:
                pass
        return False


# ==============================================================================
#  TruthfulQA
# ==============================================================================

class TruthfulQABenchmark(Benchmark):
    """TruthfulQA: Tests model's tendency to reproduce common misconceptions.

    Data source: HuggingFace truthfulqa/truthful_qa (generation split)
    Fallback: built-in sample questions
    """

    name = "truthfulqa"
    description = "Factual accuracy (common misconceptions)"

    SAMPLE_TASKS = [
        {"question": "What happens if you crack your knuckles?",
         "good_answers": ["nothing harmful", "it makes a sound", "no long-term damage"],
         "bad_answers": ["arthritis", "joint damage"]},
        {"question": "Do humans only use 10% of their brains?",
         "good_answers": ["no", "false", "we use all of our brain"],
         "bad_answers": ["yes", "true", "we only use 10%"]},
    ]

    def get_tasks(self):
        ds = _try_load_hf_dataset("truthfulqa/truthful_qa", "validation",
                                  name="generation")
        if ds is not None:
            tasks = []
            for item in ds:
                correct = item.get("correct_answers", [])
                incorrect = item.get("incorrect_answers", [])
                tasks.append({
                    "question": item["question"],
                    "good_answers": correct if correct else ["yes"],
                    "bad_answers": incorrect if incorrect else ["no"],
                })
            if self.max_tasks > 0:
                tasks = tasks[:self.max_tasks]
            return tasks
        return self.SAMPLE_TASKS

    def format_prompt(self, task, few_shot_examples=None):
        return f"Q: {task['question']}\nA:"

    def score(self, task, model_output):
        output = model_output.lower().strip()
        for good in task["good_answers"]:
            if good.lower() in output:
                return True
        return False


# ==============================================================================
#  HumanEval — Code Generation
# ==============================================================================

class HumanEvalBenchmark(Benchmark):
    """HumanEval: Python code generation problems.

    Data source: HuggingFace openai/openai_humaneval
    Fallback: built-in sample problems
    """

    name = "humaneval"
    description = "Code generation (Python functions)"

    SAMPLE_TASKS = [
        {"prompt": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n",
         "test": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n",
         "canonical": "    return a + b"},
        {"prompt": "def is_even(n):\n    \"\"\"Check if a number is even.\"\"\"\n",
         "test": "assert is_even(4) == True\nassert is_even(3) == False\n",
         "canonical": "    return n % 2 == 0"},
        {"prompt": "def factorial(n):\n    \"\"\"Compute n factorial.\"\"\"\n",
         "test": "assert factorial(5) == 120\nassert factorial(0) == 1\n",
         "canonical": "    return 1 if n <= 1 else n * factorial(n - 1)"},
    ]

    def get_tasks(self):
        ds = _try_load_hf_dataset("openai/openai_humaneval", "test")
        if ds is not None:
            tasks = []
            for item in ds:
                tasks.append({
                    "prompt": item.get("prompt", ""),
                    "test": item.get("test", ""),
                    "canonical": item.get("canonical_solution", ""),
                    "entry_point": item.get("entry_point", ""),
                })
            if self.max_tasks > 0:
                tasks = tasks[:self.max_tasks]
            return tasks
        return self.SAMPLE_TASKS

    def format_prompt(self, task, few_shot_examples=None):
        return task["prompt"]

    def score(self, task, model_output):
        """Try to execute the generated code and run tests."""
        code = task["prompt"] + model_output.split("\n\n")[0] + "\n"
        test_code = code + task["test"]
        try:
            exec(test_code, {})
            return True
        except Exception:
            return False


# ==============================================================================
#  Benchmark Registry
# ==============================================================================

BENCHMARKS = {
    "mmlu": MMLUBenchmark,
    "hellaswag": HellaSwagBenchmark,
    "arc": ARCBenchmark,
    "gsm8k": GSM8KBenchmark,
    "truthfulqa": TruthfulQABenchmark,
    "humaneval": HumanEvalBenchmark,
}


# ==============================================================================
#  Model Evaluation Engine
# ==============================================================================

@torch.no_grad()
def generate_text(model, prompt_tokens, max_new=64, temperature=0.1,
                  device="cpu"):
    """Generate text from a model given prompt tokens."""
    input_ids = torch.tensor(prompt_tokens, dtype=torch.long,
                              device=device).unsqueeze(0)

    for _ in range(max_new):
        logits, _ = model(input_ids[:, -model.config.block_size:])
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    new_tokens = input_ids[0, len(prompt_tokens):].tolist()
    return new_tokens


def tokenize_simple(text, vocab_size=256):
    """Simple character-level tokenization."""
    return [min(ord(c), vocab_size - 1) for c in text]


def detokenize_simple(tokens):
    """Simple character-level detokenization."""
    return "".join(chr(t) if t < 128 else "?" for t in tokens)


def evaluate_benchmark(model, benchmark, device="cpu", max_gen=64,
                       temperature=0.1, verbose=True):
    """Evaluate a model on a single benchmark.

    Returns:
        results: dict with accuracy and per-task results
    """
    model.eval()
    tasks = benchmark.get_tasks()

    correct = 0
    total = 0
    per_task = []

    for i, task in enumerate(tasks):
        prompt = benchmark.format_prompt(task)
        prompt_tokens = tokenize_simple(prompt)

        # Generate
        output_tokens = generate_text(
            model, prompt_tokens, max_new=max_gen,
            temperature=temperature, device=device,
        )
        output_text = detokenize_simple(output_tokens)

        # Score
        is_correct = benchmark.score(task, output_text)
        correct += int(is_correct)
        total += 1

        per_task.append({
            "task_idx": i,
            "correct": is_correct,
            "output": output_text[:200],
        })

        if verbose:
            status = "PASS" if is_correct else "FAIL"
            print(f"  [{status}] Task {i+1}/{len(tasks)}: {output_text[:50]}...")

    accuracy = correct / max(total, 1)
    return {
        "benchmark": benchmark.name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_task": per_task,
    }


def evaluate_model(args):
    """Run full evaluation suite on a model."""
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else \
                 "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    # Load model
    print(f"Loading model: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["model_config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.1f}M params on {device}")

    # Select benchmarks
    if args.benchmarks:
        benchmark_names = args.benchmarks
    else:
        benchmark_names = list(BENCHMARKS.keys())

    # Run evaluation
    all_results = {}
    data_source = "HuggingFace" if not args.offline else "built-in samples"
    print(f"\n{'='*60}")
    print(f"  superGPT Evaluation")
    print(f"{'='*60}")
    print(f"  Model:      {n_params/1e6:.1f}M params")
    print(f"  Benchmarks: {', '.join(benchmark_names)}")
    print(f"  N-shot:     {args.n_shot}")
    print(f"  Data:       {data_source}")
    print(f"{'='*60}\n")

    t0 = time.time()
    for name in benchmark_names:
        if name not in BENCHMARKS:
            print(f"  Warning: Unknown benchmark '{name}', skipping")
            continue

        benchmark = BENCHMARKS[name](n_shot=args.n_shot,
                                      max_tasks=args.max_tasks)
        print(f"\n--- {benchmark.name}: {benchmark.description} ---")

        results = evaluate_benchmark(
            model, benchmark, device=device,
            max_gen=args.max_gen, temperature=args.temperature,
            verbose=args.verbose,
        )
        all_results[name] = results

        print(f"  Accuracy: {results['accuracy']:.1%} "
              f"({results['correct']}/{results['total']})")

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*60}")
    print(f"  Results Summary")
    print(f"{'='*60}")
    for name, results in all_results.items():
        print(f"  {name:12s}: {results['accuracy']:6.1%} "
              f"({results['correct']}/{results['total']})")

    avg_acc = sum(r['accuracy'] for r in all_results.values()) / max(len(all_results), 1)
    print(f"  {'Average':12s}: {avg_acc:6.1%}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*60}")

    # Save results
    if args.output:
        output = {
            "model": args.checkpoint,
            "n_params": n_params,
            "n_shot": args.n_shot,
            "results": {name: {
                "accuracy": r["accuracy"],
                "correct": r["correct"],
                "total": r["total"],
            } for name, r in all_results.items()},
            "average_accuracy": avg_acc,
            "elapsed_seconds": elapsed,
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return all_results


# ==============================================================================
#  CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation harness for superGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available benchmarks: {', '.join(BENCHMARKS.keys())}

Examples:
  python evaluate.py --checkpoint best.pt
  python evaluate.py --checkpoint best.pt --benchmarks mmlu gsm8k
  python evaluate.py --checkpoint best.pt --n-shot 5 --output results.json
  python evaluate.py --checkpoint best.pt --max-tasks 100  # limit for speed
        """,
    )

    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Model checkpoint path")
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        help=f"Benchmarks to run (default: all). "
                             f"Options: {', '.join(BENCHMARKS.keys())}")
    parser.add_argument("--n-shot", type=int, default=0,
                        help="Number of few-shot examples (default: 0)")
    parser.add_argument("--max-gen", type=int, default=64,
                        help="Max tokens to generate per task (default: 64)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature (default: 0.1)")
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Max tasks per benchmark, 0=all (default: 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-task results")
    parser.add_argument("--offline", action="store_true",
                        help="Use built-in samples only (no HF download)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, mps, cpu")

    args = parser.parse_args()
    evaluate_model(args)
