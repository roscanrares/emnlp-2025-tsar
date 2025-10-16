import os
import json
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from datetime import datetime
import gc
import numpy as np
from sklearn.metrics import f1_score, root_mean_squared_error

# CEFR level configuration
CEFR_LABELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
LABEL2IDX = {label: idx for idx, label in enumerate(CEFR_LABELS)}

# Initialize CEFR evaluation models for ensemble prediction
cefr_labeler1 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-doc_en-Cefr")
cefr_labeler2 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr")
cefr_labeler3 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-reference_AllLang2-Cefr2")


def get_cefr_labels(simplifications: list, models=[cefr_labeler1, cefr_labeler2, cefr_labeler3]):
    """
    Predict CEFR labels using ensemble of three models.
    Returns the prediction with highest confidence score across all models.
    """
    cefr_labels = []
    for simplification in simplifications:
        top_preds = (model(simplification)[0] for model in models)
        best = max(top_preds, key=lambda d: d["score"])
        cefr_labels.append(best["label"])
    return cefr_labels


def get_cefr_compliance_score(simplifications: list, reference_levels: list,
                              models=[cefr_labeler1, cefr_labeler2, cefr_labeler3]):
    """
    Calculate CEFR compliance metrics between predicted and reference levels.
    Returns weighted F1, adjacent accuracy, and RMSE scores.
    """
    assert len(simplifications) == len(reference_levels), \
        "The number of simplifications must match the number of reference levels."

    predicted_labels = get_cefr_labels(simplifications=simplifications, models=models)
    f1 = f1_score(reference_levels, predicted_labels, average='weighted')

    true_idx = np.array([LABEL2IDX[l] for l in reference_levels])
    pred_idx = np.array([LABEL2IDX[l] for l in predicted_labels])

    adj_acc = (np.abs(true_idx - pred_idx) <= 1).mean()
    rmse = root_mean_squared_error(true_idx, pred_idx)

    return {
        'weighted_f1': round(f1, 4),
        'adj_accuracy': round(adj_acc, 4),
        'rmse': round(rmse, 4)
    }


def evaluate_single_text_cefr(text, target_level, models=[cefr_labeler1, cefr_labeler2, cefr_labeler3]):
    """
    Evaluate a single text and return predicted CEFR level with compliance metrics.
    """
    try:
        simplifications = [text]
        reference_levels = [target_level.upper()]

        predicted_labels = get_cefr_labels(simplifications=simplifications, models=models)

        if predicted_labels and len(predicted_labels) > 0 and predicted_labels[0] is not None:
            predicted_level = predicted_labels[0]

            f1 = f1_score(reference_levels, predicted_labels, average='weighted')

            true_idx = np.array([LABEL2IDX[l] for l in reference_levels])
            pred_idx = np.array([LABEL2IDX[l] for l in predicted_labels])

            adj_acc = (np.abs(true_idx - pred_idx) <= 1).mean()
            rmse = root_mean_squared_error(true_idx, pred_idx)

            metrics = {
                'weighted_f1': round(f1, 4),
                'adj_accuracy': round(adj_acc, 4),
                'rmse': round(rmse, 4)
            }
            return predicted_level, metrics
        else:
            return None, {'weighted_f1': 0.0, 'adj_accuracy': 0.0, 'rmse': 0.0}

    except Exception as e:
        print(f"Evaluation error: {e}")
        return None, {'weighted_f1': 0.0, 'adj_accuracy': 0.0, 'rmse': 0.0}


class CEFRIterativeSimplificationTester:
    """
    Handles iterative text simplification for CEFR levels using fine-tuned models.
    Supports multiple model types and automatic evaluation.
    """

    def __init__(self, models_dir: str = "models/", device: str = "auto", max_iterations: int = 5):
        """
        Initialize tester with model directory, device configuration, and iteration limit.
        """
        self.models_dir = Path(models_dir)
        self.device = device
        self.max_iterations = max_iterations
        self.output_dir = Path("simplifications/")
        self.output_dir.mkdir(exist_ok=True)

    def get_models_for_level(self, target_level: str) -> List[str]:
        """
        Return list of expected model names for a given CEFR target level.
        """
        if target_level == 'A2':
            return ["llama_a2"]
        elif target_level == 'B1':
            return ["llama_b1"]
        else:
            return []

    def find_models_for_level(self, target_level: str) -> List[Path]:
        """
        Locate and validate models for the specified CEFR level.
        """
        expected_models = self.get_models_for_level(target_level)

        models = []
        for model_name in expected_models:
            model_path = self.models_dir / model_name
            if model_path.is_dir() and (model_path / "config.json").exists():
                models.append(model_path)
            else:
                print(f"Model not found or incomplete: {model_name}")

        print(f"Found {len(models)} models for level {target_level}")
        for model in models:
            print(f"  - {model.name}")
        return models

    def load_model_and_tokenizer(self, model_path: Path):
        """
        Load model and tokenizer from specified path with memory optimization.
        Uses float16 precision and automatic device mapping.
        """
        try:
            print(f"\nLoading model: {model_path.name}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            print(f"Model loaded successfully")
            return model, tokenizer

        except Exception as e:
            print(f"Error loading model {model_path.name}: {e}")
            return None, None

    def generate_text(self, model, tokenizer, prompt: str, max_length: int = 1024):
        """
        Generate text using the loaded model with specified parameters.
        Supports optional Qwen chat template formatting (currently commented).
        """
        try:
            # Qwen-specific chat template formatting (uncomment for Qwen models)
            # messages = [{"role": "user", "content": prompt}]
            # prompt = tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=True,
            #     enable_thinking=False
            # )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    attention_mask=inputs.attention_mask
                )

            generated_text = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

            return generated_text

        except Exception as e:
            print(f"    Generation error: {e}")
            return f"ERROR: {str(e)}"

    def get_simplification_prompt(self, original_text: str, target_level: str,
                                  current_level: str = None, iteration: int = 0,
                                  model_name: str = ""):
        """
        Generate appropriate simplification prompt based on model type and iteration feedback.
        Supports standard Llama, Sergiu-formatted, and Qwen models.
        """
        base_feedback = ""
        if current_level and iteration > 0:
            if current_level != target_level.upper():
                base_feedback = f"\nCURRENT ISSUE: The previous version was classified as {current_level}, but we need {target_level.upper()} level. "

                if current_level > target_level.upper():
                    base_feedback += "The text is TOO COMPLEX. Simplify more aggressively."
                else:
                    base_feedback += "The text is TOO SIMPLE. Add more complexity while staying at target level."

        if 'sergiu' in model_name:
            return self.create_prompt(original_text, level=target_level, feedback=base_feedback)
        elif 'qwen' in model_name:
            return self.create_prompt_qwen(original_text, level=target_level, feedback=base_feedback)
        else:
            if target_level.lower() == 'a2':
                return self.create_a2_prompt(original_text, feedback=base_feedback)
            elif target_level.lower() == 'b1':
                return self.create_b1_prompt(original_text, feedback=base_feedback)

    def create_a2_prompt(self, original_text: str, feedback: str = "") -> str:
        """
        Create A2-level simplification prompt with detailed CEFR requirements.
        """
        return (
            f"You are a language teacher simplifying texts to A2 CEFR level.\n\n"
            f"OBJECTIVE: Transform this text to A2 level while preserving all original meaning and information.\n"
            f"A2 LANGUAGE REQUIREMENTS:\n"
            f"- Vocabulary: Most common 1500 English words only\n"
            f"- Sentences: 8-12 words, one clear idea per sentence\n"
            f"- Grammar: Simple present/past, basic future (will), basic modals (can/must/should)\n"
            f"- Connectors: and, but, because, so, when, if, then\n"
            f"- Style: Personal, concrete, everyday language\n\n"
            f"STRICT LEVEL CONTROL:\n"
            f"- Above A1: Include personal experiences, feelings, plans, time references\n"
            f"- Below B1: No present perfect, passive voice, or complex connectors (however, although, despite)\n"
            f"- Below B1: No abstract concepts without concrete explanation\n\n"
            f"TRANSFORMATION PROCESS:\n"
            f"1. Identify all key information and meaning\n"
            f"2. Break complex sentences into simple A2 structures\n"
            f"3. Replace advanced vocabulary with A2 equivalents\n"
            f"4. Convert complex grammar to simple A2 patterns\n"
            f"5. Verify all original meaning is preserved\n\n"
            f"CRITICAL: Do not omit, summarize, or change any information. Only change HOW it's expressed.\n\n"
            f"Return only the simplified text, with no explanations.\n\n"
            f"Text to simplify:\n"
            f'"""\n{original_text}\n"""\n'
        )

    def create_b1_prompt(self, original_text: str, feedback: str = "") -> str:
        """
        Create B1-level simplification prompt with detailed CEFR requirements.
        """
        return (
            f"You are an expert CEFR B1 text simplification specialist with deep understanding of automatic language assessment systems.\n\n"
            f"OBJECTIVE: Transform this text to precise B1 level while preserving all original meaning and information.\n"
            f"B1 LANGUAGE REQUIREMENTS:\n"
            f"- Vocabulary: 2000-3000 most common English words, avoid academic/formal terms\n"
            f"- Sentences: 15-22 words, can connect 2 related ideas with clear logic\n"
            f"- Grammar: Present perfect (have/has done), simple passive (is/was done), basic conditionals (if...will/would), modals (should, might, could, would)\n"
            f"- Connectors: however, although, while, since, unless, because, so that, even though\n"
            f"- Style: Clear intermediate language that shows reasoning and personal opinions\n\n"
            f"STRICT LEVEL CONTROL:\n"
            f"- Above A2: Include abstract concepts with simple explanation, cause-effect relationships, personal opinions with basic justification, intermediate grammar patterns\n"
            f"- Below B2: No academic/formal vocabulary (facilitate→help, demonstrate→show, utilize→use), no complex conditional structures, no sophisticated argumentation, no specialized terminology without explanation\n"
            f"- PRECISE B1 TARGET: Intermediate complexity using everyday vocabulary - never oversimplify to A2, never undersimplify leaving B2+ elements\n\n"
            f"CRITICAL B1 DIFFERENTIATORS:\n"
            f"- From A2: Can handle abstract ideas but explains them simply using common words\n"
            f"- From B2: Uses everyday vocabulary even for complex concepts, avoids formal/academic tone\n"
            f"- B1 signature: Connects ideas logically but with simple language patterns\n\n"
            f"TRANSFORMATION PROCESS:\n"
            f"1. Identify all key information and meaning\n"
            f"2. Scan for B2+ vocabulary and replace with B1 common equivalents\n"
            f"3. Convert complex sentences to B1 structures (maximum 2 clauses per sentence)\n"
            f"4. Add simple explanations for any remaining complex concepts\n"
            f"5. Include 2-3 B1 grammar markers per paragraph naturally\n"
            f"6. Verify consistent B1 complexity throughout - no A2 oversimplification, no B2+ elements remaining\n\n"
            f"CRITICAL: Do not omit, summarize, or change any information. Only change HOW it's expressed to match B1 patterns that automatic CEFR classifiers consistently recognize as B1 level.\n\n"
            f"Return only the simplified text, with no explanations.\n\n"
            f"Text to simplify:\n"
            f'"""\n{original_text}\n"""\n'
        )

    # PROMPT TEMPLATES CONFIGURATION
    # These methods define prompt templates based on the model's fine-tuning format.
    # If your model was fine-tuned with different templates or chat formats,
    # modify these methods accordingly to match your fine-tuning configuration.
    # Each model architecture (Llama, Qwen, etc.) may require specific formatting.

    def create_prompt(self, original_text: str, level='A2', feedback: str = "") -> str:
        """
        Create prompt format (Llama chat template) for CEFR simplification.
        """
        return f"<|start_header_id|>system<|end_header_id|>\nYou are an expert on CEFR with deep understanding of automatic language assessment systems.\n\nOBJECTIVE: Analyze the following text and determine its CEFR level. Return only the CEFR label (e.g., A1, A2, B1, B2, C1, C2).\n\nB1 LANGUAGE REQUIREMENTS:\n- Vocabulary: 2000-3000 most common English words, avoid academic/formal terms\n- Sentences: 15-22 words, can connect 2 related ideas with clear logic\n- Grammar: Present perfect (have/has done), simple passive (is/was done), basic conditionals (if...will/would), modals (should, might, could, would)\n- Connectors: however, although, while, since, unless, because, so that, even though\n- Style: Clear intermediate language that shows reasoning and personal opinions\n\nSTRICT LEVEL CONTROL:\n- Above A2: Include abstract concepts with simple explanation, cause-effect relationships, personal opinions with basic justification, intermediate grammar patterns\n- Below B2: No academic/formal vocabulary (facilitate→help, demonstrate→show, utilize→use), no complex conditional structures, no sophisticated argumentation, no specialized terminology without explanation\n- PRECISE B1 TARGET: Intermediate complexity using everyday vocabulary - never oversimplify to A2, never undersimplify leaving B2+ elements\n\nCRITICAL B1 DIFFERENTIATORS:\n- From A2: Can handle abstract ideas but explains them simply using common words\n- From B2: Uses everyday vocabulary even for complex concepts, avoids formal/academic tone\n- B1 signature: Connects ideas logically but with simple language patterns\n\n\nA2 LANGUAGE REQUIREMENTS:\n- Vocabulary: Most common 1500 English words only\n- Sentences: 8-12 words, one clear idea per sentence\n- Grammar: Simple present/past, basic future (will), basic modals (can/must/should)\n- Connectors: and, but, because, so, when, if, then\n- Style: Personal, concrete, everyday language\n\nSTRICT LEVEL CONTROL:\n- Above A1: Include personal experiences, feelings, plans, time references\n- Below B1: No present perfect, passive voice, or complex connectors (however, although, despite)\n- Below B1: No abstract concepts without concrete explanation\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nOBJECTIVE: Transform this text to precise {level} level while preserving all original meaning and information.\n{original_text}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    def create_prompt_qwen(self, original_text: str, level='A2', feedback: str = "") -> str:
        """
        Create prompt for Qwen models with CEFR simplification requirements.
        """
        return f"You are an expert on CEFR with deep understanding of automatic language assessment systems.\n\nB1 LANGUAGE REQUIREMENTS:\n- Vocabulary: 2000-3000 most common English words, avoid academic/formal terms\n- Sentences: 15-22 words, can connect 2 related ideas with clear logic\n- Grammar: Present perfect (have/has done), simple passive (is/was done), basic conditionals (if...will/would), modals (should, might, could, would)\n- Connectors: however, although, while, since, unless, because, so that, even though\n- Style: Clear intermediate language that shows reasoning and personal opinions\n\nSTRICT LEVEL CONTROL:\n- Above A2: Include abstract concepts with simple explanation, cause-effect relationships, personal opinions with basic justification, intermediate grammar patterns\n- Below B2: No academic/formal vocabulary (facilitate→help, demonstrate→show, utilize→use), no complex conditional structures, no sophisticated argumentation, no specialized terminology without explanation\n- PRECISE B1 TARGET: Intermediate complexity using everyday vocabulary - never oversimplify to A2, never undersimplify leaving B2+ elements\n\nCRITICAL B1 DIFFERENTIATORS:\n- From A2: Can handle abstract ideas but explains them simply using common words\n- From B2: Uses everyday vocabulary even for complex concepts, avoids formal/academic tone\n- B1 signature: Connects ideas logically but with simple language patterns\n\n\nA2 LANGUAGE REQUIREMENTS:\n- Vocabulary: Most common 1500 English words only\n- Sentences: 8-12 words, one clear idea per sentence\n- Grammar: Simple present/past, basic future (will), basic modals (can/must/should)\n- Connectors: and, but, because, so, when, if, then\n- Style: Personal, concrete, everyday language\n\nSTRICT LEVEL CONTROL:\n- Above A1: Include personal experiences, feelings, plans, time references\n- Below B1: No present perfect, passive voice, or complex connectors (however, although, despite)\n- Below B1: No abstract concepts without concrete explanation\n\nOBJECTIVE: Transform this text to precise {level} level while preserving all original meaning and information:\n\n{original_text}"

    def process_text_iteratively(self, model, tokenizer, model_name: str,
                                 original_text: str, target_level: str, text_id: str):
        """
        Process a single text through iterative refinement until target CEFR level is achieved.
        Tracks best iteration based on F1 score and stops early if target is reached.
        """
        current_text = original_text
        iteration_history = []
        best_iteration = None

        print(f"    Processing {text_id} (Target: {target_level.upper()}) with {model_name}")

        for iteration in range(self.max_iterations):
            print(f"      Iteration {iteration + 1}/{self.max_iterations}")

            if iteration == 0:
                prompt = self.get_simplification_prompt(current_text, target_level, model_name=model_name)
            else:
                last_level = iteration_history[-1]['predicted_level']
                prompt = self.get_simplification_prompt(
                    current_text, target_level, last_level, iteration, model_name
                )

            start_time = time.time()
            simplified_text = self.generate_text(model, tokenizer, prompt)
            generation_time = time.time() - start_time

            if simplified_text.startswith("ERROR:"):
                print(f"        Generation error at iteration {iteration + 1}")
                break

            predicted_level, metrics = evaluate_single_text_cefr(simplified_text, target_level)

            iteration_info = {
                'iteration': iteration + 1,
                'text': simplified_text,
                'predicted_level': predicted_level,
                'weighted_f1': metrics['weighted_f1'],
                'adj_accuracy': metrics['adj_accuracy'],
                'rmse': metrics['rmse'],
                'target_achieved': predicted_level == target_level.upper() if predicted_level else False,
                'generation_time': generation_time
            }

            iteration_history.append(iteration_info)

            if best_iteration is None or metrics['weighted_f1'] > best_iteration['weighted_f1']:
                best_iteration = iteration_info.copy()

            print(f"        Prediction: {predicted_level}")
            print(f"        F1: {metrics['weighted_f1']}, Adj Acc: {metrics['adj_accuracy']}, RMSE: {metrics['rmse']}")
            print(f"        Target achieved: {'Yes' if iteration_info['target_achieved'] else 'No'}")

            if iteration_info['target_achieved']:
                print(f"        Target {target_level.upper()} achieved at iteration {iteration + 1}!")
                break

            current_text = simplified_text

        return {
            'text_id': text_id,
            'original_text': original_text,
            'final_simplified': simplified_text,
            'iterations_used': len(iteration_history),
            'final_predicted_level': iteration_history[-1]['predicted_level'] if iteration_history else None,
            'final_weighted_f1': iteration_history[-1]['weighted_f1'] if iteration_history else 0.0,
            'final_adj_accuracy': iteration_history[-1]['adj_accuracy'] if iteration_history else 0.0,
            'final_rmse': iteration_history[-1]['rmse'] if iteration_history else 0.0,
            'target_achieved': iteration_history[-1]['target_achieved'] if iteration_history else False,
            'iteration_history': iteration_history,
            'best_iteration': best_iteration,
            'total_generation_time': sum(it['generation_time'] for it in iteration_history)
        }

    def process_model_for_level(self, model, tokenizer, model_name: str,
                                df: pd.DataFrame, target_level: str):
        """
        Process all texts for a specific CEFR level using the provided model.
        Returns comprehensive results with statistics.
        """
        results = {
            "model_name": model_name,
            "target_level": target_level,
            "timestamp": datetime.now().isoformat(),
            "iterative_results": [],
            "stats": {
                "total_processed": 0,
                "successful": 0,
                "errors": 0,
                "avg_iterations": 0,
                "total_generation_time": 0,
                "texts_by_iterations": {}
            }
        }

        target_texts = df[df['target_cefr'] == target_level]

        if len(target_texts) == 0:
            print(f"    No texts found for level {target_level}")
            return results

        print(f"  Processing {len(target_texts)} texts for level {target_level} iteratively")

        for idx, row in target_texts.iterrows():
            text_id = row['text_id']
            original_text = row['original']

            print(f"    {len(results['iterative_results']) + 1}/{len(target_texts)} - Text ID: {text_id}")

            iterative_result = self.process_text_iteratively(
                model, tokenizer, model_name, original_text, target_level, text_id
            )

            results["iterative_results"].append(iterative_result)

            if iterative_result['target_achieved']:
                results["stats"]["successful"] += 1
            if iterative_result['final_simplified'].startswith("ERROR:"):
                results["stats"]["errors"] += 1

            iterations_used = iterative_result['iterations_used']
            if iterations_used not in results["stats"]["texts_by_iterations"]:
                results["stats"]["texts_by_iterations"][iterations_used] = 0
            results["stats"]["texts_by_iterations"][iterations_used] += 1

            results["stats"]["total_generation_time"] += iterative_result['total_generation_time']

        results["stats"]["total_processed"] = len(results["iterative_results"])
        if results["iterative_results"]:
            total_iterations = sum(r['iterations_used'] for r in results["iterative_results"])
            results["stats"]["avg_iterations"] = total_iterations / len(results["iterative_results"])

        return results

    def save_results(self, results: Dict, model_name: str, level: str):
        """
        Save results in three formats: JSON, CSV summary, and best versions JSONL.
        """
        filename = f"{model_name}_{level}_iterative_simplifications.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"    Results saved: {filepath}")

        self.create_csv_summary(results, model_name, level)
        self.create_best_versions(results, model_name, level)

    def create_csv_summary(self, results: Dict, model_name: str, level: str):
        """
        Create CSV summary with key metrics for each processed text.
        """
        csv_data = []

        for item in results["iterative_results"]:
            csv_data.append({
                "model_name": model_name,
                "target_level": level,
                "text_id": item["text_id"],
                "original_length": len(item["original_text"]),
                "final_simplified_length": len(item["final_simplified"]),
                "iterations_used": item["iterations_used"],
                "target_achieved": item["target_achieved"],
                "final_predicted_level": item["final_predicted_level"],
                "final_weighted_f1": item["final_weighted_f1"],
                "final_adj_accuracy": item["final_adj_accuracy"],
                "final_rmse": item["final_rmse"],
                "total_generation_time": item["total_generation_time"],
                "best_iteration_f1": item["best_iteration"]["weighted_f1"] if item["best_iteration"] else 0.0
            })

        df_summary = pd.DataFrame(csv_data)
        filename = f"{model_name}_{level}_iterative_summary.csv"
        filepath = self.output_dir / filename
        df_summary.to_csv(filepath, index=False)

        print(f"    CSV summary saved: {filepath}")

    def create_best_versions(self, results: Dict, model_name: str, level: str):
        """
        Save best iteration for each text based on F1 score to JSONL file.
        """
        best_versions = []

        for item in results["iterative_results"]:
            if item["best_iteration"]:
                best_versions.append({
                    "text_id": item["text_id"],
                    "simplified": item["best_iteration"]["text"],
                    "predicted_level": item["best_iteration"]["predicted_level"],
                    "weighted_f1": item["best_iteration"]["weighted_f1"],
                    "iteration": item["best_iteration"]["iteration"]
                })

        filename = f"{model_name}_{level}_best_versions.jsonl"
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            for item in best_versions:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')

        print(f"    Best versions saved: {filepath}")

    def run_simplification_for_level(self, df: pd.DataFrame, target_level: str):
        """
        Execute simplification pipeline for all models associated with a CEFR level.
        Handles model loading, processing, result saving, and cleanup.
        """
        models = self.find_models_for_level(target_level)

        if not models:
            print(f"No models found for level {target_level}!")
            return

        print(f"\nStarting iterative {target_level} simplification with {len(models)} models...")
        print(f"Max iterations per text: {self.max_iterations}")

        total_stats = {
            "models_processed": 0,
            "models_failed": 0,
            "total_texts": 0,
            "total_successful": 0,
            "total_iterations": 0
        }

        for i, model_path in enumerate(models):
            print(f"\nModel {i + 1}/{len(models)} for {target_level}: {model_path.name}")

            model, tokenizer = self.load_model_and_tokenizer(model_path)

            if model is None:
                total_stats["models_failed"] += 1
                continue

            try:
                results = self.process_model_for_level(
                    model, tokenizer, model_path.name, df, target_level
                )

                self.save_results(results, model_path.name, target_level)

                total_stats["total_texts"] += results["stats"]["total_processed"]
                total_stats["total_successful"] += results["stats"]["successful"]
                total_stats["total_iterations"] += sum(r['iterations_used'] for r in results["iterative_results"])
                total_stats["models_processed"] += 1

                print(f"Model {model_path.name} processed successfully for {target_level}!")
                print(f"  Success rate: {results['stats']['successful']}/{results['stats']['total_processed']} "
                      f"({results['stats']['successful'] / results['stats']['total_processed'] * 100:.1f}%)")
                print(f"  Average iterations: {results['stats']['avg_iterations']:.2f}")

            except Exception as e:
                print(f"Error processing model {model_path.name}: {e}")
                total_stats["models_failed"] += 1

            finally:
                del model, tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        print(f"\n{target_level} LEVEL ITERATIVE PROCESSING COMPLETED!")
        print(f"Models processed: {total_stats['models_processed']}")
        print(f"Models failed: {total_stats['models_failed']}")
        print(f"Total texts processed: {total_stats['total_texts']}")
        print(f"Total successful: {total_stats['total_successful']}")
        if total_stats['total_texts'] > 0:
            print(
                f"Overall success rate: {total_stats['total_successful'] / total_stats['total_texts'] * 100:.1f}%")
            print(
                f"Average iterations per text: {total_stats['total_iterations'] / total_stats['total_texts']:.2f}")

    def analyze_results_for_level(self, target_level: str):
        """
        Analyze and display comprehensive statistics for all processed texts at a specific CEFR level.
        Includes model comparison, iteration distribution, and success rates.
        """
        print(f"\n{'=' * 60}")
        print(f"ANALYZING ITERATIVE RESULTS FOR {target_level} LEVEL")
        print(f"{'=' * 60}")

        result_files = list(self.output_dir.glob(f"*_{target_level}_iterative_simplifications.json"))

        if not result_files:
            print(f"No result files found for level {target_level}")
            return

        all_results = []
        model_stats = {}

        for result_file in result_files:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            model_name = data['model_name']
            model_stats[model_name] = {
                'total_processed': data['stats']['total_processed'],
                'successful': data['stats']['successful'],
                'success_rate': data['stats']['successful'] / data['stats']['total_processed'] * 100 if
                data['stats']['total_processed'] > 0 else 0,
                'avg_iterations': data['stats']['avg_iterations'],
                'texts_by_iterations': data['stats']['texts_by_iterations']
            }

            all_results.extend(data['iterative_results'])

        total_texts = len(all_results)
        successful_texts = sum(1 for r in all_results if r['target_achieved'])
        total_iterations = sum(r['iterations_used'] for r in all_results)

        print(f"Total texts processed: {total_texts}")
        print(f"Successful texts: {successful_texts}")
        print(f"Overall success rate: {successful_texts / total_texts * 100:.1f}%")
        print(f"Average iterations per text: {total_iterations / total_texts:.2f}")

        print(f"\nMODEL COMPARISON:")
        print(f"{'Model':<30} {'Success Rate':<12} {'Avg Iterations':<15} {'Total Processed'}")
        print("-" * 70)
        for model_name, stats in sorted(model_stats.items(), key=lambda x: x[1]['success_rate'], reverse=True):
            print(
                f"{model_name:<30} {stats['success_rate']:<11.1f}% {stats['avg_iterations']:<14.2f} {stats['total_processed']}")

        iteration_counts = {}
        for result in all_results:
            iterations = result['iterations_used']
            if iterations not in iteration_counts:
                iteration_counts[iterations] = {'total': 0, 'successful': 0}
            iteration_counts[iterations]['total'] += 1
            if result['target_achieved']:
                iteration_counts[iterations]['successful'] += 1

        print(f"\nITERATION DISTRIBUTION:")
        print(f"{'Iterations':<12} {'Total Texts':<12} {'Successful':<12} {'Success Rate'}")
        print("-" * 50)
        for iterations in sorted(iteration_counts.keys()):
            stats = iteration_counts[iterations]
            success_rate = stats['successful'] / stats['total'] * 100
            print(f"{iterations:<12} {stats['total']:<12} {stats['successful']:<12} {success_rate:.1f}%")

        def main():
            """
            Main execution function for CEFR iterative simplification pipeline.
            Loads test data, processes all CEFR levels, and generates comprehensive evaluation metrics.
            """
            MODELS_DIR = "axolotl_finetune/models/"
            MAX_ITERATIONS = 5

            data = []
            with open("tsar2025_test_blind.jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))

            df = pd.DataFrame(data)
            print(f"DataFrame loaded: {len(df)} rows")

            required_columns = ['target_cefr', 'text_id', 'original']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Missing columns: {missing_columns}")
                return

            cefr_levels = df['target_cefr'].unique()
            print(f"Found CEFR levels: {cefr_levels}")

            tester = CEFRIterativeSimplificationTester(
                models_dir=MODELS_DIR,
                max_iterations=MAX_ITERATIONS
            )

            print("=" * 60)
            print("STARTING ITERATIVE A2 SIMPLIFICATION")
            print("=" * 60)
            tester.run_simplification_for_level(df, 'A2')
            tester.analyze_results_for_level('A2')