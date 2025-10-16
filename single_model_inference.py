import json
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, root_mean_squared_error
from transformers import pipeline
from typing import Dict, List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# CEFR level configuration
CEFR_LABELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
LABEL2IDX = {label: idx for idx, label in enumerate(CEFR_LABELS)}

# Load CEFR evaluation models
cefr_labeler1 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-doc_en-Cefr")
cefr_labeler2 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr")
cefr_labeler3 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-reference_AllLang2-Cefr2")


def get_cefr_labels(simplifications: list, models=[cefr_labeler1, cefr_labeler2, cefr_labeler3]):
    """Predict CEFR labels using ensemble of three models like in TSAR 2025 evaluation script using the proposed models for CEFR classification"""
    cefr_labels = []
    for simplification in simplifications:
        top_preds = (model(simplification)[0] for model in models)
        best = max(top_preds, key=lambda d: d["score"])
        cefr_labels.append(best["label"])
    return cefr_labels


def get_cefr_compliance_score(simplifications: list, reference_levels: list,
                              models=[cefr_labeler1, cefr_labeler2, cefr_labeler3]):
    """Calculate compliance metrics between predicted and reference CEFR levels"""
    assert len(simplifications) == len(
        reference_levels), "The number of simplifications is different of the number of reference_levels."

    predicted_labels = get_cefr_labels(simplifications=simplifications, models=models)
    f1 = f1_score(reference_levels, predicted_labels, average='weighted')

    true_idx = np.array([LABEL2IDX[l] for l in reference_levels])
    pred_idx = np.array([LABEL2IDX[l] for l in predicted_labels])

    adj_acc = (np.abs(true_idx - pred_idx) <= 1).mean()
    rmse = root_mean_squared_error(true_idx, pred_idx)

    return {'weighted_f1': round(f1, 4),
            'adj_accuracy': round(adj_acc, 4),
            'rmse': round(rmse, 4)}


def evaluate_single_text_cefr(text, target_level, models=[cefr_labeler1, cefr_labeler2, cefr_labeler3]):
    """Evaluate a single text and return predicted level with metrics"""
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


class CEFRSimplificationModel:
    """Wrapper for fine-tuned Llama model for CEFR-targeted text simplification"""

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Set pad_token for Llama models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # System prompt used during model fine-tuning
        # This can be modified based on your fine-tuning configuration and target CEFR levels
        self.system_prompt = """You are an expert on CEFR with deep understanding of automatic language assessment systems.

B1 LANGUAGE REQUIREMENTS:
- Vocabulary: 2000-3000 most common English words, avoid academic/formal terms
- Sentences: 15-22 words, can connect 2 related ideas with clear logic
- Grammar: Present perfect (have/has done), simple passive (is/was done), basic conditionals (if...will/would), modals (should, might, could, would)
- Connectors: however, although, while, since, unless, because, so that, even though
- Style: Clear intermediate language that shows reasoning and personal opinions

STRICT LEVEL CONTROL:
- Above A2: Include abstract concepts with simple explanation, cause-effect relationships, personal opinions with basic justification, intermediate grammar patterns
- Below B2: No academic/formal vocabulary (facilitate→help, demonstrate→show, utilize→use), no complex conditional structures, no sophisticated argumentation, no specialized terminology without explanation
- PRECISE B1 TARGET: Intermediate complexity using everyday vocabulary - never oversimplify to A2, never undersimplify leaving B2+ elements

CRITICAL B1 DIFFERENTIATORS:
- From A2: Can handle abstract ideas but explains them simply using common words
- From B2: Uses everyday vocabulary even for complex concepts, avoids formal/academic tone
- B1 signature: Connects ideas logically but with simple language patterns


A2 LANGUAGE REQUIREMENTS:
- Vocabulary: Most common 1500 English words only
- Sentences: 8-12 words, one clear idea per sentence
- Grammar: Simple present/past, basic future (will), basic modals (can/must/should)
- Connectors: and, but, because, so, when, if, then
- Style: Personal, concrete, everyday language

STRICT LEVEL CONTROL:
- Above A1: Include personal experiences, feelings, plans, time references
- Below B1: No present perfect, passive voice, or complex connectors (however, although, despite)
- Below B1: No abstract concepts without concrete explanation"""

    def generate_simplification(self, instruction: str, input_text: str = "", max_length: int = 2048,
                                temperature: float = 0.7) -> str:
        """Generate simplification using Llama chat format with system prompt"""

        # Build prompt in Llama format
        if input_text.strip():
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{self.system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}

{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        else:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{self.system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        # Tokenize and move to model device
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        # Generate with attention mask
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2048,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        # Decode only the newly generated tokens
        input_length = input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        full_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Clean up any remaining special tags
        if "<|eot_id|>" in full_response:
            full_response = full_response.split("<|eot_id|>")[0]

        return full_response.strip()


def get_task_instruction(target_level: str, predicted_level: str = None, is_correct: bool = None) -> str:
    """Generate instruction prompt based on target level and previous prediction"""

    target_level = target_level.lower()

    def get_simplification_prompt(original_text, target_level, current_level=None, iteration=0):
        """Build the complete simplification prompt with optional feedback"""
        base_feedback = ""
        if current_level and iteration > 0:
            if current_level != target_level.upper():
                base_feedback = f"\nCURRENT ISSUE: The previous version was classified as {current_level}, but we need {target_level.upper()} level. "

                if current_level > target_level.upper():
                    base_feedback += "The text is TOO COMPLEX. Simplify more aggressively."
                else:
                    base_feedback += "The text is TOO SIMPLE. Add more complexity while staying at target level."

        if target_level.lower() == 'a2':
            return f"""You are a language teacher simplifying texts to A2 CEFR level.

OBJECTIVE: Transform this text to A2 level while preserving all original meaning and information. Return only the simplified text.
{base_feedback}

A2 LANGUAGE REQUIREMENTS:
- Vocabulary: Most common 1500 English words only
- Sentences: 8-12 words, one clear idea per sentence
- Grammar: Simple present/past, basic future (will), basic modals (can/must/should)
- Connectors: and, but, because, so, when, if, then
- Style: Personal, concrete, everyday language

STRICT LEVEL CONTROL:
- Above A1: Include personal experiences, feelings, plans, time references
- Below B1: No present perfect, passive voice, or complex connectors (however, although, despite)
- Below B1: No abstract concepts without concrete explanation

TRANSFORMATION PROCESS:
1. Identify all key information and meaning
2. Break complex sentences into simple A2 structures
3. Replace advanced vocabulary with A2 equivalents
4. Convert complex grammar to simple A2 patterns
5. Verify all original meaning is preserved

CRITICAL: Do not omit, summarize, or change any information. Only change HOW it's expressed.

Return only the simplified text. Do not include any other comments, notes, or additional information."""

        elif target_level.lower() == 'b1':
            return f"""You are an expert CEFR B1 text simplification specialist with deep understanding of automatic language assessment systems.

OBJECTIVE: Transform this text to precise B1 level while preserving all original meaning and information. Return only the simplified text.
{base_feedback}

B1 LANGUAGE REQUIREMENTS:
- Vocabulary: 2000-3000 most common English words, avoid academic/formal terms
- Sentences: 15-22 words, can connect 2 related ideas with clear logic
- Grammar: Present perfect (have/has done), simple passive (is/was done), basic conditionals (if...will/would), modals (should, might, could, would)
- Connectors: however, although, while, since, unless, because, so that, even though
- Style: Clear intermediate language that shows reasoning and personal opinions

STRICT LEVEL CONTROL:
- Above A2: Include abstract concepts with simple explanation, cause-effect relationships, personal opinions with basic justification, intermediate grammar patterns
- Below B2: No academic/formal vocabulary (facilitate→help, demonstrate→show, utilize→use), no complex conditional structures, no sophisticated argumentation, no specialized terminology without explanation
- PRECISE B1 TARGET: Intermediate complexity using everyday vocabulary - never oversimplify to A2, never undersimplify leaving B2+ elements

CRITICAL B1 DIFFERENTIATORS:
- From A2: Can handle abstract ideas but explains them simply using common words
- From B2: Uses everyday vocabulary even for complex concepts, avoids formal/academic tone
- B1 signature: Connects ideas logically but with simple language patterns

TRANSFORMATION PROCESS:
1. Identify all key information and meaning
2. Scan for B2+ vocabulary and replace with B1 common equivalents
3. Convert complex sentences to B1 structures (maximum 2 clauses per sentence)
4. Add simple explanations for any remaining complex concepts
5. Include 2-3 B1 grammar markers per paragraph naturally
6. Verify consistent B1 complexity throughout - no A2 oversimplification, no B2+ elements remaining

CRITICAL: Do not omit, summarize, or change any information. Only change HOW it's expressed to match B1 patterns that automatic CEFR classifiers consistently recognize as B1 level.

Return only the simplified text. Do not include any other comments, notes, or additional information."""

    # Determine iteration and current_level for feedback
    iteration = 0 if is_correct or predicted_level is None else 1
    current_level = predicted_level if predicted_level else None

    return get_simplification_prompt("", target_level, current_level, iteration)


def create_chat_format_simplification(original_text: str, target_level: str, predicted_level: str = None,
                                      is_correct: bool = None) -> Dict:
    """Create chat format with complete instruction prompt"""

    # Get base instruction
    base_instruction = get_task_instruction(target_level, predicted_level, is_correct)

    # Append original text to instruction
    full_instruction = f"{base_instruction}\n\nText to simplify:\n\"\"\"\n{original_text}\n\"\"\""

    return {
        "instruction": full_instruction,
        "input": "",
        "target_level": target_level
    }


def iterative_simplification_with_finetuned_model(
        texts_data: List[Dict],
        model_path: str,
        max_iterations: int = 5
) -> Tuple[List[Dict], List[Dict]]:
    """
    Perform iterative simplification with fine-tuned model

    Args:
        texts_data: List of dicts with 'text_id', 'original', 'target_cefr'
        model_path: Path to fine-tuned model
        max_iterations: Maximum number of iterations per text

    Returns:
        results: Final results for each text
        all_simplifications: All generated simplifications across iterations
    """

    # Load model
    print("Loading fine-tuned model...")
    model = CEFRSimplificationModel(model_path)
    print("Model loaded successfully!")

    results = []
    all_simplifications = []

    for idx, data in enumerate(texts_data):
        text_id = data['text_id']
        original_text = data['original']
        target_level = data['target_cefr'].lower()

        print(f"\n=== Processing {text_id} (Target: {target_level.upper()}) ===")

        current_text = original_text
        iteration_history = []

        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}/{max_iterations}")

            # Determine if first iteration or refinement
            if iteration == 0:
                chat_format = create_chat_format_simplification(current_text, target_level)
            else:
                last_level = iteration_history[-1]['predicted_level']
                is_correct = last_level == target_level.upper()
                chat_format = create_chat_format_simplification(
                    current_text, target_level, last_level, is_correct
                )

            # Generate simplification
            simplified_text = model.generate_simplification(
                instruction=chat_format["instruction"],
                input_text=chat_format["input"]
            )

            # Evaluate result
            predicted_level, metrics = evaluate_single_text_cefr(simplified_text, target_level)

            iteration_info = {
                'iteration': iteration + 1,
                'text': simplified_text,
                'predicted_level': predicted_level,
                'weighted_f1': metrics['weighted_f1'],
                'adj_accuracy': metrics['adj_accuracy'],
                'rmse': metrics['rmse'],
                'target_achieved': predicted_level == target_level.upper() if predicted_level else False
            }

            iteration_history.append(iteration_info)

            # Save to all simplifications list
            all_simplifications.append({
                'text_id': text_id,
                'iteration': iteration + 1,
                'target_level': target_level,
                'predicted_level': predicted_level,
                'weighted_f1': metrics['weighted_f1'],
                'adj_accuracy': metrics['adj_accuracy'],
                'rmse': metrics['rmse'],
                'simplified_text': simplified_text,
                'original_text': original_text if iteration == 0 else None,
                'instruction_used': chat_format["instruction"]
            })

            print(f"Prediction: {predicted_level}")
            print(f"F1: {metrics['weighted_f1']}, Adj Acc: {metrics['adj_accuracy']}, RMSE: {metrics['rmse']}")
            print(f"Target achieved: {'Yes' if iteration_info['target_achieved'] else 'No'}")

            # Check if target achieved
            if iteration_info['target_achieved']:
                print(f"Target {target_level.upper()} achieved at iteration {iteration + 1}!")
                break

            # Use simplified text for next iteration
            current_text = simplified_text

        # Save final result
        final_result = {
            'text_id': text_id,
            'original': original_text,
            'target_level': target_level,
            'final_simplified': simplified_text,
            'iterations_used': len(iteration_history),
            'final_predicted_level': iteration_history[-1]['predicted_level'],
            'final_weighted_f1': iteration_history[-1]['weighted_f1'],
            'final_adj_accuracy': iteration_history[-1]['adj_accuracy'],
            'final_rmse': iteration_history[-1]['rmse'],
            'target_achieved': iteration_history[-1]['target_achieved'],
            'iteration_history': iteration_history
        }

        results.append(final_result)

        if final_result['target_achieved']:
            print(f"Success: {text_id} -> {target_level.upper()} in {len(iteration_history)} iterations")
        else:
            print(f"Failed: {text_id} -> {final_result['final_predicted_level']} instead of {target_level.upper()}")

    return results, all_simplifications


def analyze_results(results):
    """Analyze and display results statistics"""
    total_texts = len(results)
    successful_texts = sum(1 for r in results if r['target_achieved'])
    success_rate = successful_texts / total_texts * 100

    print(f"\n{'=' * 50}")
    print(f"ITERATIVE SIMPLIFICATION RESULTS ANALYSIS")
    print(f"{'=' * 50}")
    print(f"Processed texts: {total_texts}")
    print(f"Successful texts: {successful_texts}")
    print(f"Success rate: {success_rate:.1f}%")

    # Statistics by iterations
    iteration_stats = {}
    for result in results:
        iterations_used = result['iterations_used']
        if iterations_used not in iteration_stats:
            iteration_stats[iterations_used] = {'total': 0, 'successful': 0}

        iteration_stats[iterations_used]['total'] += 1
        if result['target_achieved']:
            iteration_stats[iterations_used]['successful'] += 1

    print(f"\nDistribution by iterations:")
    for iterations, stats in sorted(iteration_stats.items()):
        success_rate_iter = stats['successful'] / stats['total'] * 100
        print(f"  {iterations} iterations: {stats['successful']}/{stats['total']} ({success_rate_iter:.1f}% success)")

    # Statistics by levels
    level_stats = {}
    for result in results:
        target = result['target_level']
        if target not in level_stats:
            level_stats[target] = {'total': 0, 'successful': 0}

        level_stats[target]['total'] += 1
        if result['target_achieved']:
            level_stats[target]['successful'] += 1

    print(f"\nPerformance by levels:")
    for level, stats in sorted(level_stats.items()):
        success_rate_level = stats['successful'] / stats['total'] * 100
        print(f"  {level.upper()}: {stats['successful']}/{stats['total']} ({success_rate_level:.1f}% success)")

    return {
        'total_texts': total_texts,
        'successful_texts': successful_texts,
        'success_rate': success_rate,
        'iteration_stats': iteration_stats,
        'level_stats': level_stats
    }


def save_results(results, all_simplifications, output_prefix=""):
    """Save results to JSONL files"""

    # Save final results
    with open(f"{output_prefix}_final.jsonl", 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

    # Save all simplifications
    with open(f"{output_prefix}_all.jsonl", 'w', encoding='utf-8') as f:
        for simplification in all_simplifications:
            json.dump(simplification, f, ensure_ascii=False)
            f.write('\n')

    print(f"Results saved to {output_prefix}_final.jsonl and {output_prefix}_all.jsonl")


def load_test_data(file_path: str) -> List[Dict]:
    """Load test data from JSONL or CSV file"""

    if file_path.endswith('.jsonl'):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Extract only required fields
                    test_item = {
                        'text_id': item['text_id'],
                        'original': item['original'],
                        'target_cefr': item['target_cefr'],
                        'dataset_id': item.get('dataset_id', '')
                    }
                    data.append(test_item)
        return data

    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        return df.to_dict('records')

    else:
        raise ValueError("Unsupported file format. Use .jsonl or .csv")


def main():
    """Main execution function"""

    # Configuration
    MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
    TEST_DATA_PATH = "../tsar2025_test_blind.jsonl"
    MAX_ITERATIONS = 5

    print("CEFR Simplification with Fine-tuned Model")
    print("=" * 50)

    # Load test data
    print(f"Loading test data from {TEST_DATA_PATH}...")
    test_data = load_test_data(TEST_DATA_PATH)
    print(f"Loaded {len(test_data)} texts for processing")

    # Perform iterative simplification
    results, all_simplifications = iterative_simplification_with_finetuned_model(
        texts_data=test_data,
        model_path=MODEL_PATH,
        max_iterations=MAX_ITERATIONS
    )

    # Analyze results
    analysis = analyze_results(results)

    # Calculate final metrics
    final_texts = [r['final_simplified'] for r in results]
    reference_levels = [r['target_level'].upper() for r in results]

    print(f"\n{'=' * 50}")
    print(f"FINAL EVALUATION METRICS")
    print(f"{'=' * 50}")

    compliance_scores = get_cefr_compliance_score(final_texts, reference_levels)
    print(f"Weighted F1: {compliance_scores['weighted_f1']}")
    print(f"Adjacent Accuracy: {compliance_scores['adj_accuracy']}")
    print(f"RMSE: {compliance_scores['rmse']}")

    # Save results
    save_results(results, all_simplifications, "llama_3.1-8b")

    return results, all_simplifications, analysis, compliance_scores


if __name__ == "__main__":
    results, all_simplifications, analysis, compliance_scores = main()