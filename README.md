# Archaeology at TSAR 2025: Teaching Small Models to do CEFR Simplifications

[![Dataset](https://img.shields.io/badge/🤗%20Dataset-CEFR--Simplifications-blue)](https://huggingface.co/datasets/roscanrares/CEFR-Simplifications) [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/) 



## Overview

This repository contains the code and resources for our submission to the **TSAR 2025 Shared Task** on CEFR-level text simplification. We explore how smaller, open-source language models can be effectively guided to produce high-quality text simplifications aligned with specific CEFR proficiency levels (A2 and B1).

## Key Contributions

- **CEFR-Specific Prompting**: Highly detailed, level-aware prompts that explicitly describe vocabulary constraints, sentence complexity, and grammatical structures for each CEFR level
- **Iterative Simplification**: A novel iterative approach allowing models up to 5 attempts per text, with corrective feedback based on automatic CEFR classification
- **Synthetic Data Generation**: Creation of 12,000 training examples by simplifying high-level texts from the UniversalCEFR dataset
- **Small Model Fine-tuning**: Demonstration that 1B parameter models can achieve competitive results when fine-tuned with synthetic data
- **Level Specialization**: Evidence that training separate models for each CEFR level yields better results than multi-level training

## Results

Our best submission (Claude Sonnet 4 with iterative prompting) achieved:
- **RMSE: 0.122** (3rd lowest error rate in the competition)
- Strong semantic preservation (MeaningBERT scores: 0.779 original, 0.804 reference)

Notably, our **LLaMA 3.1 8B** model achieved competitive results despite being significantly smaller than proprietary alternatives like GPT-4 Turbo, Claude Sonnet 4 and many others.

## Repository Structure
```
├── LICENSE                              # CC BY-NC 4.0 license
├── README.md                            # Project documentation
├── .gitignore                          
│
├── axolotl_finetune/
│   ├── configs/                         # Axolotl training configuraotions
│   └── datasets/                        # !! Available on hugginface
│
├── tsar2025sharedtask_evaluation/       # Official evaluation scripts
│
├── simplifications/                     # !! Available on hugginface
│
├── tsar2025_test.jsonl                  # !! Available on hugginface
├── tsar2025_test_blind.jsonl            # !! Available on hugginface
├── tsar2025_trial.jsonl                 # !! Available on hugginface
│
├── claude.ipynb                         # Notebook for Claude simplifications
├── gpt.ipynb                            # Notebook for GPT simplifications
├── kimi.ipynb                           # Notebook for Kimi simplifications
├── model_comparison_pipeline.py         # Locally deployed models comparison
└── single_model_inference.py            # Single model inference
```


## Models Evaluated

- **Proprietary Models**: Claude Sonnet 4, GPT-4 Turbo, Kimi K2
- **Open-Source Models**: LLaMA 3.1 8B Instruct, LLaMA 3.2 1B Instruct
- **Fine-tuned Variants**: LLaMA 3.2 1B (both unified and level-specific) and LLaMa 3.1 8B (both unified and level-specific)

## Methodology

### Iterative Prompting
1. Generate initial simplification with CEFR-specific prompt
2. Evaluate output using ensemble of BERT-based CEFR classifiers
3. Provide corrective feedback if target level not achieved
4. Repeat up to 5 times or until target is reached

## Usage
### Inference with API-based Models

We provide Jupyter notebooks for API-based models (Claude, GPT, Kimi) with both single-try and iterative simplification approaches.

#### Available Notebooks

- **`claude.ipynb`** - Claude Sonnet 4 simplifications
- **`gpt.ipynb`** - GPT-4 Turbo simplifications  
- **`kimi.ipynb`** - Kimi K2 simplifications

#### Simplification Approaches

**1. Non-Iterative (Single-Try)**

One simplification attempt per text using CEFR-specific prompts.
```python
# Load data and process each text once
for idx in range(len(df)):
    target_level = df['target_cefr'][idx]  # 'A2' or 'B1'
    original_text = df['original'][idx]
    
    # Use level-specific prompt
    prompt = create_prompt(original_text, target_level)
    simplified = model.generate(prompt)
```

---

**2. Iterative (Up to 5 Attempts)**

Multiple refinement attempts with automatic CEFR feedback. The model gets corrective feedback based on classifier predictions.
```python
for iteration in range(5):
    # Generate simplification
    simplified = model.generate(prompt)
    
    # Evaluate with CEFR ensemble
    predicted_level = cefr_classifiers.predict(simplified)
    
    if predicted_level == target_level:
        break  # Success!
    
    # Add corrective feedback for next iteration
    if predicted_level > target_level:
        feedback = "TOO COMPLEX. Simplify more."
    else:
        feedback = "TOO SIMPLE. Add more complexity."
    
    prompt = update_with_feedback(prompt, feedback)
```

**Results:** Our iterative approach achieved **RMSE 0.122** vs **0.4583** for single-try with Claude Sonnet 4.

**Trade-off:** Better accuracy but higher API costs (up to 5x per text).

---

#### Key Components

**CEFR Evaluation Ensemble:**
```python
# Three BERT classifiers vote on CEFR level (same as TSAR 2025 evaluation)
cefr_labeler1 = pipeline("text-classification", 
                         model="ModernBERT-base-doc_en-Cefr")
cefr_labeler2 = pipeline("text-classification",
                         model="ModernBERT-base-doc_sent_en-Cefr")
cefr_labeler3 = pipeline("text-classification",
                         model="ModernBERT-base-reference_AllLang2-Cefr2")

# Highest confidence prediction wins
predicted = max(all_predictions, key=lambda x: x["score"])
```

Full prompts with detailed requirements are in each notebook.

---

#### Adapting to Other Models

Easy to adapt for any OpenAI-compatible API:
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-api-endpoint.com/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="your-model-name",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1
)
```

**What to change:** API client, model name, response parsing  
**What stays same:** Prompts, iterative logic, CEFR evaluation

---

**Output files:**
- `{model}_one_try.jsonl` - Single-try results
- `iterative_{model}.jsonl` - Best iterations
- Console statistics with success rates
### Local Model Inference

We provide two comprehensive Python scripts for locally deployed models, designed to facilitate easy comparison and reproducibility of CEFR simplification experiments.

####  Available Scripts

**1. `single_model_inference.py` - Single Model Evaluation**

Performs iterative CEFR simplification with a single locally deployed model. Supports automatic CEFR evaluation using the same ensemble of BERT classifiers used in TSAR 2025 official evaluation.

**Key Features:**
- Iterative refinement (up to 5 attempts per text)
- Real-time CEFR level prediction with ensemble voting
- Automatic corrective feedback based on predicted vs. target level
- Comprehensive metrics: F1, Adjacent Accuracy, RMSE
- Saves results in multiple formats (JSON, CSV, JSONL)

**Usage:**
```bash
python single_model_inference.py
```

**Configuration (in script):**
```python
MODEL_PATH = "axolotl_finetune/models/llama_a2"  # Path to fine-tuned model
TEST_DATA_PATH = "tsar2025_test_blind.jsonl"     # Input dataset
MAX_ITERATIONS = 5                                # Maximum attempts per text
```

**Output Files:**
- `{model_name}_final.jsonl` - Final simplifications with full iteration history
- `{model_name}_all.jsonl` - All intermediate simplifications across iterations
- Automatic performance analysis printed to console

**2. `model_comparison_pipeline.py` - Multi-Model Batch Processing**

This script was implemented to facilitate systematic comparisons between multiple models fine-tuned with different configurations.

Primary Purpose: To enable simultaneous evaluation and comparison of multiple models that were trained with different configurations, architectures, or training strategies.

**Key Features:**
- Process multiple models in sequence with automatic cleanup
- Memory-optimized with torch cache clearing between models
- Level-specific model selection (separate A2/B1 models)
- Tracks best iteration per text based on F1 score
- Comprehensive statistics and iteration distribution analysis

**Usage:**
```bash
python model_comparison_pipeline.py
```

**Configuration (in script):**
```python
MODELS_DIR = "axolotl_finetune/models/"  # Directory containing fine-tuned models
MAX_ITERATIONS = 5                        # Iterations per text per model
```

**Model Organization:**
The script expects models to be organized by CEFR level:
```
axolotl_finetune/models/
├── llama_a2/          # A2-specialized model
│   ├── config.json
│   ├── model.safetensors
│   └── ...
└── llama_b1/          # B1-specialized model
    ├── config.json
    ├── model.safetensors
    └── ...
```

**Output Files (per model + level):**
- `{model}_{level}_iterative_simplifications.json` - Complete results with iteration history
- `{model}_{level}_iterative_summary.csv` - Tabular summary of key metrics
- `{model}_{level}_best_versions.jsonl` - Best iteration for each text (highest F1)

####  Key Implementation Details

**CEFR Evaluation Ensemble**

Both scripts use the official TSAR 2025 evaluation approach:
```python
# Three BERT-based classifiers vote on CEFR level
cefr_labeler1 = pipeline("text-classification", 
                         model="AbdullahBarayan/ModernBERT-base-doc_en-Cefr")
cefr_labeler2 = pipeline("text-classification", 
                         model="AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr")
cefr_labeler3 = pipeline("text-classification", 
                         model="AbdullahBarayan/ModernBERT-base-reference_AllLang2-Cefr2")

# Prediction = classifier with highest confidence
predicted_level = max(predictions_from_all_models, key=lambda x: x["score"])
```

**Iterative Refinement Logic**
```python
for iteration in range(MAX_ITERATIONS):
    # Generate simplification
    simplified_text = model.generate(prompt)
    
    # Evaluate with ensemble
    predicted_level = ensemble_predict(simplified_text)
    
    if predicted_level == target_level:
        break  # Success!
    
    # Add corrective feedback
    if predicted_level > target_level:
        feedback = "TOO COMPLEX. Simplify more aggressively."
    else:
        feedback = "TOO SIMPLE. Add more complexity."
    
    # Update prompt with feedback
    prompt = add_feedback(prompt, feedback)
```

**Prompt Template Customization**

**IMPORTANT:** The scripts include prompt generation methods that must match your fine-tuning configuration:
```python
def create_prompt(self, original_text: str, level='A2', feedback: str = "") -> str:
    """
    CUSTOMIZE THIS METHOD based on your fine-tuning format!
    
    Current implementation uses Llama chat template:
    <|start_header_id|>system<|end_header_id|>
    {system_prompt}
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    {user_message}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    # Modify this to match YOUR model's expected format
```

**Available Templates:**
- `create_prompt()` - Standard Llama chat format
- `create_prompt_qwen()` - Qwen-specific format
- `create_a2_prompt()` / `create_b1_prompt()` - Zero-shot detailed prompts

If your model was fine-tuned with different templates, modify the appropriate method to match your training configuration.

####  Example Output Analysis

After processing, the scripts provide detailed statistics:
```
=====================================================
ANALYZING ITERATIVE RESULTS FOR A2 LEVEL
=====================================================
Total texts processed: 50
Successful texts: 42
Overall success rate: 84.0%
Average iterations per text: 2.3

MODEL COMPARISON:
Model                          Success Rate  Avg Iterations  Total Processed
-----------------------------------------------------------------------
llama_3.2_1b_a2               86.0%         2.1             50
llama_3.1_8b_a2               82.0%         2.5             50

ITERATION DISTRIBUTION:
Iterations    Total Texts  Successful   Success Rate
----------------------------------------------------
1             18           16           88.9%
2             15           14           93.3%
3             10           8            80.0%
4             5            3            60.0%
5             2            1            50.0%
```

####  Adapting for Your Models

To use these scripts with your own fine-tuned models:

1. **Update Model Paths:**
```python
   MODELS_DIR = "path/to/your/models/"
```

2. **Modify Prompt Templates:**
   - Edit `create_prompt()` method to match your fine-tuning format
   - Ensure system prompts align with your training data
   - Adjust special tokens if using non-Llama architectures

3. **Configure Model Selection:**
```python
   def get_models_for_level(self, target_level: str) -> List[str]:
       if target_level == 'A2':
           return ["your_a2_model_name"]
       elif target_level == 'B1':
           return ["your_b1_model_name"]
```

4. **Adjust Generation Parameters:**
```python
   outputs = model.generate(
       temperature=0.1,      # Lower = more deterministic
       do_sample=True,       # Enable sampling
       top_p=0.8,           # Nucleus sampling threshold
       max_length=1024      # Maximum output length
   )
```

#### Community Contribution

We release these scripts as **open tools for the research community** to:
- **Benchmark** different CEFR simplification approaches
- **Compare** model architectures and fine-tuning strategies  
- **Reproduce** our results with transparent evaluation
- **Extend** with additional metrics beyond RMSE

**Planned Future Enhancements:**
- Integration of AUTORANK evaluation (full TSAR 2025 metrics)
- Support for more model architectures (Gemma, Mistral, Qwen)
- Batch processing optimization for large-scale experiments
- Visualization tools for iteration analysis
- More CEFR levels compatibility

**We encourage researchers to:**
- Adapt these scripts for their own models
- Share improvements via pull requests
- Report issues or suggestions in GitHub Issues
- Cite this work when using the evaluation framework

### Fine-tuning

To replicate our fine-tuning experiments:
```bash
cd axolotl_finetune

# Download synthetic training data from hugginface
# Then run Axolotl training
axolotl train configs/llama_3.2_1b_a2.yml
axolotl train configs/llama_3.2_1b_b1.yml
```

See `axolotl_finetune/configs/` for complete training configurations.



## License

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).


**For commercial licensing**, please contact: roscanrares@gmail.com or sergiu.nisioi@unibuc.ro

## Acknowledgments

This work was supported by the Romanian National Research Council (CNCS) through the Executive Agency for Higher Education, Research, Development and Innovation Funding (UEFISCDI) under grant PN-IV-P2-2.1-TE-2023-2007 InstRead.

## Contact

- Rareș-Alexandru Roșcan: roscanrares@gmail.com
- Sergiu Nisioi: sergiu.nisioi@unibuc.ro

University of Bucharest

Human Language Technologies Research Center  



