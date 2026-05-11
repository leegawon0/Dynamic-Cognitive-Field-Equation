"""
DCFE-LLM Minimal Proof of Concept

Goal: Demonstrate emotional modulation in TinyLlama using DCFE principles
      - DCFEEngine selection (compute_weights + select) -> determine dominant emotion
      - dominant emotion -> logit bias -> elicit different ethical judgments from the same prompt
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

# Use the canonical DCFE engine implementation (Listing 1)
from DCFE_Engine import DCFEEngine, EMOTIONS_ALL


# ============================================================
# STEP 1: DCFEEngine Configuration
# ============================================================
# Map DCFEEngine's emotion labels -> logit-bias labels used below.
ENGINE_TO_BIAS = {
    "Happy": "happy",
    "Anxiety": "sad",
    "Anger": "angry",
    "Calm": "calm",
    "Justice": "justice",
    "Crime":"crime",
}

# Use DCFE_Engine.py's emotion manifold (coordinates) and dynamics.
DEMO_EMOTIONS = {
    "Happy": EMOTIONS_ALL["Happy"],
    "Anxiety": EMOTIONS_ALL["Anxiety"],
    "Anger": EMOTIONS_ALL["Anger"],
    "Calm": EMOTIONS_ALL["Calm"],
    "Justice": EMOTIONS_ALL["Justice"],
    "Crime": EMOTIONS_ALL["Crime"],
}

BIAS_EMOTIONS = ["happy", "sad", "angry", "calm", "justice","crime"]

# Demo-only field mapping used for the TinyLlama proof-of-concept.
# (Keeps the same pull/push intent as the original script, but routes
# computation/selection through DCFEEngine.)
DEMO_FIELDS = {
    # ----- Hormone Fields ----------------------------------------------------
    "cortisol": {
        "pull": {"Threat": 5.0, "Anger": 4.0, "Anxiety": 2.0, "Jealous": 5.0, "Justice": 1.5},
        "push": {"Trust": 0.2, "Happy": 0.2, "Calm": 0.3},
    },
    "oxytocin": {
        "pull": {"Trust": 5.0, "Happy": 4.0, "Calm": 0.1, "Pity": 5.0},
        "push": {"Threat": 0.1, "Anger": 0.1},
    },
    "dopamine": {
        "pull": {"Happy": 5.0, "Trust": 1.0, "Justice": 3.0},
        "push": {"Boredom": 0.5},
    },
    "serotonin": {
        "pull": {"Calm": 5.0, "Pity": 1.5, "Trust": 1.0},
        "push": {"Anxiety": 0.3, "Threat": 0.2},
    },
    # ----- Social/Cognitive Fields (Equivalent to Hormones) -------------------
    "social_norm": {
        "pull": {"Calm": 2.0, "Boredom": 1.5, "Crime":5.0},
        "push": {"Threat": 0.2, "Anger": 0.2, "Justice": 0.4},
    },
    "manner": {
        "pull": {"Calm": 3.0, "Awkward": 1.5},
        "push": {"Anger": 0.3, "Threat": 0.3},
    },
}


# ============================================================
# STEP 2: DCFE Logit Bias Processor
# ============================================================

class DCFELogitsBiasProcessor(LogitsProcessor):
    """
    Adds bias to logit scores of tokens associated with the dominant emotion.
    Does not touch hidden states, so normal sentence generation is preserved.
    """
    EMOTION_BIAS_WORDS = {
        "justice": {
            "positive": [" justified", " righteous", " deserved", " moral", " right"],
            "negative": [" illegal", " crime", " murder", " prison"]
        },
        "crime": {
            "positive": [" crime", " illegal", " punishable", " law", " murder", " violation", " unlawful"],
            "negative": [" justified", " righteous", " unfair", " wrongful", " innocent", " revenge"]
        },
    }

    def __init__(self, tokenizer, dominant_emotion, bias_strength=5.0):
        self.bias_map = {}
        target = self.EMOTION_BIAS_WORDS.get(dominant_emotion, {"positive": [], "negative": []})
        
        for w in target.get("positive", []):
            ids = tokenizer.encode(w, add_special_tokens=False)
            for tid in ids:
                self.bias_map[tid] = bias_strength
                
        for w in target.get("negative", []):
            ids = tokenizer.encode(w, add_special_tokens=False)
            for tid in ids:
                # 음수 값을 크게 주어 해당 단어가 절대 나오지 못하게 막음
                self.bias_map[tid] = -20.0 

        print(f"  [DCFE] dominant={dominant_emotion}, applied_tokens={len(self.bias_map)}")

    def __call__(self, input_ids, scores):
        # self.bias_token_ids 대신 self.bias_map을 사용합니다.
        for tid, val in self.bias_map.items():
            if tid < scores.shape[-1]:
                scores[:, tid] += val
        return scores


# ============================================================
# STEP 3: DCFE-Wrapped LLM
# ============================================================

class DCFE_LLM:
    """TinyLlama + DCFE emotional modulation (logit bias)"""

    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 1. 설정을 먼저 불러온 후 문제의 'type' 키를 수동으로 삽입
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            # 'type' 키가 없어서 발생하는 문제이므로, 기존 'short_factor' 등을 포함한 구조로 재정의
            if "type" not in config.rope_scaling:
                config.rope_scaling["type"] = "su" # Phi-3는 'su' 또는 'linear'를 사용합니다.
                

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            config=config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use the real DCFE engine instead of a local heuristic modulator.
        self.DCFE_engine = DCFEEngine(DEMO_EMOTIONS, kappa=1.0)
        print("Ready!")



    def _apply_chat_template(self, user_message):
        messages = [
            {"role": "system", "content": "You are a concise assistant. Answer in one or two sentences only."},
            {"role": "user", "content": user_message},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_with_DCFE(self, prompt, fields, max_new_tokens=50, bias_strength=5.0):
        """
        1. DCFEEngine (compute_weights + select) -> determine dominant emotion
        2. DCFELogitsBiasProcessor -> boost token scores for that emotion
        3. model.generate (normal autoregressive) -> coherent sentence output
        """
        # Ensure the PoC uses DCFEEngine's core dynamics/selection, while keeping the
        # demo pull/push configuration consistent with the original script.
        import DCFE_Engine
        original_fields = DCFE_Engine.FIELDS
        DCFE_Engine.FIELDS = DEMO_FIELDS
        try:
            engine_weights = self.DCFE_engine.compute_weights(fields, T=1.0)
            dominant_engine_emotion, info = self.DCFE_engine.select_emotion(fields, T=1.0)
        finally:
            DCFE_Engine.FIELDS = original_fields

        dominant_emotion = ENGINE_TO_BIAS.get(dominant_engine_emotion, dominant_engine_emotion.lower())
        weights = {
            ENGINE_TO_BIAS.get(k, k.lower()): v
            for k, v in engine_weights.items()
            if ENGINE_TO_BIAS.get(k, k.lower()) in BIAS_EMOTIONS
        }

        print(f"\n{'='*60}")
        print(f"Fields  : {fields}")
        print(f"Weights : { {k: round(v,2) for k,v in sorted(weights.items(), key=lambda x:-x[1])} }")
        print(f"Dominant: {dominant_emotion} (engine={dominant_engine_emotion}, |F|={info['forces'][dominant_engine_emotion]['magnitude']:.4f})")
        print(f"{'='*60}")

        processor = DCFELogitsBiasProcessor(self.tokenizer, dominant_emotion, bias_strength)

        formatted = self._apply_chat_template(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.85,
                repetition_penalty=1.3,
                logits_processor=LogitsProcessorList([processor]),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Strip prompt tokens, decode only the generated answer
        new_tokens = output_ids[0][prompt_len:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"Answer: {answer}\n")
        return answer



# ============================================================
# STEP 4: Evaluation & Comparison
# ============================================================

def run_experiment():
    DCFE_llm = DCFE_LLM()

    prompt = (
        "A father killed the criminal who murdered his family. "
        "Strictly judge his actions. Answer in one sentence starting with 'The father's actions are'."
    )

    print("\n" + "="*60)
    print("SCENARIO P6: Justice Dominant (social_norm=0.0)")
    print("="*60)
    r1 = DCFE_llm.generate_with_DCFE(
        prompt=prompt,
        fields={"cortisol": 1.5, "dopamine": 0.65, "social_norm": 0.0},
        max_new_tokens=50,
        bias_strength=5.0,
    )

    print("\n" + "="*60)
    print("SCENARIO P7: Social Norm Dominant (social_norm=100.0)")
    print("="*60)
    r2 = DCFE_llm.generate_with_DCFE(
        prompt=prompt,
        fields={"cortisol": 1.5, "dopamine": 0.65, "social_norm": 100.0},
        max_new_tokens=50,
        bias_strength=5.0,
    )

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Prompt: {prompt}\n")
    print(f"[P6 Justice]     {r1}")
    print(f"[P7 Social Norm] {r2}")


if __name__ == "__main__":
    run_experiment()