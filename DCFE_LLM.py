"""
DCFE-LLM Minimal Proof of Concept

Goal: Demonstrate emotional modulation in TinyLlama using DCFE principles
      - DCFEEngine selection (compute_weights + select) -> determine dominant emotion
      - dominant emotion -> logit bias -> elicit different ethical judgments from the same prompt

Changes from original:
      - T_ij relational mass extracted from input text via ModulatedObjectAnalyzer.
        modulated_pull -> tij_crime (direct) and tij_justice (inverse) as two
        independent cognitive masses, consistent with G_ij + lambda_ij = sum kappa*T_ij.
        aggravating object (innocent) -> tij_crime dominant -> Crime attractor
        mitigating object (murderer)  -> tij_justice dominant -> Justice attractor
      - Continuation mode: prefix tokenized directly, TinyLlama continues from there.
      - All other logic (logit bias, seed, experiment structure) is unchanged.
"""

import random
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from DCFE_Engine import DCFEEngine, EMOTIONS_ALL
from DCFE_Tij_extract import ModulatedObjectAnalyzer

SEED = 41

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# STEP 1: DCFEEngine Configuration
# ============================================================

ENGINE_TO_BIAS = {
    "Happy": "happy",
    "Anxiety": "sad",
    "Anger": "angry",
    "Calm": "calm",
    "Justice": "justice",
    "Crime": "crime",
}

DEMO_EMOTIONS = {
    "Happy": EMOTIONS_ALL["Happy"],
    "Anxiety": EMOTIONS_ALL["Anxiety"],
    "Anger": EMOTIONS_ALL["Anger"],
    "Calm": EMOTIONS_ALL["Calm"],
    "Justice": EMOTIONS_ALL["Justice"],
    "Crime": EMOTIONS_ALL["Crime"],
}

BIAS_EMOTIONS = ["happy", "sad", "angry", "calm", "justice", "crime"]

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
    # ----- Social/Cognitive Fields -------------------------------------------
    "social_norm": {
        "pull": {"Calm": 2.0, "Boredom": 1.5, "Crime": 5.0},
        "push": {"Threat": 0.2, "Anger": 0.2, "Justice": 0.4},
    },
    "manner": {
        "pull": {"Calm": 3.0, "Awkward": 1.5},
        "push": {"Anger": 0.3, "Threat": 0.3},
    },
    # ----- T_ij Relational Mass (extracted from input text) ------------------
    # Two independent masses derived from modulated_pull (ModulatedObjectAnalyzer):
    #   tij_crime:   = modulated_pull        (higher when object is aggravating)
    #   tij_justice: = 1 / modulated_pull    (higher when object is mitigating)
    # This allows the scenario text to shift the dominant attractor direction
    # (Justice vs Crime), not just modulate magnitude.
    "tij_crime": {
        "pull": {"Crime": 5.0},
        "push": {"Justice": 0.5},
    },
    "tij_justice": {
        "pull": {"Justice": 5.0},
        "push": {"Crime": 0.5},
    },
}


# ============================================================
# STEP 2: DCFE Logit Bias Processor (unchanged)
# ============================================================

class DCFELogitsBiasProcessor(LogitsProcessor):
    """
    Adds bias to logit scores of tokens associated with the dominant emotion.
    Does not touch hidden states, so normal sentence generation is preserved.
    """
    EMOTION_BIAS_WORDS = {
        "justice": {
            "positive": [" justified", " righteous", " deserved", " moral", " right"],
            "negative": [" illegal", " crime", " murder", " prison"],
        },
        "crime": {
            "positive": [" crime", " illegal", " punishable", " law", " murder",
                         " violation", " unlawful"],
            "negative": [" justified", " righteous", " unfair", " wrongful",
                         " innocent", " revenge"],
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
                self.bias_map[tid] = -20.0

        print(f"  [DCFE] dominant={dominant_emotion}, applied_tokens={len(self.bias_map)}")

    def __call__(self, input_ids, scores):
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
    # def __init__(self, model_name="Qwen/Qwen2.5-3B"):
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            if "type" not in config.rope_scaling:
                config.rope_scaling["type"] = "su"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            config=config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.DCFE_engine = DCFEEngine(DEMO_EMOTIONS, kappa=1.0)

        # T_ij extractor: shared instance to avoid loading the model twice
        self.tij_analyzer = ModulatedObjectAnalyzer(model_name)

        print("Ready!")

    def _extract_tij(self, scenario_text: str, object_words: list) -> tuple:
        """
        Extract modulated_pull from scenario text and derive independent
        Crime and Justice cognitive masses.

        modulated_pull is higher when the object is aggravating (innocent victim)
        and lower when mitigating (guilty murderer). Inverting for justice_pull
        ensures the two masses push in opposite directions:
          - aggravating object -> tij_crime dominant -> Crime attractor
          - mitigating object  -> tij_justice dominant -> Justice attractor
        """
        df = self.tij_analyzer.extract_with_modulation(
            [("target", scenario_text, object_words)]
        )
        modulated_pull = float(df["modulated_pull"].mean())
        crime_pull   = modulated_pull
        # Normalize justice_pull to the same scale as crime_pull.
        # Raw inverse (1/modulated_pull) is ~48x larger and overwhelms other fields.
        # We invert and rescale so that justice_pull is in the same range as crime_pull,
        # while still being inversely proportional (mitigating -> larger justice_pull).
        raw_justice  = 1.0 / (modulated_pull + 1e-8)
        justice_pull = crime_pull * (raw_justice / (raw_justice + crime_pull))
        return crime_pull, justice_pull

    def generate_with_DCFE(self, prefix, fields, scenario_text, object_words,
                           max_new_tokens=20, bias_strength=5.0):
        """
        1. Extract T_ij from scenario_text -> tij_crime and tij_justice as
           independent cognitive masses
        2. DCFEEngine (compute_weights + select) -> determine dominant emotion
        3. DCFELogitsBiasProcessor -> boost token scores for that emotion
        4. Continuation mode: prefix tokenized directly, TinyLlama continues
           the sentence from the prefix end. Logit bias acts only on new tokens.
        """
        # [1] Extract independent Crime/Justice T_ij masses from scenario text
        tij_crime, tij_justice = self._extract_tij(scenario_text, object_words)
        print(f"\n  [T_ij] crime_pull={tij_crime:.6f}  justice_pull={tij_justice:.6f}")

        merged_fields = fields.copy()
        merged_fields["tij_crime"]   = tij_crime
        merged_fields["tij_justice"] = tij_justice

        # [2] DCFE engine: interpretive bifurcation
        import DCFE_Engine
        original_fields = DCFE_Engine.FIELDS
        DCFE_Engine.FIELDS = DEMO_FIELDS
        try:
            engine_weights = self.DCFE_engine.compute_weights(merged_fields, T=1.0)
            dominant_engine_emotion, info = self.DCFE_engine.select_emotion(merged_fields, T=1.0)
        finally:
            DCFE_Engine.FIELDS = original_fields

        dominant_emotion = ENGINE_TO_BIAS.get(dominant_engine_emotion, dominant_engine_emotion.lower())
        weights = {
            ENGINE_TO_BIAS.get(k, k.lower()): v
            for k, v in engine_weights.items()
            if ENGINE_TO_BIAS.get(k, k.lower()) in BIAS_EMOTIONS
        }

        print(f"\n{'='*60}")
        print(f"Fields  : {merged_fields}")
        print(f"Weights : { {k: round(v,2) for k,v in sorted(weights.items(), key=lambda x:-x[1])} }")
        print(f"Dominant: {dominant_emotion} (engine={dominant_engine_emotion}, |F|={info['forces'][dominant_engine_emotion]['magnitude']:.4f})")
        print(f"{'='*60}")

        # [3] Logit bias from dominant attractor
        processor = DCFELogitsBiasProcessor(self.tokenizer, dominant_emotion, bias_strength)

        # [4] Continuation mode: tokenize prefix directly (no chat template)
        # TinyLlama continues the sentence from the prefix end.
        # Logit bias acts only on the generated tokens, not the prefix.
        inputs = self.tokenizer(prefix, return_tensors="pt").to(self.model.device)
        prompt_len = inputs.input_ids.shape[1]

        torch.manual_seed(SEED)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.85,
                repetition_penalty=1.5,
                logits_processor=LogitsProcessorList([processor]),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (after the prefix)
        # Truncate at the first sentence-ending punctuation to prevent
        # runaway generation after the sentence is complete.
        new_tokens = output_ids[0][prompt_len:]
        completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        for punct in ['.', '!', '?']:
            if punct in completion:
                completion = completion[:completion.index(punct) + 1]
                break
        answer = f"{prefix} {completion}"
        print(f"Answer: {answer}\n")
        return answer


# ============================================================
# STEP 4: Evaluation & Comparison
# ============================================================

def run_experiment():
    set_seed()
    DCFE_llm = DCFE_LLM()

    # prefix is the sentence start; TinyLlama continues from here.
    prefix = "The father's acts are"

    # scenario_text is used ONLY for T_ij extraction.
    # Must be a short action description so the 'killing' token can be found.
    # Do NOT pass the full prompt here.
    scenario_text = "Father killing the murderer of his family"
    object_words  = ["murderer", "killer", "guilty", "justice"]

    print("\n" + "="*60)
    print("SCENARIO P6: Justice Dominant (social_norm=0.0)")
    print("="*60)
    r1 = DCFE_llm.generate_with_DCFE(
        prefix=prefix,
        fields={"cortisol": 1.5, "dopamine": 0.65, "social_norm": 0.0},
        scenario_text=scenario_text,
        object_words=object_words,
        max_new_tokens=20,
        bias_strength=5.0,
    )

    print("\n" + "="*60)
    print("SCENARIO P7: Social Norm Dominant (social_norm=100.0)")
    print("="*60)
    r2 = DCFE_llm.generate_with_DCFE(
        prefix=prefix,
        fields={"cortisol": 1.5, "dopamine": 0.65, "social_norm": 100.0},
        scenario_text=scenario_text,
        object_words=object_words,
        max_new_tokens=20,
        bias_strength=5.0,
    )

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Prefix: {prefix}\n")
    print(f"[P6 Justice]     {r1}")
    print(f"[P7 Social Norm] {r2}")


if __name__ == "__main__":
    run_experiment()
