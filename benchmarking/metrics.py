"""
Evaluation metrics for benchmarking.

This module provides various metrics for comparing model predictions
to ground truth answers.
"""

import re
from typing import Optional


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison.
    
    - Lowercase
    - Strip whitespace
    - Remove extra spaces
    - Remove common punctuation
    
    Args:
        text: The text to normalize.
        
    Returns:
        Normalized text.
    """
    if not text:
        return ""
    
    # Lowercase and strip
    text = text.lower().strip()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common punctuation at start/end
    text = text.strip('.,!?;:')
    
    return text


def exact_match(prediction: str, target: str, normalize: bool = True) -> bool:
    """
    Check if prediction exactly matches target.
    
    Args:
        prediction: The predicted answer.
        target: The ground truth answer.
        normalize: Whether to normalize both texts before comparison.
        
    Returns:
        True if exact match, False otherwise.
    """
    if normalize:
        prediction = normalize_answer(prediction)
        target = normalize_answer(target)
    
    return prediction == target


def contains_match(
    prediction: str, 
    target: str, 
    normalize: bool = True,
    bidirectional: bool = False
) -> bool:
    """
    Check if target is contained within prediction (or vice versa if bidirectional).
    
    Args:
        prediction: The predicted answer.
        target: The ground truth answer.
        normalize: Whether to normalize both texts before comparison.
        bidirectional: If True, also check if prediction is in target.
        
    Returns:
        True if match found, False otherwise.
    """
    if normalize:
        pred_norm = normalize_answer(prediction)
        target_norm = normalize_answer(target)
    else:
        pred_norm = prediction
        target_norm = target
    
    if target_norm in pred_norm:
        return True
    
    if bidirectional and pred_norm in target_norm:
        return True
    
    return False


def extract_choice(prediction: str, choices: list[str]) -> Optional[str]:
    """
    Extract which choice was selected from the prediction text.
    
    This function tries multiple strategies to find which of the given choices
    appears in the prediction.
    
    Strategies:
    1. Exact match (normalized)
    2. Contains match (normalized)
    3. Look for choice in quotes
    4. Letter label match (A, B, C, ...) mapped to choice by index
    
    Args:
        prediction: The model's output text.
        choices: List of valid choices.
        
    Returns:
        The matched choice, or None if no match found.
    """
    if not prediction or not choices:
        return None
    
    pred_norm = normalize_answer(prediction)
    
    # Strategy 1: Exact match
    for choice in choices:
        if exact_match(prediction, choice):
            return choice
    
    # Strategy 2: Contains match
    for choice in choices:
        choice_norm = normalize_answer(choice)
        if choice_norm in pred_norm:
            return choice
    
    # Strategy 3: Look for choice in quotes
    for choice in choices:
        if f'"{choice}"' in prediction or f"'{choice}'" in prediction:
            return choice
    
    # Strategy 4: Letter label match
    # Look for standalone letter labels like "A", "A.", "(A)", "Answer: A"
    # and map them to the corresponding choice by index.
    max_letter = chr(64 + len(choices))  # A=65, B=66, ...
    letter_pattern = rf'(?:^|[\s:,-])\(?([A-{max_letter}])\)?\.?(?:\s|$)'
    letter_match = re.search(letter_pattern, prediction.strip(), re.IGNORECASE)
    if letter_match:
        idx = ord(letter_match.group(1).upper()) - ord('A')
        if 0 <= idx < len(choices):
            return choices[idx]
    
    return None


def choice_match(
    prediction: str,
    target: str,
    choices: list[str],
    allow_contains: bool = True
) -> dict:
    """
    Evaluate if the prediction matches the target for multiple choice questions.
    
    Args:
        prediction: The predicted answer.
        target: The ground truth answer.
        choices: List of valid choices.
        allow_contains: Whether to allow contains matching.
        
    Returns:
        Dictionary with:
        - correct: Whether prediction matches target
        - extracted_choice: The choice extracted from prediction (if any)
        - match_method: How the match was determined
    """
    result = {
        "correct": False,
        "extracted_choice": None,
        "match_method": None,
    }
    
    # Extract the choice from prediction
    extracted = extract_choice(prediction, choices)
    result["extracted_choice"] = extracted
    
    if extracted is None:
        result["match_method"] = "no_extraction"
        return result
    
    # Check if extracted matches target
    if exact_match(extracted, target):
        result["correct"] = True
        result["match_method"] = "exact"
    elif allow_contains and contains_match(extracted, target):
        result["correct"] = True
        result["match_method"] = "contains"
    else:
        result["match_method"] = "mismatch"
    
    return result


def f1_score(prediction: str, target: str) -> float:
    """
    Compute token-level F1 score between prediction and target.
    
    Args:
        prediction: The predicted answer.
        target: The ground truth answer.
        
    Returns:
        F1 score (0.0 to 1.0).
    """
    pred_tokens = set(normalize_answer(prediction).split())
    target_tokens = set(normalize_answer(target).split())
    
    if not pred_tokens and not target_tokens:
        return 1.0
    
    if not pred_tokens or not target_tokens:
        return 0.0
    
    common = pred_tokens & target_tokens
    
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(target_tokens) if target_tokens else 0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)
