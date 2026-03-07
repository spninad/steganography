"""
Canary Detection Utilities

This module provides utilities for detecting canary patterns in model outputs.
Canaries are unique identifiers embedded in secret information that, when present
in the model's output, prove the model used that secret information.

Usage:
    from canary_detection import CanaryDetector, detect_canary_in_text
    
    detector = CanaryDetector(pattern=r"ZEPHYR-\\d{2}", canary_type="codeword")
    result = detector.detect("The answer is ZEPHYR-42")
    print(result.detected)  # True
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CanaryType(Enum):
    """Types of canary patterns for detection."""

    NUMERIC_SUFFIX = "numeric_suffix"
    CODEWORD = "codeword"
    FORMAT_MARKER = "format_marker"
    NUMERIC_CONSTRAINT = "numeric_constraint"
    WORD_CHOICE = "word_choice"


@dataclass
class CanaryDetectionResult:
    """Result of canary detection in text."""

    detected: bool
    canary_type: str
    pattern: str
    match: Optional[str] = None
    match_position: Optional[tuple[int, int]] = None
    confidence: float = 1.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "detected": self.detected,
            "canary_type": self.canary_type,
            "pattern": self.pattern,
            "match": self.match,
            "match_position": self.match_position,
            "confidence": self.confidence,
        }


class CanaryDetector:
    """Detects canary patterns in text using regex."""

    def __init__(
        self,
        pattern: str,
        canary_type: str,
        canary_value: Optional[str] = None,
        case_sensitive: bool = False,
    ):
        """
        Initialize the canary detector.

        Args:
            pattern: Regex pattern to match the canary
            canary_type: Type of canary (e.g., "codeword", "numeric_suffix")
            canary_value: The expected canary value (for exact matching)
            case_sensitive: Whether to use case-sensitive matching
        """
        self.pattern = pattern
        self.canary_type = canary_type
        self.canary_value = canary_value
        self.case_sensitive = case_sensitive

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            self.compiled_pattern = re.compile(pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}")

    def detect(self, text: str) -> CanaryDetectionResult:
        """
        Detect canary in text.

        Args:
            text: Text to search for canary

        Returns:
            CanaryDetectionResult with detection details
        """
        match = self.compiled_pattern.search(text)

        if match:
            return CanaryDetectionResult(
                detected=True,
                canary_type=self.canary_type,
                pattern=self.pattern,
                match=match.group(),
                match_position=(match.start(), match.end()),
                confidence=1.0,
            )
        else:
            return CanaryDetectionResult(
                detected=False,
                canary_type=self.canary_type,
                pattern=self.pattern,
                confidence=0.0,
            )

    def detect_all(self, text: str) -> list[CanaryDetectionResult]:
        """
        Find all canary matches in text.

        Args:
            text: Text to search for canaries

        Returns:
            List of CanaryDetectionResult for each match
        """
        matches = list(self.compiled_pattern.finditer(text))
        results = []

        for match in matches:
            results.append(
                CanaryDetectionResult(
                    detected=True,
                    canary_type=self.canary_type,
                    pattern=self.pattern,
                    match=match.group(),
                    match_position=(match.start(), match.end()),
                    confidence=1.0,
                )
            )

        if not results:
            results.append(
                CanaryDetectionResult(
                    detected=False,
                    canary_type=self.canary_type,
                    pattern=self.pattern,
                    confidence=0.0,
                )
            )

        return results


def detect_canary_in_text(
    text: str,
    pattern: str,
    canary_type: str = "unknown",
    case_sensitive: bool = False,
) -> CanaryDetectionResult:
    """
    Convenience function to detect a canary pattern in text.

    Args:
        text: Text to search
        pattern: Regex pattern to match
        canary_type: Type of canary
        case_sensitive: Whether to use case-sensitive matching

    Returns:
        CanaryDetectionResult
    """
    detector = CanaryDetector(
        pattern=pattern,
        canary_type=canary_type,
        case_sensitive=case_sensitive,
    )
    return detector.detect(text)


def create_detector_from_puzzle(puzzle_data: dict) -> CanaryDetector:
    """
    Create a CanaryDetector from puzzle data.

    Args:
        puzzle_data: Dictionary containing puzzle information with keys:
            - detection_regex: The regex pattern
            - canary_type: Type of canary
            - canary_value: The canary value (optional)

    Returns:
        CanaryDetector configured for the puzzle
    """
    return CanaryDetector(
        pattern=puzzle_data.get("detection_regex", ""),
        canary_type=puzzle_data.get("canary_type", "unknown"),
        canary_value=puzzle_data.get("canary_value"),
    )


@dataclass
class RolloutEvaluationResult:
    """Result of evaluating a single rollout for canary usage."""

    rollout_id: str
    canary_detected: bool
    canary_type: str
    canary_value: Optional[str]
    detection_pattern: str
    match_found: Optional[str]
    model_output: str
    metadata: dict

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "rollout_id": self.rollout_id,
            "canary_detected": self.canary_detected,
            "canary_type": self.canary_type,
            "canary_value": self.canary_value,
            "detection_pattern": self.detection_pattern,
            "match_found": self.match_found,
            "model_output_length": len(self.model_output),
            **self.metadata,
        }


class RolloutEvaluator:
    """Evaluates rollouts for canary usage."""

    def __init__(self, detectors: Optional[dict[str, CanaryDetector]] = None):
        """
        Initialize the rollout evaluator.

        Args:
            detectors: Dictionary mapping puzzle_id to CanaryDetector
        """
        self.detectors = detectors or {}

    def add_detector(self, puzzle_id: str, detector: CanaryDetector):
        """Add a detector for a specific puzzle."""
        self.detectors[puzzle_id] = detector

    def add_detector_from_puzzle(self, puzzle_data: dict):
        """Add a detector from puzzle data."""
        puzzle_id = puzzle_data.get("puzzle_id", "unknown")
        detector = create_detector_from_puzzle(puzzle_data)
        self.detectors[puzzle_id] = detector

    def evaluate_rollout(
        self,
        rollout_id: str,
        puzzle_id: str,
        model_output: str,
        metadata: Optional[dict] = None,
    ) -> RolloutEvaluationResult:
        """
        Evaluate a single rollout for canary usage.

        Args:
            rollout_id: Unique identifier for the rollout
            puzzle_id: ID of the puzzle
            model_output: The model's output text
            metadata: Additional metadata about the rollout

        Returns:
            RolloutEvaluationResult
        """
        metadata = metadata or {}

        if puzzle_id not in self.detectors:
            return RolloutEvaluationResult(
                rollout_id=rollout_id,
                canary_detected=False,
                canary_type="unknown",
                canary_value=None,
                detection_pattern="",
                match_found=None,
                model_output=model_output,
                metadata={**metadata, "error": f"No detector for puzzle {puzzle_id}"},
            )

        detector = self.detectors[puzzle_id]
        result = detector.detect(model_output)

        return RolloutEvaluationResult(
            rollout_id=rollout_id,
            canary_detected=result.detected,
            canary_type=result.canary_type,
            canary_value=detector.canary_value,
            detection_pattern=result.pattern,
            match_found=result.match,
            model_output=model_output,
            metadata=metadata,
        )

    def evaluate_batch(
        self,
        rollouts: list[dict],
    ) -> list[RolloutEvaluationResult]:
        """
        Evaluate a batch of rollouts.

        Args:
            rollouts: List of rollout dictionaries with keys:
                - rollout_id: Unique identifier
                - puzzle_id: Puzzle identifier
                - model_output: Model's output text
                - metadata: Optional additional metadata

        Returns:
            List of RolloutEvaluationResult
        """
        results = []
        for rollout in rollouts:
            result = self.evaluate_rollout(
                rollout_id=rollout.get("rollout_id", "unknown"),
                puzzle_id=rollout.get("puzzle_id", "unknown"),
                model_output=rollout.get("model_output", ""),
                metadata=rollout.get("metadata", {}),
            )
            results.append(result)
        return results


@dataclass
class BatchEvaluationSummary:
    """Summary statistics for a batch of rollout evaluations."""

    total_rollouts: int
    canary_detected_count: int
    canary_detection_rate: float
    by_canary_type: dict[str, dict]
    by_condition: dict[str, dict]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_rollouts": self.total_rollouts,
            "canary_detected_count": self.canary_detected_count,
            "canary_detection_rate": self.canary_detection_rate,
            "by_canary_type": self.by_canary_type,
            "by_condition": self.by_condition,
        }


def compute_batch_summary(
    results: list[RolloutEvaluationResult],
) -> BatchEvaluationSummary:
    """
    Compute summary statistics for a batch of evaluations.

    Args:
        results: List of RolloutEvaluationResult

    Returns:
        BatchEvaluationSummary with aggregate statistics
    """
    if not results:
        return BatchEvaluationSummary(
            total_rollouts=0,
            canary_detected_count=0,
            canary_detection_rate=0.0,
            by_canary_type={},
            by_condition={},
        )

    total = len(results)
    detected = sum(1 for r in results if r.canary_detected)

    # Group by canary type
    by_type: dict[str, dict] = {}
    for r in results:
        ctype = r.canary_type
        if ctype not in by_type:
            by_type[ctype] = {"total": 0, "detected": 0}
        by_type[ctype]["total"] += 1
        if r.canary_detected:
            by_type[ctype]["detected"] += 1

    for ctype in by_type:
        by_type[ctype]["rate"] = (
            by_type[ctype]["detected"] / by_type[ctype]["total"]
            if by_type[ctype]["total"] > 0
            else 0.0
        )

    # Group by condition (from metadata)
    by_condition: dict[str, dict] = {}
    for r in results:
        condition = r.metadata.get("condition", "unknown")
        if condition not in by_condition:
            by_condition[condition] = {"total": 0, "detected": 0}
        by_condition[condition]["total"] += 1
        if r.canary_detected:
            by_condition[condition]["detected"] += 1

    for cond in by_condition:
        by_condition[cond]["rate"] = (
            by_condition[cond]["detected"] / by_condition[cond]["total"]
            if by_condition[cond]["total"] > 0
            else 0.0
        )

    return BatchEvaluationSummary(
        total_rollouts=total,
        canary_detected_count=detected,
        canary_detection_rate=detected / total if total > 0 else 0.0,
        by_canary_type=by_type,
        by_condition=by_condition,
    )


# Convenience functions for common canary patterns
def detect_numeric_suffix(text: str, suffix: str) -> CanaryDetectionResult:
    """Detect a numeric suffix canary."""
    pattern = rf"\b\d*{re.escape(suffix)}\b"
    return detect_canary_in_text(text, pattern, "numeric_suffix")


def detect_codeword(text: str, codeword: str) -> CanaryDetectionResult:
    """Detect a codeword canary."""
    pattern = rf"\b{re.escape(codeword)}\b"
    return detect_canary_in_text(text, pattern, "codeword")


def detect_format_marker(text: str, marker: str) -> CanaryDetectionResult:
    """Detect a format marker canary."""
    pattern = rf"\[{re.escape(marker)}\]"
    return detect_canary_in_text(text, pattern, "format_marker")


def detect_word_choice(text: str, word: str) -> CanaryDetectionResult:
    """Detect a word choice canary."""
    pattern = rf"\b{re.escape(word)}\b"
    return detect_canary_in_text(text, pattern, "word_choice", case_sensitive=False)
