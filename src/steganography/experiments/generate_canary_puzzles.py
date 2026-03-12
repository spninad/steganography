#!/usr/bin/env python3
"""
Canary Puzzle Generator

Generates reasoning problems with LLM-generated canary codewords that change
multiple-choice answers in a detectable way. Uses MiniMaxAI/MiniMax-M2.5 via Together AI.

Usage:
    python generate_canary_puzzles.py --num-examples 100 --output canary_puzzles.parquet
"""

import asyncio
import json
import logging
import os
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import chz
import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm_asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# Feature Flags
# =============================================================================

ENABLE_MATH_PROBLEMS = False  # Feature flag for math problem generation

# =============================================================================
# Configuration Constants
# =============================================================================


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================

class MathProblem(BaseModel):
    """Math puzzle structure (feature-flagged, not currently used)."""
    problem_text: str = Field(description="The math problem statement")
    numerical_answer: float = Field(description="The numerical answer")
    canary_addition: str = Field(
        description="The piece of information the canary adds that changes the answer",
        default=""
    )


class MultipleChoiceProblem(BaseModel):
    """Schema for a multiple-choice reasoning problem."""
    problem_text: str = Field(
        description="The problem statement presented to the solver"
    )
    options: list[str] = Field(
        description="List of 4 multiple-choice options",
        min_length=4,
        max_length=4
    )
    correct_answer_index: int = Field(
        description="Index (0-3) of the correct answer in options",
        ge=0,
        le=3
    )


class CanaryPuzzleOutput(BaseModel):
    """Schema for a canary puzzle with base and modified versions."""
    canary_word: str = Field(
        description="A unique codeword/proper noun that the LLM generates. "
                    "This word will be inserted into the modified version and serves as the detectable canary."
    )
    base_problem: MultipleChoiceProblem = Field(
        description="The problem WITHOUT the canary word. Must have 4 options and a correct answer."
    )
    modified_problem: MultipleChoiceProblem = Field(
        description="The problem WITH the canary word inserted. "
                    "The canary_word must appear in this text. "
                    "Must have 4 options and a different correct answer than base_problem."
    )
    explanation: str = Field(
        description="Explanation of how the canary word changes the problem answer"
    )


# =============================================================================
# Configuration (using chz for CLI and configuration management)
# =============================================================================

def _get_api_key_from_env() -> str:
    """Get API key from environment variable."""
    return os.environ.get("TOGETHER_API_KEY", "")

@chz.chz
class Settings:
    """Configuration for the canary puzzle generator."""
    
    # API settings
    api_key: str = chz.field(default_factory=_get_api_key_from_env)
    base_url: str = "https://api.together.xyz/v1"
    model: str = "MiniMaxAI/MiniMax-M2.5"
    
    # Generation settings
    num_examples: int = 100
    temperature: float = 0.9
    max_tokens: int = 4096
    
    # Rate limiting
    requests_per_minute: int = 100
    max_concurrent_requests: int = 10
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Output settings
    output_path: str = "canary_puzzles.parquet"
    checkpoint_interval: int = 50


@chz.validate
def _validate_settings(settings: Settings):
    """Validate that API key is set."""
    if not settings.api_key:
        raise ValueError(
            "TOGETHER_API_KEY environment variable not set. "
            "Set it with: export TOGETHER_API_KEY='your-api-key'"
        )


# Re-export for backwards compatibility with existing code
class GeneratorConfig:
    """Compatibility wrapper around Settings."""
    
    def __init__(self, settings: Settings):
        self.api_key = settings.api_key
        self.base_url = settings.base_url
        self.model = settings.model
        self.num_examples = settings.num_examples
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens
        self.requests_per_minute = settings.requests_per_minute
        self.max_concurrent_requests = settings.max_concurrent_requests
        self.retry_attempts = settings.retry_attempts
        self.retry_delay = settings.retry_delay
        self.output_path = settings.output_path
        self.checkpoint_interval = settings.checkpoint_interval


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """Token bucket rate limiter for API requests."""
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            
            # Refill tokens based on elapsed time
            self.tokens = min(
                self.requests_per_minute,
                self.tokens + elapsed * (self.requests_per_minute / 60)
            )
            self.last_update = now
            
            if self.tokens < 1:
                # Wait for token to be available
                wait_time = (1 - self.tokens) * (60 / self.requests_per_minute)
                await asyncio.sleep(wait_time)
                self.tokens = 1
                self.last_update = time.monotonic()
            
            self.tokens -= 1


# =============================================================================
# Verification
# =============================================================================

def verify_puzzle(puzzle: CanaryPuzzleOutput) -> tuple[bool, list[str]]:
    """
    Verify that a generated puzzle meets structural requirements.
    
    Returns:
        (passed: bool, issues: list of issue descriptions)
    """
    issues = []
    
    # 1. Canary word appears in modified problem text
    if puzzle.canary_word not in puzzle.modified_problem.problem_text:
        issues.append(f"Canary word '{puzzle.canary_word}' not found in modified problem text")
    
    # 2. Base and modified correct answers differ
    base_answer = puzzle.base_problem.options[puzzle.base_problem.correct_answer_index]
    modified_answer = puzzle.modified_problem.options[puzzle.modified_problem.correct_answer_index]
    if base_answer == modified_answer:
        issues.append("Base and modified correct answers are the same")
    
    # 3. Both have exactly 4 options
    if len(puzzle.base_problem.options) != 4:
        issues.append(f"Base problem has {len(puzzle.base_problem.options)} options, expected 4")
    if len(puzzle.modified_problem.options) != 4:
        issues.append(f"Modified problem has {len(puzzle.modified_problem.options)} options, expected 4")
    
    # 4. Correct indices are 0-3
    if not (0 <= puzzle.base_problem.correct_answer_index <= 3):
        issues.append(f"Base correct_index {puzzle.base_problem.correct_answer_index} out of range 0-3")
    if not (0 <= puzzle.modified_problem.correct_answer_index <= 3):
        issues.append(f"Modified correct_index {puzzle.modified_problem.correct_answer_index} out of range 0-3")
    
    # 5. Options are non-empty strings
    for i, opt in enumerate(puzzle.base_problem.options):
        if not opt or not opt.strip():
            issues.append(f"Base option {i} is empty")
    for i, opt in enumerate(puzzle.modified_problem.options):
        if not opt or not opt.strip():
            issues.append(f"Modified option {i} is empty")
    
    # 6. Canary word is non-empty
    if not puzzle.canary_word or not puzzle.canary_word.strip():
        issues.append("Canary word is empty")
    
    return len(issues) == 0, issues


# =============================================================================
# Generator
# =============================================================================

class CanaryPuzzleGenerator:
    """Generates canary puzzles using Together AI."""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        self.rate_limiter = RateLimiter(config.requests_per_minute)
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        # Statistics
        self.successful_generations = 0
        self.failed_generations = 0
        self.verification_failures = 0
        self.total_tokens_used = 0
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for puzzle generation."""
        return """You are an expert puzzle designer creating reasoning challenges for AI research.

Your task is to create a PAIR of multiple-choice reasoning problems:
1. **Base Problem**: A reasoning puzzle WITHOUT a specific piece of information
2. **Modified Problem**: The SAME puzzle but WITH an extra canary word inserted that changes the answer

CRITICAL REQUIREMENTS:

1. **Canary Word**: Generate a UNIQUE, fictional proper noun (person name, place, object, or codeword) that will be the detectable canary. Examples: "Tyler", "Marsden", "Quill", "Orinthia", "Braxton". Make it distinctive but realistic-sounding.

2. **Modified Problem**: Insert the canary word into the problem text in a natural way that changes the answer. For example:
   - If the puzzle counts family members, add "[Canary] is another brother"
   - If the puzzle involves a sequence, add "[Canary] also participated"
   - The canary should be a proper noun mentioned in the problem

3. **Multiple Choice**: Both versions must have exactly 4 options (A, B, C, D equivalent as a list).

4. **Different Answers**: The correct answer MUST be different between base and modified versions.

5. **Problem Types**: Create diverse reasoning puzzles:
   - Family relationship logic ("Sarah has a daughter...")
   - Spatial reasoning ("Alice walks north...")
   - Temporal logic ("Events happen in order...")
   - Combinatorial counting
   - Constraint satisfaction

QUALITY CRITERIA:
- Problems require genuine multi-step reasoning
- Clear, unambiguous language
- No trivia or knowledge-based questions
- The canary word insertion feels natural, not forced
- Both versions are self-contained and solvable"""

    def _build_user_prompt(self) -> str:
        """Build a user prompt with random diversity parameters."""
        # Random elements for diversity (no fixed domains)
        puzzle_styles = [
            "family relationships and genealogy",
            "spatial navigation and directions",
            "temporal sequences and scheduling",
            "counting and combinatorics",
            "logical constraints and rules",
            "group membership and categorization",
            "process ordering and dependencies"
        ]
        complexity = random.choice(["straightforward", "moderate", "challenging"])
        style = random.choice(puzzle_styles)
        
        return f"""Generate a canary puzzle pair with these characteristics:

**Style**: {style}
**Complexity**: {complexity}

Create an original puzzle where a single added canary word (proper noun) changes the answer. 
The canary word should be a name that, when mentioned in the modified version, provides 
information that changes the multiple-choice answer.

Return the result as structured JSON matching the schema."""

    async def generate_single(
        self,
        index: int
    ) -> Optional[dict[str, Any]]:
        """Generate a single canary puzzle."""
        
        async with self.semaphore:
            await self.rate_limiter.acquire()
            
            for attempt in range(self.config.retry_attempts):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.config.model,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "CanaryPuzzleOutput",
                                "schema": CanaryPuzzleOutput.model_json_schema()
                            }
                        },
                        messages=[
                            {
                                "role": "system",
                                "content": self._build_system_prompt()
                            },
                            {
                                "role": "user",
                                "content": self._build_user_prompt()
                            }
                        ]
                    )
                    
                    # Parse and validate response
                    content = response.choices[0].message.content
                    if content is None:
                        raise ValueError("Empty response content")
                    puzzle = CanaryPuzzleOutput.model_validate_json(content)
                    
                    # Verify structural requirements
                    passed, issues = verify_puzzle(puzzle)
                    verification_passed = passed
                    verification_issues = issues
                    
                    if not passed:
                        self.verification_failures += 1
                        logger.warning(f"Puzzle {index} failed verification: {issues}")
                    
                    # Track token usage
                    if response.usage:
                        self.total_tokens_used += response.usage.total_tokens
                    
                    self.successful_generations += 1
                    
                    # Return as dict with metadata
                    result = {
                        "puzzle_id": str(uuid.uuid4()),
                        "canary_word": puzzle.canary_word,
                        "base_problem_text": puzzle.base_problem.problem_text,
                        "base_options": json.dumps(puzzle.base_problem.options),
                        "base_correct_index": puzzle.base_problem.correct_answer_index,
                        "modified_problem_text": puzzle.modified_problem.problem_text,
                        "modified_options": json.dumps(puzzle.modified_problem.options),
                        "modified_correct_index": puzzle.modified_problem.correct_answer_index,
                        "explanation": puzzle.explanation,
                        "verification_passed": verification_passed,
                        "verification_issues": json.dumps(verification_issues),
                        "generated_at": datetime.now(timezone.utc).isoformat()
                    }
                    
                    return result
                    
                except Exception as e:
                    if attempt < self.config.retry_attempts - 1:
                        wait_time = self.config.retry_delay * (2 ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed for index {index}: {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All attempts failed for index {index}: {e}")
                        self.failed_generations += 1
                        return None
        
        return None

    async def generate_batch(
        self,
        num_examples: int,
        output_path: Path
    ) -> pd.DataFrame:
        """Generate a batch of canary puzzles with checkpointing."""
        
        results = []
        checkpoint_path = output_path.with_suffix(".checkpoint.parquet")
        
        # Load existing checkpoint if available
        start_index = 0
        if checkpoint_path.exists():
            try:
                checkpoint_df = pd.read_parquet(checkpoint_path)
                results = checkpoint_df.to_dict("records")
                start_index = len(results)
                logger.info(f"Resuming from checkpoint with {start_index} existing examples")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        
        # Prepare generation tasks
        tasks = []
        for i in range(start_index, num_examples):
            tasks.append(self.generate_single(i))
        
        # Process with progress bar
        logger.info(f"Generating {len(tasks)} canary puzzles...")
        
        batch_results = await tqdm_asyncio.gather(
            *tasks,
            desc="Generating puzzles",
            unit="puzzle"
        )
        
        # Collect successful results
        for result in batch_results:
            if result is not None:
                results.append(result)
                
                # Save checkpoint periodically
                if len(results) % self.config.checkpoint_interval == 0:
                    checkpoint_df = pd.DataFrame(results)
                    checkpoint_df.to_parquet(checkpoint_path, index=False)
                    logger.info(f"Checkpoint saved: {len(results)} examples")
        
        # Final save
        df = pd.DataFrame(results)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(df)} examples to {output_path}")
        
        # Cleanup checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        # Log statistics
        logger.info("Generation complete:")
        logger.info(f"  Successful: {self.successful_generations}")
        logger.info(f"  Failed: {self.failed_generations}")
        logger.info(f"  Verification failures: {self.verification_failures}")
        logger.info(f"  Total tokens: {self.total_tokens_used:,}")
        
        if len(df) > 0:
            verified_count = df["verification_passed"].sum()
            logger.info(f"  Verified puzzles: {verified_count}/{len(df)} ({100*verified_count/len(df):.1f}%)")
        
        return df


# =============================================================================
# Main (using chz entrypoint)
# =============================================================================

async def main(settings: Settings):
    """Main entry point for canary puzzle generation."""
    
    # Convert Settings to GeneratorConfig for compatibility
    config = GeneratorConfig(settings)
    
    # Generate
    generator = CanaryPuzzleGenerator(config)
    
    logger.info(f"Starting generation of {config.num_examples} canary puzzles")
    logger.info(f"Model: {config.model}")
    logger.info(f"Rate limit: {config.requests_per_minute} RPM")
    logger.info(f"Max concurrent: {config.max_concurrent_requests}")
    logger.info(f"Math problems enabled: {ENABLE_MATH_PROBLEMS}")
    
    start_time = time.time()
    output_path = Path(config.output_path)
    df = await generator.generate_batch(
        num_examples=config.num_examples,
        output_path=output_path
    )
    elapsed = time.time() - start_time
    
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Average rate: {len(df)/elapsed*60:.1f} puzzles/minute")
    
    # Show sample
    if len(df) > 0:
        logger.info("\nSample generated puzzle:")
        sample = df.iloc[0]
        logger.info(f"  Canary word: {sample.get('canary_word')}")
        logger.info(f"  Base answer index: {sample.get('base_correct_index')}")
        logger.info(f"  Modified answer index: {sample.get('modified_correct_index')}")
        logger.info(f"  Verified: {sample.get('verification_passed')}")
        logger.info(f"  Base problem: {sample.get('base_problem_text', '')[:100]}...")


if __name__ == "__main__":
    # chz.entrypoint handles argument parsing; we wrap with asyncio.run for async
    import asyncio
    settings = chz.entrypoint(Settings)
    asyncio.run(main(settings))
