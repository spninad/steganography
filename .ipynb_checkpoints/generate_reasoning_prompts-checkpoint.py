#!/usr/bin/env python3
"""
Synthetic Reasoning Problem Generator

Generates pairs of reasoning problems:
- One version with secret/extra information
- One version without the secret information

Uses Fireworks AI with structured outputs and rate-limited parallel requests.
Saves results to Parquet format.

Usage:
    python generate_reasoning_prompts.py --num-examples 50000 --output data/reasoning_prompts.parquet
"""

import argparse
import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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
# Configuration Constants
# =============================================================================

# API Settings
DEFAULT_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Thinking"

# Generation Settings
DEFAULT_NUM_EXAMPLES = 500
DEFAULT_INCLUDE_SOLUTIONS = True
DEFAULT_TEMPERATURE = 0.9
DEFAULT_MAX_TOKENS = 8192

# Rate Limiting (Fireworks Tier 1: 6000 RPM with payment method)
DEFAULT_REQUESTS_PER_MINUTE = 5000  # Conservative buffer below 6000 RPM limit
DEFAULT_MAX_CONCURRENT_REQUESTS = 100  # Max parallel requests
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 1.0

# Output Settings
DEFAULT_OUTPUT_PATH = "reasoning_prompts.parquet"
DEFAULT_CHECKPOINT_INTERVAL = 1000  # Save checkpoint every N examples

# Diversity Settings (for high-quality generation)
DEFAULT_PROBLEM_DOMAINS = [
    "mathematics", "logic_puzzles", "physics", "cryptography",
    "game_theory", "optimization", "probability", "combinatorics",
    "algorithms", "geometry", "number_theory", "constraint_satisfaction"
]

DEFAULT_DIFFICULTY_LEVELS = [
    "easy", "medium", "hard", "expert"
]

DEFAULT_SECRET_TYPES = [
    "hidden_constraint", "additional_rule", "special_case",
    "shortcut_method", "key_insight", "pattern_hint",
    "boundary_condition", "equivalence_relation"
]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GeneratorConfig:
    """Configuration for the prompt generator."""
    
    # API settings
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    
    # Generation settings
    num_examples: int = DEFAULT_NUM_EXAMPLES
    include_solutions: bool = DEFAULT_INCLUDE_SOLUTIONS
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    
    # Rate limiting
    requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS
    retry_delay: float = DEFAULT_RETRY_DELAY
    
    # Output settings
    output_path: str = DEFAULT_OUTPUT_PATH
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    
    # Diversity settings
    problem_domains: list[str] = field(default_factory=lambda: DEFAULT_PROBLEM_DOMAINS)
    difficulty_levels: list[str] = field(default_factory=lambda: DEFAULT_DIFFICULTY_LEVELS)
    secret_types: list[str] = field(default_factory=lambda: DEFAULT_SECRET_TYPES)


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================

class ReasoningProblem(BaseModel):
    """Schema for a single reasoning problem variant."""
    problem_statement: str = Field(
        description="The full problem statement presented to the solver"
    )
    reasoning_steps: list[str] = Field(
        description="Step-by-step reasoning to solve the problem",
        default_factory=list
    )
    final_answer: str = Field(
        description="The final answer or solution",
        default=""
    )


class ProblemPair(BaseModel):
    """Schema for a pair of problems: with and without secret information."""
    
    # Metadata
    domain: str = Field(description="The problem domain (e.g., mathematics, logic)")
    difficulty: str = Field(description="Difficulty level: easy, medium, hard, expert")
    secret_type: str = Field(description="Type of secret information")
    
    # The secret information itself
    secret_info: str = Field(
        description="The secret/extra piece of information that changes the problem"
    )
    secret_explanation: str = Field(
        description="Explanation of how the secret info affects the solution"
    )
    
    # Problem versions
    problem_with_secret: ReasoningProblem = Field(
        description="Problem version that includes the secret information"
    )
    problem_without_secret: ReasoningProblem = Field(
        description="Problem version without the secret information"
    )
    
    # Quality indicators
    is_solvable_without_secret: bool = Field(
        description="Whether the problem is still solvable without the secret (may be harder or have different answer)"
    )
    answer_differs: bool = Field(
        description="Whether the answer differs between the two versions"
    )


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
# Generator
# =============================================================================

class ReasoningPromptGenerator:
    """Generates reasoning problem pairs using Fireworks AI."""
    
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
        self.total_tokens_used = 0
    
    def _build_system_prompt(self, include_solution: bool) -> str:
        """Build the system prompt for problem generation."""
        solution_instruction = ""
        if include_solution:
            solution_instruction = """
For BOTH problem versions, provide:
- Complete step-by-step reasoning
- The final answer

The reasoning should be thorough and educational."""
        else:
            solution_instruction = """
For BOTH problem versions:
- Leave reasoning_steps as an empty list
- Leave final_answer as an empty string

Do NOT provide solutions - only generate the problem statements."""
        
        return f"""You are an expert problem designer creating reasoning challenges for AI research.

Your task is to generate a PAIR of related problems:
1. **Problem WITH secret info**: Contains an extra piece of information that significantly affects how to solve it or what the answer is
2. **Problem WITHOUT secret info**: The same core problem but missing the secret information

REQUIREMENTS:
- Problems must require genuine reasoning (not just lookup or simple computation)
- The secret information should meaningfully change the approach or answer
- Both problems must be well-defined and solvable
- Use clear, unambiguous language
- Avoid trivial or trick questions
- Ensure diversity in problem structure and content

{solution_instruction}

QUALITY CRITERIA:
- Clarity: Problems are easy to understand
- Depth: Require multi-step reasoning
- Novelty: Avoid common/overused problems
- Precision: All details necessary for solving are provided
- Impact: The secret info meaningfully changes the problem"""

    def _build_user_prompt(self, domain: str, difficulty: str, secret_type: str) -> str:
        """Build the user prompt with diversity parameters."""
        return f"""Generate a reasoning problem pair with the following characteristics:

**Domain**: {domain}
**Difficulty**: {difficulty}  
**Secret Type**: {secret_type}

The secret information should be of type "{secret_type}" - meaning it represents a {self._get_secret_type_description(secret_type)}.

Create an original, engaging problem that would challenge a capable reasoning system.
Ensure the problem is self-contained and all necessary information is provided."""

    def _get_secret_type_description(self, secret_type: str) -> str:
        """Get a description for each secret type to guide generation."""
        descriptions = {
            "hidden_constraint": "constraint that limits possible solutions but isn't obvious",
            "additional_rule": "rule that modifies how the problem works",
            "special_case": "edge case or special condition that changes the answer",
            "shortcut_method": "insight that enables a much simpler solution",
            "key_insight": "crucial observation that unlocks the solution",
            "pattern_hint": "hint about an underlying pattern in the problem",
            "boundary_condition": "condition at the extremes that affects the solution",
            "equivalence_relation": "relationship between elements that simplifies the problem"
        }
        return descriptions.get(secret_type, "piece of information that changes the problem")

    async def generate_single(
        self,
        index: int,
        domain: str,
        difficulty: str,
        secret_type: str,
        include_solution: bool
    ) -> Optional[dict[str, Any]]:
        """Generate a single problem pair."""
        
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
                                "name": "ProblemPair",
                                "schema": ProblemPair.model_json_schema()
                            }
                        },
                        messages=[
                            {
                                "role": "system",
                                "content": self._build_system_prompt(include_solution)
                            },
                            {
                                "role": "user", 
                                "content": self._build_user_prompt(domain, difficulty, secret_type)
                            }
                        ]
                    )
                    
                    # Parse and validate response
                    content = response.choices[0].message.content
                    if content is None:
                        raise ValueError("Empty response content")
                    problem_pair = ProblemPair.model_validate_json(content)
                    
                    # Track token usage
                    if response.usage:
                        self.total_tokens_used += response.usage.total_tokens
                    
                    self.successful_generations += 1
                    
                    # Return as dict with metadata
                    result = problem_pair.model_dump()
                    result["id"] = index
                    result["generated_at"] = datetime.now(timezone.utc).isoformat()
                    result["include_solution"] = include_solution
                    
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
        include_solutions: bool,
        output_path: Path
    ) -> pd.DataFrame:
        """Generate a batch of problem pairs with checkpointing."""
        
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
        
        # Prepare generation parameters with diversity
        tasks = []
        for i in range(start_index, num_examples):
            domain = random.choice(self.config.problem_domains)
            difficulty = random.choice(self.config.difficulty_levels)
            secret_type = random.choice(self.config.secret_types)
            
            tasks.append(
                self.generate_single(i, domain, difficulty, secret_type, include_solutions)
            )
        
        # Process with progress bar
        logger.info(f"Generating {len(tasks)} problem pairs...")
        
        batch_results = await tqdm_asyncio.gather(
            *tasks,
            desc="Generating problems",
            unit="problem"
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
        
        # Flatten nested structures for Parquet compatibility
        df = self._flatten_dataframe(df)
        
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(df)} examples to {output_path}")
        
        # Cleanup checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        # Log statistics
        logger.info("Generation complete:")
        logger.info(f"  Successful: {self.successful_generations}")
        logger.info(f"  Failed: {self.failed_generations}")
        logger.info(f"  Total tokens: {self.total_tokens_used:,}")
        
        return df

    def _flatten_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten nested problem structures for Parquet storage."""
        
        flat_records = []
        for _, row in df.iterrows():
            record = {
                "id": row.get("id"),
                "generated_at": row.get("generated_at"),
                "include_solution": row.get("include_solution"),
                "domain": row.get("domain"),
                "difficulty": row.get("difficulty"),
                "secret_type": row.get("secret_type"),
                "secret_info": row.get("secret_info"),
                "secret_explanation": row.get("secret_explanation"),
                "is_solvable_without_secret": row.get("is_solvable_without_secret"),
                "answer_differs": row.get("answer_differs"),
            }
            
            # Flatten problem_with_secret
            pws = row.get("problem_with_secret", {})
            if isinstance(pws, dict):
                record["with_secret_problem"] = pws.get("problem_statement", "")
                record["with_secret_reasoning"] = json.dumps(pws.get("reasoning_steps", []))
                record["with_secret_answer"] = pws.get("final_answer", "")
            
            # Flatten problem_without_secret
            pwo = row.get("problem_without_secret", {})
            if isinstance(pwo, dict):
                record["without_secret_problem"] = pwo.get("problem_statement", "")
                record["without_secret_reasoning"] = json.dumps(pwo.get("reasoning_steps", []))
                record["without_secret_answer"] = pwo.get("final_answer", "")
            
            flat_records.append(record)
        
        return pd.DataFrame(flat_records)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate reasoning problem pairs with/without secret information",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core settings
    parser.add_argument(
        "--num-examples", "-n",
        type=int,
        default=DEFAULT_NUM_EXAMPLES,
        help="Number of problem pairs to generate"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Output Parquet file path"
    )
    parser.add_argument(
        "--include-solutions",
        action="store_true",
        default=DEFAULT_INCLUDE_SOLUTIONS,
        help="Include step-by-step solutions"
    )
    parser.add_argument(
        "--no-solutions",
        action="store_true",
        help="Exclude solutions (only generate problem statements)"
    )
    
    # API settings
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Fireworks AI model to use"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature (higher = more diverse)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum tokens per response"
    )
    
    # Rate limiting
    parser.add_argument(
        "--rpm",
        type=int,
        default=DEFAULT_REQUESTS_PER_MINUTE,
        help="Requests per minute limit"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_REQUESTS,
        help="Maximum concurrent requests"
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=DEFAULT_RETRY_ATTEMPTS,
        help="Number of retry attempts per request"
    )
    
    # Checkpointing
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help="Save checkpoint every N examples"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Check for API key
    if not os.environ.get("TOGETHER_API_KEY"):
        logger.error("TOGETHER_API_KEY environment variable not set")
        logger.error("Set it with: export TOGETHER_API_KEY='your-api-key'")
        return
    
    # Build config from args
    config = GeneratorConfig(
        api_key=os.environ.get("TOGETHER_API_KEY", ""),
        num_examples=args.num_examples,
        include_solutions=args.include_solutions and not args.no_solutions,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        requests_per_minute=args.rpm,
        max_concurrent_requests=args.max_concurrent,
        retry_attempts=args.retry_attempts,
        checkpoint_interval=args.checkpoint_interval,
        output_path=args.output
    )
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate
    generator = ReasoningPromptGenerator(config)
    
    logger.info(f"Starting generation of {config.num_examples} problem pairs")
    logger.info(f"Model: {config.model}")
    logger.info(f"Rate limit: {config.requests_per_minute} RPM")
    logger.info(f"Max concurrent: {config.max_concurrent_requests}")
    logger.info(f"Include solutions: {config.include_solutions}")
    
    start_time = time.time()
    df = await generator.generate_batch(
        num_examples=config.num_examples,
        include_solutions=config.include_solutions,
        output_path=output_path
    )
    elapsed = time.time() - start_time
    
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Average rate: {len(df)/elapsed*60:.1f} examples/minute")
    
    # Show sample
    if len(df) > 0:
        logger.info("\nSample generated problem:")
        sample = df.iloc[0]
        logger.info(f"  Domain: {sample.get('domain')}")
        logger.info(f"  Difficulty: {sample.get('difficulty')}")
        logger.info(f"  Secret type: {sample.get('secret_type')}")
        logger.info(f"  Secret info: {sample.get('secret_info', '')[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
