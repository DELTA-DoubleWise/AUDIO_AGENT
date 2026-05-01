"""
Multi-round retry orchestrator for benchmark failed samples.

Acts as a "safety net" that runs after the main benchmark loop and
deferred retry phase. Collects retryable failures and retries them in
multiple rounds with fixed cooldown between rounds.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from tqdm import tqdm


class RetryOrchestrator:
    """
    Orchestrates multi-round retry for failed benchmark samples.

    After an initial benchmark run completes, this orchestrator collects
    all failed/retryable samples and retries them in multiple rounds
    with configurable cooldown between rounds.
    """

    # Error patterns that indicate transient/rate-limit failures worth retrying
    DEFERRABLE_ERROR_PATTERNS = ["10001", "token limit", "rate limit", "too many tokens"]

    # Match methods that are typically worth retrying
    DEFAULT_RETRYABLE_MATCH_METHODS = ["runtime_error", "no_prediction"]

    def __init__(
        self,
        max_rounds: int = 3,
        cooldown_seconds: float = 10.0,
        retryable_match_methods: list[str] | None = None,
    ):
        """
        Args:
            max_rounds: Maximum retry rounds (0 = disable).
            cooldown_seconds: Seconds to wait between retry rounds.
            retryable_match_methods: List of match_methods to retry.
        """
        self.max_rounds = max_rounds
        self.cooldown_seconds = cooldown_seconds
        self.retryable_match_methods = retryable_match_methods or list(
            self.DEFAULT_RETRYABLE_MATCH_METHODS
        )

    def is_retryable(self, result: dict) -> bool:
        """Determine if a failed result is worth retrying."""
        if result.get("correct", False):
            return False

        error_str = str(result.get("error", "")).lower()
        match_method = result.get("match_method", "")

        # Always retry known transient errors
        for pattern in self.DEFERRABLE_ERROR_PATTERNS:
            if pattern in error_str:
                return True

        # Retry based on match method
        if match_method in self.retryable_match_methods:
            return True

        # Retry if there's no prediction but no explicit non-retryable error
        if match_method == "no_prediction" or not match_method:
            return True

        return False

    def get_failed_samples(self, results: list[dict]) -> list[dict]:
        """Extract retryable failed samples from results."""
        return [r for r in results if self.is_retryable(r)]

    async def run_retry_rounds(
        self,
        results: list[dict],
        retry_fn: Callable[[int, dict], dict],
        save_fn: Callable[[list[dict]], Any],
    ) -> list[dict]:
        """
        Run multi-round retry on failed samples.

        Args:
            results: Current results list (modified in-place).
            retry_fn: Callable(idx: int, old_result: dict) -> new_result_dict
            save_fn: Callable(updated_results: list[dict]) -> None

        Returns:
            Updated results list after all retry rounds.
        """
        results = list(results)  # Work on a copy

        for round_num in range(1, self.max_rounds + 1):
            failed = self.get_failed_samples(results)

            if not failed:
                print(
                    f"\n[RetryOrchestrator] No retryable failures remaining "
                    f"after round {round_num - 1}. Done!"
                )
                break

            print(f"\n{'='*60}")
            print(f"  Retry Round {round_num}/{self.max_rounds}")
            print(f"  Retryable failures: {len(failed)}")
            print(f"  Cooling down for {self.cooldown_seconds}s...")
            print(f"{'='*60}")

            await asyncio.sleep(self.cooldown_seconds)

            success_count = 0
            still_failed_count = 0

            for failed_result in tqdm(failed, desc=f"Retry R{round_num}"):
                idx = failed_result["index"]

                try:
                    new_result = await retry_fn(idx, failed_result)

                    # Replace old result in list
                    for i, r in enumerate(results):
                        if r["index"] == idx:
                            results[i] = new_result
                            break

                    if new_result.get("correct", False):
                        success_count += 1
                    else:
                        still_failed_count += 1

                except Exception as e:
                    print(f"\n[Retry R{round_num}] Sample {idx} retry exception: {e}")
                    still_failed_count += 1

            # Save after each round
            save_fn(results)

            print(
                f"\n[RetryOrchestrator] Round {round_num} complete: "
                f"{success_count} fixed, {still_failed_count} still failed"
            )

        return results
