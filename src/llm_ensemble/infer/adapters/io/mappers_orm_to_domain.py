"""ORM to domain mappers for INFER CLI (reading direction).

This module provides conversion functions for reconstructing Pydantic domain objects
from SQLAlchemy ORM entities. These are the reverse operations of mappers_domain_to_orm.

Design principles:
- ORM entities are queried with eager loading (avoid N+1 queries)
- Mappers convert ORMs → domain objects for reconstruction (SELECT operations)
- Stateless pure functions
- Reuse mappers from ingest CLI for JudgingSample reconstruction
- Used by SqlJudgementReader (in aggregate CLI) for reading inference results

The domain layer works with Pydantic models (LLMJudgement).
The persistence layer works with SQLAlchemy ORMs (LLMCallORM, etc.).
These mappers handle the impedance mismatch for the read path.
"""

from __future__ import annotations

from llm_ensemble.infer.schemas.llm_judgement import (
    LLMJudgement,
    LLMResponse,
    LLMScore,
)
from llm_ensemble.infer.schemas.orms_normalized import LLMCallORM
from llm_ensemble.infer.schemas.warnings import BaseWarning
from llm_ensemble.ingest.adapters.io.mappers import judging_sample_from_orm


def llm_judgement_from_orm(call_orm: LLMCallORM) -> LLMJudgement:
    """Reconstruct complete LLMJudgement from LLMCallORM.
    
    Requires eager loading of relationships:
    - llm_request (with judging_sample)
    - response (with parser_spec)
    
    Args:
        call_orm: LLMCallORM entity with eager-loaded relationships
        
    Returns:
        Complete LLMJudgement domain object
        
    Raises:
        ValueError: If required relationships are not loaded
    """
    if not call_orm.llm_request:
        raise ValueError("LLMCallORM.llm_request must be eager-loaded")
    if not call_orm.response:
        raise ValueError("LLMCallORM.response must be eager-loaded")
    if not call_orm.llm_request.judging_sample:
        raise ValueError("LLMRequestORM.judging_sample must be eager-loaded")
    
    request_orm = call_orm.llm_request
    response_orm = call_orm.response
    sample_orm = request_orm.judging_sample
    
    # Reconstruct JudgingSample using ingest mapper
    judging_sample = judging_sample_from_orm(
        sample_orm,
        query=None,  # Will be reconstructed by the mapper from sample_orm.query
        document=None,  # Will be reconstructed by the mapper from sample_orm.document
    )
    
    # Reconstruct LLMResponse from response ORM and call metadata
    llm_response = LLMResponse(
        raw_response=response_orm.raw_response,
        latency_ms=call_orm.latency_ms,
        retries=call_orm.retries,
        cost_estimate_usd=call_orm.cost_estimate_usd,
        generation_id=call_orm.generation_id,
        prompt_tokens=call_orm.prompt_tokens,
        completion_tokens=call_orm.completion_tokens,
        total_tokens=call_orm.total_tokens,
        warnings=[],  # Provider warnings not stored in current schema
    )
    
    # Reconstruct LLMScore from parsed response fields
    parser_warnings = []
    if response_orm.parser_warnings:
        # Parser warnings stored as JSONB array of dicts
        # Reconstruct as BaseWarning objects (polymorphic list)
        parser_warnings = [
            BaseWarning(**w) for w in response_orm.parser_warnings
        ]
    
    llm_score = LLMScore(
        label=response_orm.label,
        confidence=response_orm.confidence,
        rationale=response_orm.rationale,
        warnings=parser_warnings,
    )
    
    # Reconstruct complete LLMJudgement
    return LLMJudgement(
        judging_sample=judging_sample,
        prompt=request_orm.prompt,
        llm_response=llm_response,
        llm_score=llm_score,
    )
