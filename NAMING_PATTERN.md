# Clean Naming Pattern: Config vs AdapterConfig vs Entity

## Overview

Clear separation between wiring (how to load) and domain entities (what gets persisted).

## Pattern

### `{Thing}AdapterConfig` = Pure Wiring (NOT Persisted)

Infrastructure configuration specifying module/class paths for dynamic adapter loading.

**Examples:**
- `StrategyAdapterConfig` - wiring for aggregation strategy adapter
- `PromptAdapterConfig` - wiring for prompt builder + parser adapters (planned)
- `ProviderAdapterConfig` - wiring for provider adapter (planned)

**Characteristics:**
- Contains `*_module` and `*_class` fields
- Has `get_{thing}()` method for adapter instantiation
- **NO `id` field** - not an entity
- Lives in `*_adapter_config.py` files

### `{Thing}` = Domain Entity (Persisted)

Actual business domain objects that get persisted to database.

**Examples:**
- `AggregationStrategy` - minimal entity (id, name)
- `PromptTemplate` - full entity (id, name, template_text)
- `Parser` - minimal entity (id, name, code_hash)
- `Provider` - minimal entity (id, name)
- `Model` - full entity (id, name, all params)

**Characteristics:**
- Has `id: UUID` field
- Has `.create(name, ...)` classmethod that computes ID
- Corresponds to `{Thing}ORM` in database
- Pure domain entities

### `{Thing}Port` = Adapter Port (ABC)

Abstract base class defining contract for adapters.

**Examples:**
- `AggregationStrategyPort` - ABC for strategy adapters
- `PromptBuilderPort` - ABC for prompt builder adapters (planned)
- `ResponseParserPort` - ABC for parser adapters (planned)

**Characteristics:**
- Inherits from `ABC`
- Defines abstract methods adapters must implement
- May include template method pattern (concrete + abstract methods)

## Aggregate Pipeline Example

### Wiring (AdapterConfig)

```python
# strategy_adapter_config.py
class StrategyAdapterConfig(BaseConfig):
    """Pure wiring - NOT persisted."""
    strategy_module: str  # e.g., "...majority_vote_adapter"
    strategy_class: str   # e.g., "MajorityVoteAdapter"
    name_hint: Optional[str]  # for run naming only

    def get_strategy(self) -> AggregationStrategyPort:
        return self._instantiate_adapter(
            self.strategy_module,
            self.strategy_class
        )
```

### Entity

```python
# aggregation_strategy.py
class AggregationStrategy(BaseModel):
    """Domain entity - persisted."""
    id: UUID
    name: str  # e.g., "majority_vote"

    @classmethod
    def create(cls, strategy_name: str) -> "AggregationStrategy":
        strategy_id = compute_aggregation_spec_uuid(strategy_name)
        return cls(id=strategy_id, name=strategy_name)
```

### Port

```python
# ports/aggregation_strategy.py
class AggregationStrategyPort(ABC):
    """Port for strategy adapters."""

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Natural key - adapter owns its identity."""
        pass

    @abstractmethod
    def aggregate_raw(self, judgements) -> dict:
        """Pure logic - returns dict."""
        pass

    def aggregate(self, judgements) -> AggregatedVote:
        """Template method - creates domain objects."""
        vote_data = self.aggregate_raw(judgements)
        strategy = AggregationStrategy.create(self.strategy_name)
        return AggregatedVote.create(
            aggregation_strategy=strategy,
            llm_judgements=judgements,
            **vote_data
        )
```

### Adapter

```python
# adapters/strategies/majority_vote_adapter.py
class MajorityVoteAdapter(AggregationStrategyPort):
    """Concrete adapter - implements port."""

    @property
    def strategy_name(self) -> str:
        return "majority_vote"

    def aggregate_raw(self, judgements) -> dict:
        # Pure voting logic
        return {"final_label": ..., "final_confidence": ..., "final_reasoning": ...}
```

## Key Principles

1. **AdapterConfig** = Wiring only (module/class paths) - NOT persisted
2. **Entity** = Domain objects with IDs - persisted
3. **Port** = ABC defining adapter contract
4. **Adapter** = Concrete implementation, owns its entity identity via `{thing}_name` property
5. **Port creates entities** using adapter's `{thing}_name` property (template method pattern)

## Benefits

1. **Clear separation** - wiring vs domain
2. **No ID injection** - adapter owns its identity
3. **Minimal entities** - only essential data persisted
4. **Clean testing** - adapters return simple dicts, port handles domain object creation
5. **Explicit** - code clearly shows what's wiring vs what's business logic
