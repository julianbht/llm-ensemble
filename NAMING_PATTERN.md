# Naming Pattern: Entity vs AdapterSpec

## Core Principle
Separate **domain entities** (persisted) from **adapter specifications** (wiring).

## The Pattern

```
{Thing}                     → Domain entity (persisted, has UUID)
{Thing}AdapterSpec          → Wiring spec (NOT persisted, module/class paths)
{Thing}Port                 → ABC defining adapter contract
{Thing}Adapter              → Concrete implementation
```

## Example: AggregationStrategy

```python
# Entity (persisted to DB)
class AggregationStrategy(BaseModel):
    id: UUID
    name: str
    @classmethod
    def create(cls, name: str) -> "AggregationStrategy": ...

# Adapter Spec (wiring only)
class AggregationStrategyAdapterSpec(BaseConfig):
    strategy_module: str
    strategy_class: str
    def get_strategy(self) -> AggregationStrategyPort: ...

# Port (ABC)
class AggregationStrategyPort(ABC):
    @property
    @abstractmethod
    def strategy_name(self) -> str: ...
    
    def aggregate(self, judgements) -> AggregatedVote:
        # Template method: creates entity from adapter's strategy_name
        data = self.aggregate_raw(judgements)
        strategy = AggregationStrategy.create(self.strategy_name)
        return AggregatedVote.create(aggregation_strategy=strategy, **data)

# Adapter (implementation)
class MajorityVoteAdapter(AggregationStrategyPort):
    @property
    def strategy_name(self) -> str:
        return "majority_vote"
    
    def aggregate_raw(self, judgements) -> dict:
        return {"final_label": ..., "final_confidence": ...}
```

## CLI Naming

```python
# Loader: load_{thing}_adapter(name: str) -> {Thing}AdapterSpec
def load_aggregation_strategy_adapter(spec_name: str) -> AggregationStrategyAdapterSpec: ...

# Param type: {Thing}AdapterParamType
class AggregationStrategyAdapterParamType(ConfigParamType): ...

# Param annotation: {Thing}AdapterSpecName
AggregationStrategyAdapterSpecName = Annotated[str, typer.Option(...)]

# CLI variable: {thing}_adapter_spec_name
def aggregate(aggregation_strategy_adapter_spec_name: AggregationStrategyAdapterSpecName): ...
```

## Rules
1. AdapterSpec = wiring only (NOT persisted)
2. Entity = domain object with UUID (persisted)
3. Port creates entities via template methods
4. Adapter owns identity via `{thing}_name` property
5. Use full entity name in variables (e.g., `aggregation_strategy`, not `strategy`)
