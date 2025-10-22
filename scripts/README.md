# Scripts

Utility scripts for development and maintenance.

## generate_schemas.py

Generates JSON schemas from Pydantic models and consolidates them in `src/llm_ensemble/libs/schemas/`.

**Purpose:**
- Creates a centralized location for all schemas (data contracts, configs, domain models)
- Generates JSON schemas from Pydantic models for documentation and validation
- Useful for creating class diagrams, API documentation, and schema registries

**Usage:**
```bash
# Using Make (recommended)
make schemas

# Or directly
python scripts/generate_schemas.py
```

**Output:**
- JSON schema files in `src/llm_ensemble/libs/schemas/`
- `schema-index.json` - index of all schemas organized by category

**Schema Categories:**

1. **Data Contracts** (`data_contracts/`) - Define data flow between CLI stages
   - `judging-example.schema.json` (JudgingExample) - ingest → infer
   - `model-judgement.schema.json` (ModelJudgement) - infer → aggregate
   - `ensemble-result.schema.json` (EnsembleResult) - aggregate → evaluate
   - `evaluation-metrics.schema.json` (EvaluationMetrics) - evaluate → final

2. **Configuration Schemas** (`configurations/`) - Define YAML config file formats
   - `model-config.schema.json` (ModelConfig)
   - `prompt-config.schema.json` (PromptConfig)
   - `io-config.schema.json` (IOConfig)

3. **Internal Domain Models** (`internal/`) - CLI-specific components (for documentation)
   - `query.schema.json` (Query)
   - `document.schema.json` (Document)
   - `relevance.schema.json` (Relevance)

**When to run:**
- After modifying any Pydantic schema definitions
- Before generating documentation or class diagrams
- When setting up the project for the first time

**Adding new schemas:**
Edit the `SCHEMAS` list in `generate_schemas.py` to include new Pydantic models.
