# Prompt Templates

Jinja2 templates for LLM judge prompts. Each prompt consists of two files:

## Files

- **`<name>.jinja`** — Jinja2 template with `{{ query }}`, `{{ page_text }}` placeholders
- **`<name>.yaml`** — Config with metadata and template variables

## How It Works

1. **Infer CLI** receives `--prompt <name>` flag
2. **Prompt loader** reads `<name>.yaml` to get config
3. **Config specifies** which `.jinja` file to use and what variables to pass
4. **Template engine** renders `.jinja` with variables from config + runtime data (query, document)
5. **Rendered prompt** sent to LLM

## Config Structure

```yaml
name: thomas-et-al-prompt
template_file: thomas-et-al-prompt.jinja   # Which Jinja2 file to use
variables:
  role: true          # Custom variables passed to template
  aspects: false
expected_output_format: json
response_parser: parse_thomas_response
```

## Usage

```bash
infer --prompt thomas-et-al-prompt --model gpt-oss-20b --input samples.ndjson
```
