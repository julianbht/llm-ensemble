"""Tag management for run artifacts.

Tags provide user-friendly aliases for run names, allowing easy reference
between pipeline stages without typing full run names.

Each run directory contains a .tag file with the tag name, making tags
self-contained and preventing dangling references when artifacts are cleaned.

Example run directory structure:
    artifacts/runs/ingest/test/20251117_102232_llmjudge-json/
        .tag              ← contains "my-experiment"
        judging_samples.json
        run.log
        summary.json

Pipeline usage:
    ingest --tag my-experiment
    infer --input-tag my-experiment --tag my-experiment
    aggregate --input-tag my-experiment --tag my-experiment
    evaluate --input-tag my-experiment
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

from llm_ensemble.libs.runtime.path_manager import PathManager


class TagManager:
    """Manages run tags for easy cross-CLI referencing.
    
    Tags are stored as .tag files within each run directory, ensuring
    tags are automatically cleaned up when runs are deleted.
    """

    TAG_FILENAME = ".tag"

    @staticmethod
    def create_tag(run_dir: Path, tag_name: str) -> None:
        """Create a tag file in the run directory.
        
        Args:
            run_dir: Path to run directory
            tag_name: User-provided tag name (e.g., "my-experiment")
        """
        tag_file = run_dir / TagManager.TAG_FILENAME
        tag_file.write_text(tag_name)

    @staticmethod
    def get_tag(run_dir: Path) -> Optional[str]:
        """Get the tag name for a run directory, if it has one.
        
        Args:
            run_dir: Path to run directory
            
        Returns:
            Tag name if exists, None otherwise
        """
        tag_file = run_dir / TagManager.TAG_FILENAME
        if not tag_file.exists():
            return None
        return tag_file.read_text().strip()

    @staticmethod
    def resolve_tag(tag_name: str, source_cli: str) -> str:
        """Resolve a tag to its run_name by searching run directories.
        
        Searches both test and official runs for the source CLI.
        
        Args:
            tag_name: Tag name to resolve (e.g., "my-experiment")
            source_cli: CLI that created the tagged run (e.g., "ingest" when infer wants to read)
            
        Returns:
            Run name that the tag points to
            
        Raises:
            FileNotFoundError: If tag doesn't exist for the source CLI
        """
        # Search for tag in both test and official runs
        for official in [False, True]:
            run_type = "official" if official else "test"
            runs_dir = PathManager.get_artifacts_dir() / "runs" / source_cli / run_type
            
            if not runs_dir.exists():
                continue
            
            # Search all run directories for matching tag
            for run_dir in runs_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                
                tag = TagManager.get_tag(run_dir)
                if tag == tag_name:
                    return run_dir.name
        
        # Not found - provide helpful error
        available_tags = TagManager.list_tags(source_cli)
        raise FileNotFoundError(
            f"Tag '{tag_name}' not found for CLI '{source_cli}'.\n"
            f"Available tags: {', '.join(available_tags) or 'none'}"
        )

    @staticmethod
    def list_tags(cli_name: str) -> list[str]:
        """List all available tags for a CLI by scanning run directories.
        
        Args:
            cli_name: CLI name (e.g., "ingest", "infer")
            
        Returns:
            Sorted list of tag names
        """
        tags = []
        
        # Search both test and official runs
        for official in [False, True]:
            run_type = "official" if official else "test"
            runs_dir = PathManager.get_artifacts_dir() / "runs" / cli_name / run_type
            
            if not runs_dir.exists():
                continue
            
            # Check each run directory for a tag
            for run_dir in runs_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                
                tag = TagManager.get_tag(run_dir)
                if tag:
                    tags.append(tag)
        
        return sorted(set(tags))  # Remove duplicates and sort

    @staticmethod
    def tag_exists(tag_name: str, cli_name: str) -> bool:
        """Check if a tag exists for a CLI.
        
        Args:
            tag_name: Tag name
            cli_name: CLI name
            
        Returns:
            True if tag exists
        """
        try:
            TagManager.resolve_tag(tag_name, cli_name)
            return True
        except FileNotFoundError:
            return False

    @staticmethod
    def find_run_by_tag(tag_name: str, cli_name: str) -> Path:
        """Find the run directory path for a tagged run.
        
        Convenience method that combines resolve_tag with resolve_run_dir.
        
        Args:
            tag_name: Tag name to resolve
            cli_name: CLI name
            
        Returns:
            Full path to the run directory
            
        Raises:
            FileNotFoundError: If tag doesn't exist or run directory not found
        """
        run_name = TagManager.resolve_tag(tag_name, cli_name)
        return PathManager.resolve_run_dir(cli_name, run_name)

    @staticmethod
    def resolve_input(input_value: str, source_cli: str) -> str:
        """Resolve an input value that may be a tag (prefixed with @) or a run name.
        
        This is the main helper for CLI input parameters that support tags.
        
        Args:
            input_value: Input string, either "@tag_name" or "run_name"
            source_cli: CLI that created the run (e.g., "ingest" when infer reads input)
            
        Returns:
            Resolved run_name
            
        Raises:
            FileNotFoundError: If tag doesn't exist
            
        Examples:
            >>> resolve_input("@my-experiment", "ingest")
            "20251117_102232_llmjudge-json"
            
            >>> resolve_input("20251117_102232_llmjudge-json", "ingest")
            "20251117_102232_llmjudge-json"
        """
        if input_value.startswith("@"):
            # Strip @ and resolve as tag
            tag_name = input_value[1:]
            return TagManager.resolve_tag(tag_name, source_cli)
        else:
            # Already a run name
            return input_value
