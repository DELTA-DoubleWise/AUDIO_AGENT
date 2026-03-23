#!/usr/bin/env python3
"""
Tool environment setup utility using uv.

This module provides CLI and programmatic interfaces for setting up
isolated Python environments for MCP tools in the catalog.

Usage:
    # Setup single tool
    python -m audio_agent.tools.catalog.setup_tool asr_qwen

    # Setup all tools
    python -m audio_agent.tools.catalog.setup_tool --all

    # Force recreation of environment
    python -m audio_agent.tools.catalog.setup_tool asr_qwen --force

    # List tool environments and their status
    python -m audio_agent.tools.catalog.setup_tool --list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


class ToolStatus(NamedTuple):
    """Status of a tool environment."""
    name: str
    exists: bool
    has_pyproject: bool
    has_venv: bool
    venv_path: Path | None
    python_executable: Path | None


def get_catalog_dir() -> Path:
    """Get the path to the tools catalog directory."""
    return Path(__file__).parent


def get_tool_dir(tool_name: str, catalog_dir: Path | None = None) -> Path:
    """Get the path to a specific tool's directory."""
    if catalog_dir is None:
        catalog_dir = get_catalog_dir()
    return catalog_dir / tool_name


def check_uv_installed() -> bool:
    """Check if uv is installed and available."""
    try:
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_tool_status(tool_name: str, catalog_dir: Path | None = None) -> ToolStatus:
    """
    Check the status of a tool's environment.
    
    Args:
        tool_name: Name of the tool
        catalog_dir: Optional catalog directory path
        
    Returns:
        ToolStatus with environment information
    """
    tool_dir = get_tool_dir(tool_name, catalog_dir)
    
    exists = tool_dir.exists()
    pyproject_path = tool_dir / "pyproject.toml"
    has_pyproject = pyproject_path.exists()
    
    # Check for .venv in tool directory
    venv_path = tool_dir / ".venv"
    python_executable = venv_path / "bin" / "python"
    has_venv = venv_path.exists() and python_executable.exists()
    
    return ToolStatus(
        name=tool_name,
        exists=exists,
        has_pyproject=has_pyproject,
        has_venv=has_venv,
        venv_path=venv_path if has_venv else None,
        python_executable=python_executable if has_venv else None,
    )


def setup_tool_env(
    tool_name: str,
    catalog_dir: Path | None = None,
    force: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Set up isolated environment for a tool using uv.
    
    This function:
    1. Locates the tool directory
    2. Checks for pyproject.toml
    3. Runs `uv sync` or `uv venv` + `uv pip install`
    4. Verifies the environment works
    
    Args:
        tool_name: Name of the tool to set up
        catalog_dir: Optional catalog directory path
        force: If True, recreate environment even if it exists
        verbose: If True, print progress messages
        
    Returns:
        True if setup succeeded, False otherwise
    """
    if verbose:
        print(f"Setting up environment for tool: {tool_name}")
    
    # Check uv is installed
    if not check_uv_installed():
        print("Error: uv is not installed. Please install uv first:")
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("Or visit: https://github.com/astral-sh/uv")
        return False
    
    tool_dir = get_tool_dir(tool_name, catalog_dir)
    
    # Check tool directory exists
    if not tool_dir.exists():
        print(f"Error: Tool directory not found: {tool_dir}")
        return False
    
    # Check for pyproject.toml
    pyproject_path = tool_dir / "pyproject.toml"
    if not pyproject_path.exists():
        print(f"Error: No pyproject.toml found in {tool_dir}")
        print("Tools must define their dependencies in pyproject.toml")
        return False
    
    venv_path = tool_dir / ".venv"
    
    # Handle existing environment
    if venv_path.exists():
        if force:
            if verbose:
                print(f"  Removing existing environment (force=True)")
            import shutil
            shutil.rmtree(venv_path)
        else:
            if verbose:
                print(f"  Environment already exists at {venv_path}")
                print(f"  Use --force to recreate")
            return True
    
    # Create environment using uv
    if verbose:
        print(f"  Creating virtual environment...")
    
    try:
        # Use uv sync if uv.lock exists, otherwise uv venv + uv pip install
        lock_file = tool_dir / "uv.lock"
        
        # Check for Python version requirement in pyproject.toml
        import re
        python_version = "3.11"  # default
        try:
            with open(pyproject_path) as f:
                content = f.read()
                match = re.search(r'requires-python\s*=\s*">=?(\d+\.\d+)"', content)
                if match:
                    python_version = match.group(1)
        except:
            pass
        
        if lock_file.exists():
            # Use uv sync for reproducible installs
            result = subprocess.run(
                ["uv", "sync", f"--python={python_version}"],
                cwd=tool_dir,
                capture_output=True,
                text=True,
                check=True,
            )
        else:
            # Create venv with specific Python version and install dependencies
            result = subprocess.run(
                ["uv", "venv", ".venv", f"--python={python_version}"],
                cwd=tool_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            result = subprocess.run(
                ["uv", "pip", "install", ".", f"--python={python_version}"],
                cwd=tool_dir,
                capture_output=True,
                text=True,
                check=True,
            )
        
        if verbose:
            print(f"  ✓ Environment created successfully")
            print(f"  Location: {venv_path}")
            print(f"  Python: {venv_path / 'bin' / 'python'}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to create environment")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error: Unexpected error: {e}")
        return False


def setup_all_tools(
    catalog_dir: Path | None = None,
    force: bool = False,
    verbose: bool = True,
) -> dict[str, bool]:
    """
    Set up environments for all tools in the catalog.
    
    Args:
        catalog_dir: Optional catalog directory path
        force: If True, recreate environments even if they exist
        verbose: If True, print progress messages
        
    Returns:
        Dictionary mapping tool names to success status
    """
    if catalog_dir is None:
        catalog_dir = get_catalog_dir()
    
    # Find all tool directories (those with config.yaml)
    tool_dirs = [
        d for d in catalog_dir.iterdir()
        if d.is_dir() and (d / "config.yaml").exists() and not d.name.startswith("_")
    ]
    
    results = {}
    
    for tool_dir in sorted(tool_dirs):
        tool_name = tool_dir.name
        if verbose:
            print()
        success = setup_tool_env(tool_name, catalog_dir, force, verbose)
        results[tool_name] = success
    
    return results


def verify_tool_env(
    tool_name: str,
    catalog_dir: Path | None = None,
    verbose: bool = True,
) -> bool:
    """
    Verify that a tool's environment is ready for use.
    
    This checks:
    1. Tool directory exists
    2. pyproject.toml exists
    3. Virtual environment exists
    4. Python executable is functional
    
    Args:
        tool_name: Name of the tool to verify
        catalog_dir: Optional catalog directory path
        verbose: If True, print status messages
        
    Returns:
        True if environment is ready, False otherwise
    """
    if verbose:
        print(f"Verifying environment for tool: {tool_name}")
    
    status = check_tool_status(tool_name, catalog_dir)
    
    # Check directory exists
    if not status.exists:
        if verbose:
            print(f"  ✗ Tool directory not found")
        return False
    
    if verbose:
        print(f"  ✓ Tool directory exists")
    
    # Check pyproject.toml
    if not status.has_pyproject:
        if verbose:
            print(f"  ✗ pyproject.toml not found")
        return False
    
    if verbose:
        print(f"  ✓ pyproject.toml exists")
    
    # Check virtual environment
    if not status.has_venv:
        if verbose:
            print(f"  ✗ Virtual environment not found (.venv)")
            print(f"    Run: python -m audio_agent.tools.catalog.setup_tool {tool_name}")
        return False
    
    if verbose:
        print(f"  ✓ Virtual environment exists")
    
    # Test Python executable
    if status.python_executable:
        try:
            result = subprocess.run(
                [str(status.python_executable), "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                if verbose:
                    print(f"  ✓ Python executable works: {result.stdout.strip()}")
            else:
                if verbose:
                    print(f"  ✗ Python executable failed")
                return False
        except Exception as e:
            if verbose:
                print(f"  ✗ Python executable error: {e}")
            return False
    
    if verbose:
        print(f"  ✓ Environment is ready for use")
    
    return True


def verify_all_tools(
    catalog_dir: Path | None = None,
    verbose: bool = True,
) -> dict[str, bool]:
    """
    Verify all tool environments.
    
    Args:
        catalog_dir: Optional catalog directory path
        verbose: If True, print progress messages
        
    Returns:
        Dictionary mapping tool names to verification status
    """
    if catalog_dir is None:
        catalog_dir = get_catalog_dir()
    
    # Find all tool directories
    tool_dirs = [
        d for d in catalog_dir.iterdir()
        if d.is_dir() and (d / "config.yaml").exists() and not d.name.startswith("_")
    ]
    
    results = {}
    
    for tool_dir in sorted(tool_dirs):
        tool_name = tool_dir.name
        if verbose:
            print()
        success = verify_tool_env(tool_name, catalog_dir, verbose)
        results[tool_name] = success
    
    return results


def list_tool_statuses(catalog_dir: Path | None = None) -> list[ToolStatus]:
    """
    List status of all tool environments.
    
    Args:
        catalog_dir: Optional catalog directory path
        
    Returns:
        List of ToolStatus for all tools
    """
    if catalog_dir is None:
        catalog_dir = get_catalog_dir()
    
    # Find all tool directories
    tool_dirs = [
        d for d in catalog_dir.iterdir()
        if d.is_dir() and (d / "config.yaml").exists() and not d.name.startswith("_")
    ]
    
    return [check_tool_status(d.name, catalog_dir) for d in sorted(tool_dirs)]


def print_tool_list(catalog_dir: Path | None = None) -> None:
    """Print formatted list of tool environments."""
    statuses = list_tool_statuses(catalog_dir)
    
    print("Tool Environment Status")
    print("=" * 70)
    print(f"{'Tool Name':<20} {'Status':<12} {'pyproject':<10} {'venv':<8} {'Python':<12}")
    print("-" * 70)
    
    for status in statuses:
        if not status.exists:
            state = "MISSING"
        elif status.has_venv:
            state = "READY"
        elif status.has_pyproject:
            state = "NEEDS_SETUP"
        else:
            state = "NO_DEPS"
        
        pyproject = "✓" if status.has_pyproject else "✗"
        venv = "✓" if status.has_venv else "✗"
        python_ver = ""
        if status.python_executable:
            try:
                result = subprocess.run(
                    [str(status.python_executable), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    python_ver = result.stdout.strip().replace("Python ", "")
            except:
                pass
        
        print(f"{status.name:<20} {state:<12} {pyproject:<10} {venv:<8} {python_ver:<12}")
    
    print()
    print("Legend:")
    print("  READY       - Environment exists and is ready to use")
    print("  NEEDS_SETUP - Has pyproject.toml but needs uv setup")
    print("  NO_DEPS     - No pyproject.toml (no dependencies declared)")
    print("  MISSING     - Tool directory not found")
    print()
    print("To setup a tool:")
    print("  python -m audio_agent.tools.catalog.setup_tool <tool_name>")
    print()
    print("To verify a tool is ready:")
    print("  python -m audio_agent.tools.catalog.setup_tool <tool_name> --verify")


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Set up and manage tool environments using uv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s asr_qwen              # Set up single tool
  %(prog)s --all                 # Set up all tools
  %(prog)s asr_qwen --force      # Force recreate environment
  %(prog)s --list                # List all tool statuses
  %(prog)s asr_qwen --verify     # Verify tool is ready
  %(prog)s --verify-all          # Verify all tools
        """,
    )
    
    parser.add_argument(
        "tool",
        nargs="?",
        help="Name of the tool to set up or verify",
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Set up all tools in the catalog",
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List status of all tool environments",
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreation of environment even if it exists",
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify tool environment is ready (don't create)",
    )
    
    parser.add_argument(
        "--verify-all",
        action="store_true",
        help="Verify all tool environments",
    )
    
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=None,
        help="Path to tools catalog directory (default: auto-detect)",
    )
    
    return parser


def main() -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        print_tool_list(args.catalog_dir)
        return 0
    
    # Handle --verify-all
    if args.verify_all:
        results = verify_all_tools(args.catalog_dir, verbose=True)
        
        # Summary
        print()
        print("=" * 60)
        print("Verification Summary")
        print("-" * 60)
        ready_count = sum(1 for v in results.values() if v)
        not_ready_count = len(results) - ready_count
        print(f"  Ready: {ready_count}")
        print(f"  Not Ready: {not_ready_count}")
        
        if not_ready_count > 0:
            print()
            print("Tools not ready:")
            for name, ready in results.items():
                if not ready:
                    print(f"  - {name}")
            print()
            print("To setup a tool, run:")
            print("  python -m audio_agent.tools.catalog.setup_tool <tool_name>")
        
        return 0 if not_ready_count == 0 else 1
    
    # Handle --verify for single tool
    if args.verify and args.tool:
        success = verify_tool_env(args.tool, args.catalog_dir, verbose=True)
        return 0 if success else 1
    
    # Handle --all (setup)
    if args.all:
        results = setup_all_tools(args.catalog_dir, args.force, verbose=True)
        
        # Summary
        print()
        print("=" * 60)
        print("Setup Summary")
        print("-" * 60)
        success_count = sum(1 for v in results.values() if v)
        failed_count = len(results) - success_count
        print(f"  Successful: {success_count}")
        print(f"  Failed: {failed_count}")
        
        if failed_count > 0:
            print()
            print("Failed tools:")
            for name, success in results.items():
                if not success:
                    print(f"  - {name}")
        
        return 0 if failed_count == 0 else 1
    
    # Handle single tool setup
    if args.tool:
        success = setup_tool_env(args.tool, args.catalog_dir, args.force, verbose=True)
        return 0 if success else 1
    
    # No arguments provided
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
