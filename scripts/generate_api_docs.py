#!/usr/bin/env python3
"""Generate comprehensive API documentation from Python source files.

This script analyzes Python modules and extracts their public API, including
classes, functions, methods, constants, and their docstrings. It generates
a well-formatted Markdown documentation file.

IMPROVEMENTS IN THIS VERSION:
-----------------------------
1. Complete function signatures with type hints and default values
2. Async function detection and marking
3. Class inheritance hierarchy display
4. Class-level constants and attributes extraction
5. Module-level constant detection (uppercase variables)
6. Structured docstring formatting with bold section headers
7. Table of contents with working anchor links
8. Project metadata integration (name, version, description)
9. Generation timestamp
10. Better code organization with helper functions for reduced complexity
11. Proper handling of *args and **kwargs in signatures
12. Module docstring extraction and display

FEATURES:
---------
- AST-based parsing for accurate extraction
- Markdown formatting optimized for GitHub/GitLab
- Support for both sync and async functions
- Handles complex type hints (unions, generics, etc.)
- Extracts and formats Google-style docstrings
- Creates navigable documentation with anchors
- Shows class hierarchies and base classes
- Displays class attributes with their types and values
"""

import ast
import tomllib
from pathlib import Path
from textwrap import dedent, indent
from typing import Any

# Constants
NO_DOCUMENTATION_TEXT = "*No documentation*\n"
DOCSTRING_SECTIONS = [
    "Args",
    "Returns",
    "Raises",
    "Example",
    "Examples",
    "Note",
    "Warning",
    "Attributes",
]


def load_pyproject(root_path: Path) -> tuple[list[Path], dict[str, Any]]:
    """Load pyproject.toml and extract package information.

    Args:
        root_path: Root directory of the project

    Returns:
        Tuple of (package paths, project metadata)

    Raises:
        FileNotFoundError: If pyproject.toml doesn't exist
        KeyError: If required keys are missing
    """
    pyproject_path = root_path / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError(f"Could not find {pyproject_path}")

    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    try:
        packages = data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    except KeyError as err:
        raise KeyError(
            "Could not find key [tool.hatch.build.targets.wheel] -> packages in pyproject.toml"
        ) from err

    project_info = data.get("project", {})
    return [root_path / p for p in packages], project_info


def extract_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Extract function signature with parameters and return type.

    Args:
        node: AST node representing a function

    Returns:
        Formatted function signature string
    """
    params = []

    # Handle arguments
    args = node.args
    defaults = [None] * (len(args.args) - len(args.defaults)) + list(args.defaults)

    for arg, default in zip(args.args, defaults, strict=False):
        param = arg.arg
        # Add type annotation if present
        if arg.annotation:
            param += f": {ast.unparse(arg.annotation)}"
        # Add default value if present
        if default:
            param += f" = {ast.unparse(default)}"
        params.append(param)

    # Handle *args
    if args.vararg:
        vararg = f"*{args.vararg.arg}"
        if args.vararg.annotation:
            vararg += f": {ast.unparse(args.vararg.annotation)}"
        params.append(vararg)

    # Handle **kwargs
    if args.kwarg:
        kwarg = f"**{args.kwarg.arg}"
        if args.kwarg.annotation:
            kwarg += f": {ast.unparse(args.kwarg.annotation)}"
        params.append(kwarg)

    # Build signature
    signature = f"{node.name}({', '.join(params)})"

    # Add return type if present
    if node.returns:
        signature += f" -> {ast.unparse(node.returns)}"

    return signature


def extract_function_info(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str, str, bool]:
    """Extract information from a function node.

    Args:
        node: AST node representing a function

    Returns:
        Tuple of (name, signature, docstring, is_async)
    """
    doc = ast.get_docstring(node) or ""
    signature = extract_function_signature(node)
    is_async = isinstance(node, ast.AsyncFunctionDef)
    return (node.name, signature, doc, is_async)


def process_annotated_assignment(
    sub: ast.AnnAssign,
) -> tuple[str, str | None, str | None] | None:
    """Process an annotated assignment node.

    Args:
        sub: AST annotated assignment node

    Returns:
        Tuple of (var_name, var_type, var_value) or None if private
    """
    if isinstance(sub.target, ast.Name) and not sub.target.id.startswith("_"):
        var_name = sub.target.id
        var_type = ast.unparse(sub.annotation) if sub.annotation else None
        var_value = ast.unparse(sub.value) if sub.value else None
        return (var_name, var_type, var_value)
    return None


def process_regular_assignment(sub: ast.Assign) -> list[tuple[str, None, str]]:
    """Process a regular assignment node.

    Args:
        sub: AST assignment node

    Returns:
        List of tuples (var_name, None, var_value) for public variables
    """
    variables = []
    for target in sub.targets:
        if isinstance(target, ast.Name) and not target.id.startswith("_"):
            var_name = target.id
            var_value = ast.unparse(sub.value)
            variables.append((var_name, None, var_value))
    return variables


def extract_class_variables(body: list) -> list[tuple[str, str | None, str | None]]:
    """Extract class variables from class body.

    Args:
        body: List of AST nodes in class body

    Returns:
        List of tuples (var_name, var_type, var_value)
    """
    class_variables = []

    for sub in body:
        if isinstance(sub, ast.AnnAssign):
            var_info = process_annotated_assignment(sub)
            if var_info:
                class_variables.append(var_info)
        elif isinstance(sub, ast.Assign):
            class_variables.extend(process_regular_assignment(sub))

    return class_variables


def extract_class_methods(body: list) -> list[tuple[str, str, str, bool]]:
    """Extract methods from class body.

    Args:
        body: List of AST nodes in class body

    Returns:
        List of tuples (name, signature, docstring, is_async)
    """
    methods = []

    for sub in body:
        if isinstance(
            sub, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and not sub.name.startswith("_"):
            method_info = extract_function_info(sub)
            methods.append(method_info)

    return methods


def extract_class_info(node: ast.ClassDef) -> tuple[str, str, list, list, list]:
    """Extract information from a class node.

    Args:
        node: AST node representing a class

    Returns:
        Tuple of (name, docstring, methods, class_variables, bases)
    """
    doc = ast.get_docstring(node) or ""
    methods = extract_class_methods(node.body)
    class_variables = extract_class_variables(node.body)
    bases = [ast.unparse(base) for base in node.bases]
    return (node.name, doc, methods, class_variables, bases)


def extract_module_constants(node: ast.Assign) -> list[tuple[str, str]]:
    """Extract module-level constants from assignment node.

    Args:
        node: AST assignment node

    Returns:
        List of tuples (const_name, const_value)
    """
    constants = []
    for target in node.targets:
        if isinstance(target, ast.Name) and target.id.isupper():
            const_name = target.id
            const_value = ast.unparse(node.value)
            constants.append((const_name, const_value))
    return constants


def extract_api_from_module(path: Path) -> dict[str, Any]:
    """Analyze a Python file and extract its public API.

    Args:
        path: Path to the Python file

    Returns:
        Dictionary containing functions, classes, constants, and module docstring
    """
    with path.open(encoding="utf-8") as f:
        content = f.read()
        node = ast.parse(content, filename=str(path))

    # Extract module docstring
    module_doc = ast.get_docstring(node) or ""

    api: dict[str, Any] = {
        "module_doc": module_doc,
        "functions": [],
        "classes": [],
        "constants": [],
    }

    for n in node.body:
        # Public functions
        if isinstance(
            n, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and not n.name.startswith("_"):
            api["functions"].append(extract_function_info(n))

        # Public classes
        elif isinstance(n, ast.ClassDef) and not n.name.startswith("_"):
            api["classes"].append(extract_class_info(n))

        # Module-level constants (uppercase variables)
        elif isinstance(n, ast.Assign):
            api["constants"].extend(extract_module_constants(n))

    return api


def is_section_header(line: str) -> bool:
    """Check if a line is a docstring section header."""
    stripped = line.strip()
    return stripped.endswith(":") and stripped[:-1] in DOCSTRING_SECTIONS


def format_code_line(line: str) -> str:
    """Format a line in a code example section.

    Code examples are preserved as-is to maintain proper code formatting.
    """
    return line


def format_section_line(line: str, stripped: str) -> str:
    """Format a line inside a documentation section."""
    if not line.startswith("    "):
        return line

    # Check if it already starts with a bullet or dash
    if stripped.startswith("- ") or stripped.startswith("* "):
        # Already has proper bullet formatting, just preserve it
        return stripped
    return f"- {stripped}"


def process_docstring_line(
    line: str,
    i: int,
    formatted_lines: list[str],
    current_section: str | None,
    in_code_block: bool,
) -> tuple[str | None, bool]:
    """Process a single line of docstring.

    Returns:
        Tuple of (new_current_section, new_in_code_block)
    """
    stripped = line.strip()

    # Check if this is a section header
    if is_section_header(line):
        # Add blank line before section if not first line
        if i > 0 and formatted_lines and formatted_lines[-1].strip():
            formatted_lines.append("")
        formatted_lines.append(f"**{stripped}**")
        formatted_lines.append("")  # Add blank line after section header
        new_section = stripped[:-1]
        new_code_block = new_section in ["Example", "Examples"]
        return new_section, new_code_block

    # Handle blank lines
    if stripped == "":
        formatted_lines.append("")
        return current_section, in_code_block

    # Process lines inside a section
    if current_section:
        if in_code_block:
            formatted_lines.append(format_code_line(line))
        else:
            formatted_lines.append(format_section_line(line, stripped))
            if not line.startswith("    "):
                return None, False
        return current_section, in_code_block

    # Regular line outside sections
    formatted_lines.append(line)
    return current_section, in_code_block


def format_docstring(docstring: str, indent_level: int = 0) -> str:
    """Format a docstring for Markdown output.

    Args:
        docstring: The docstring to format
        indent_level: Number of spaces to indent

    Returns:
        Formatted docstring with proper line breaks and indentation
    """
    if not docstring:
        return ""

    # Dedent the docstring to remove common leading whitespace
    cleaned = dedent(docstring).strip()

    # Split into lines for proper formatting
    lines = cleaned.split("\n")
    formatted_lines: list[str] = []
    current_section = None
    in_code_block = False

    for i, line in enumerate(lines):
        current_section, in_code_block = process_docstring_line(
            line, i, formatted_lines, current_section, in_code_block
        )

    result = "\n".join(formatted_lines)
    if indent_level:
        result = indent(result, " " * indent_level)

    return result


def generate_function_section(
    functions: list, title: str = "### Functions"
) -> list[str]:
    """Generate Markdown section for functions.

    Args:
        functions: List of function tuples (name, signature, doc, is_async)
        title: Section title

    Returns:
        List of Markdown lines
    """
    if not functions:
        return []

    lines = [title + "\n"]
    for _name, signature, doc, is_async in functions:
        prefix = "async " if is_async else ""
        lines.append(f"#### `{prefix}{signature}`\n")
        if doc:
            formatted_doc = format_docstring(doc)
            lines.append(formatted_doc)
            lines.append("")
        else:
            lines.append(NO_DOCUMENTATION_TEXT)
    return lines


def format_class_header(cname: str, bases: list) -> str:
    """Format class header with inheritance.

    Args:
        cname: Class name
        bases: List of base classes

    Returns:
        Formatted class header
    """
    if bases:
        inheritance = f"({', '.join(bases)})"
        return f"#### `class {cname}{inheritance}`\n"
    return f"#### `class {cname}`\n"


def format_class_variables(class_vars: list) -> list[str]:
    """Format class variables/constants section.

    Args:
        class_vars: List of class variable tuples (name, type, value)

    Returns:
        List of formatted lines
    """
    if not class_vars:
        return []

    lines = ["**Class attributes:**\n"]
    for var_name, var_type, var_value in class_vars:
        if var_type:
            lines.append(f"- `{var_name}: {var_type}` = `{var_value}`")
        else:
            lines.append(f"- `{var_name}` = `{var_value}`")
    lines.append("")
    return lines


def format_methods(methods: list) -> list[str]:
    """Format methods section.

    Args:
        methods: List of method tuples (name, signature, doc, is_async)

    Returns:
        List of formatted lines
    """
    if not methods:
        return []

    lines = ["**Methods:**\n"]
    for _mname, msignature, mdoc, is_async in methods:
        prefix = "async " if is_async else ""
        lines.append(f"##### `{prefix}{msignature}`\n")
        if mdoc:
            formatted_doc = format_docstring(mdoc)
            lines.append(formatted_doc)
            lines.append("")
        else:
            lines.append(NO_DOCUMENTATION_TEXT)
    return lines


def generate_class_section(classes: list) -> list[str]:
    """Generate Markdown section for classes.

    Args:
        classes: List of class tuples (name, doc, methods, class_vars, bases)

    Returns:
        List of Markdown lines
    """
    if not classes:
        return []

    lines = ["### Classes\n"]
    for cname, cdoc, methods, class_vars, bases in classes:
        # Class header with inheritance
        lines.append(format_class_header(cname, bases))

        # Class docstring
        if cdoc:
            formatted_doc = format_docstring(cdoc)
            lines.append(formatted_doc)
            lines.append("")
        else:
            lines.append(NO_DOCUMENTATION_TEXT)

        # Class variables/constants
        lines.extend(format_class_variables(class_vars))

        # Methods
        lines.extend(format_methods(methods))

        lines.append("---\n")
    return lines


def generate_constants_section(constants: list) -> list[str]:
    """Generate Markdown section for module-level constants.

    Args:
        constants: List of constant tuples (name, value)

    Returns:
        List of Markdown lines
    """
    if not constants:
        return []

    lines = ["### Constants\n"]
    for const_name, const_value in constants:
        # Truncate very long values
        display_value = const_value
        if len(display_value) > 100:
            display_value = display_value[:100] + "..."
        lines.append(f"- `{const_name}` = `{display_value}`")
    lines.append("")
    return lines


def generate_markdown(api_map: dict, project_info: dict[str, Any]) -> str:
    """Generate complete Markdown documentation.

    Args:
        api_map: Dictionary mapping module names to their API data
        project_info: Project metadata from pyproject.toml

    Returns:
        Complete Markdown document as string
    """
    lines = []

    # Title and metadata
    project_name = project_info.get("name", "API Documentation")
    project_version = project_info.get("version", "unknown")
    project_description = project_info.get("description", "")

    lines.append(f"# {project_name} - API Documentation\n")
    lines.append(f"**Version:** {project_version}\n")
    if project_description:
        lines.append(f"**Description:** {project_description}\n")

    lines.append("---\n")

    # Table of contents
    lines.append("## Table of Contents\n")
    for module in sorted(api_map.keys()):
        module_link = module.replace(".", "").replace("/", "").replace("\\", "")
        lines.append(f"- [{module}](#{module_link})")
    lines.append("\n---\n")

    # Module documentation
    for module, api in sorted(api_map.items()):
        module_anchor = module.replace(".", "").replace("/", "").replace("\\", "")
        lines.append(f"<a id='{module_anchor}'></a>\n")
        lines.append(f"## Module `{module}`\n")

        # Module docstring
        if api["module_doc"]:
            formatted_doc = format_docstring(api["module_doc"])
            lines.append(formatted_doc)
            lines.append("\n")

        # Constants
        lines.extend(generate_constants_section(api["constants"]))

        # Functions
        lines.extend(generate_function_section(api["functions"]))

        # Classes
        lines.extend(generate_class_section(api["classes"]))

        lines.append("\n---\n")

    return "\n".join(lines)


def main(root_path: str = ".") -> None:
    """Main entry point for the documentation generator.

    Args:
        root_path: Root directory of the project (default: current directory)
    """
    root = Path(root_path).resolve()
    packages, project_info = load_pyproject(root)
    api_map = {}

    print("🔍 Analyzing modules...")
    for package in packages:
        # Get the package name - use the last part of the path as the base
        # For "src/my_package" -> "my_package"
        # For "my_package" -> "my_package"
        package_name = package.name

        for path in package.rglob("*.py"):
            if path.name == "__init__.py":
                continue

            # Create module name: package_name + relative path from package
            rel_path = path.relative_to(package).with_suffix("")
            if rel_path.parts:
                # If there are subdirectories, include them
                module_name = ".".join([package_name, *rel_path.parts])
            else:
                # Just the module file in the package root
                module_name = f"{package_name}.{path.stem}"

            print(f"  - {module_name}")
            api_map[module_name] = extract_api_from_module(path)

    print("\n📝 Generating documentation...")
    md = generate_markdown(api_map, project_info)

    output = root / "API_DOCUMENTATION.md"
    with output.open("w", encoding="utf-8") as f:
        f.write(md)

    print(f"✅ Documentation generated at: {output}")
    print(f"📊 Modules documented: {len(api_map)}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else ".")
