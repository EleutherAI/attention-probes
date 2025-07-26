
from pathlib import Path
import argparse
import html

# Configuration for the aggregation
CACHE_DIR = Path("cache")
DEFAULT_OUTPUT = Path("plots/all_activations.html")
TRIM_LINES = 100  # How many <br>-separated lines to keep from each activations.html


def get_configs():
    """Return a sorted list of configuration directory names matching the pattern used in serve.py."""
    if not CACHE_DIR.exists():
        raise FileNotFoundError(f"Cache directory '{CACHE_DIR}' not found")
    return sorted(
        d.name for d in CACHE_DIR.iterdir() if d.is_dir() and d.name.endswith("-0") and "attn-1" in d.name
    )


def get_runs(config: str):
    """Return a sorted list of runs inside a given configuration that contain an activations.html file."""
    runs = []
    for run_dir in (CACHE_DIR / config).glob("*"):
        html_path = run_dir / "activations.html"
        if html_path.exists():
            runs.append(run_dir.name)
    return sorted(runs)


def load_trimmed_html(config: str, run: str, max_lines: int = TRIM_LINES) -> str:
    """Load the activations HTML for a run and trim it to the first *max_lines* of <br>-split content."""
    path = CACHE_DIR / config / run / "activations.html"
    if not path.exists():
        return "<em>No visualization found</em>"

    raw_html = path.read_text(encoding="utf-8", errors="ignore")

    # Split on <br> and keep only the first *max_lines* chunks
    parts = raw_html.split("<br>")
    trimmed = "<br>".join(parts[:max_lines])

    return trimmed


def build_html(output_path: Path = DEFAULT_OUTPUT):
    configs = get_configs()
    if not configs:
        raise RuntimeError("No configuration directories found in cache/")

    html_sections = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<title>Attention Probe Visualizations</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; }",
        "details { margin-bottom: 1.5em; }",
        "summary { font-size: 1.1em; font-weight: bold; cursor: pointer; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Attention Probe Visualizations</h1>",
    ]

    for config in configs:
        html_sections.append(f"<h2>Configuration: {html.escape(config)}</h2>")
        runs = get_runs(config)
        if not runs:
            html_sections.append("<p><em>No runs with activations found.</em></p>")
            continue

        for run in runs:
            trimmed_html = load_trimmed_html(config, run)
            # Use <details> for collapsible runs
            html_sections.append("<details>")
            html_sections.append(f"<summary>Run: {html.escape(run)}</summary>")
            html_sections.append(trimmed_html)
            html_sections.append("</details>")

    html_sections.extend(["</body>", "</html>"])

    output_path.write_text("\n".join(html_sections), encoding="utf-8")
    print(f"Generated {output_path} containing visualizations for {len(configs)} configs.")


def main():
    global TRIM_LINES
    parser = argparse.ArgumentParser(description="Generate a single HTML file aggregating attention probe visualizations.")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="Path to the output HTML file.")
    parser.add_argument("-n", "--lines", type=int, default=TRIM_LINES, help="Number of <br>-separated lines to keep from each visualization.")
    args = parser.parse_args()

    TRIM_LINES = args.lines  # update global for load_trimmed_html

    build_html(args.output)


if __name__ == "__main__":
    main()
