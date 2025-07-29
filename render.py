
from pathlib import Path
import argparse
from natsort import natsorted

# Configuration for the aggregation
CACHE_DIR = Path("cache")
DEFAULT_OUTPUT = Path("plots/all_activations.html")
TRIM_LINES = 100  # How many <br>-separated lines to keep from each activations.html
dataset_descriptions = {
    "Anthropic_election_questions": "<a href='https://huggingface.co/datasets/Anthropic/election_questions' target='_blank'>Election Questions</a>",
    "fancyzhx_ag_news": "<a href='https://huggingface.co/datasets/fancyzhx/ag_news' target='_blank'>AG News</a>",
    "jackhhao_jailbreak-classification": "<a href='https://huggingface.co/datasets/jackhhao/jailbreak-classification' target='_blank'>Jailbreak Classification</a>",
    "willcb_massive-scenario": "<a href='https://huggingface.co/datasets/willcb/massive-scenario' target='_blank'>Massive Scenario</a>",
    "canrager_amazon_reviews_mcauley_1and5": "<a href='https://huggingface.co/datasets/canrager/amazon_reviews_mcauley_1and5' target='_blank'>Amazon Reviews</a>",
    "LabHC_bias_in_bios": "<a href='https://huggingface.co/datasets/LabHC/bias_in_bios' target='_blank'>Bias in Bios</a>",
    "AIM-Harvard_reject_prompts": "<a href='https://huggingface.co/datasets/AIM-Harvard/reject_prompts' target='_blank'>Reject Prompts</a>",
}

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
        # html_sections.append(f"<h2>Configuration: {html.escape(config)}</h2>")
        runs = get_runs(config)
        if not runs:
            html_sections.append("<p><em>No runs with activations found.</em></p>")
            continue

        names, htmls = [], []

        for run in runs:
            if run.startswith("google-gemma-2b-"):
                model_name = "Gemma 2B"
            elif run.startswith("google-gemma-2-2b-"):
                model_name = "Gemma 2 2B"
            # hack
            layer_idx = int(run.partition("2b-")[2].partition("-")[0])
            
            haystack = False
            dataset_name = None
            if "gurnee_data" in run:
                haystack = True
                dataset_name = run.partition("._data_gurnee_data_processed_")[2]
            else:
                haystack = False
                dataset_name = run.partition("-16k-")[2]
            
            trimmed_html = load_trimmed_html(config, run)
            descriptives = [
                f"Dataset: {'(Neurons in A Haystack) ' if haystack else ''}{dataset_descriptions.get(dataset_name, dataset_name)}",
                f"Model: {model_name}",
                f"Layer: {layer_idx}",
            ]
            name = ", ".join(descriptives)
            names.append(name)
            htmls.append(trimmed_html)
        for name, html in natsorted(zip(names, htmls)):
            # Use <details> for collapsible runs
            html_sections.append("<details>")
            html_sections.append(f"<summary>{name}</summary>")
            html_sections.append(html)
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
