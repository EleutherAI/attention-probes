from pathlib import Path
import json
import gradio as gr
import os

def get_available_runs(config):
    cache_dir = Path("cache")
    runs = []
    for run in (cache_dir / config).glob("*"):
        html_path = run / "activations.html"
        if html_path.exists():
            runs.append(run.stem)
    return sorted(runs)

def load_html(config, run_name):
    cache_dir = Path("cache")
    if run_name is None:
        return None
    html_path = cache_dir / config / run_name / "activations.html"
    if not html_path.exists():
        return None
    return html_path.read_text()

def show_htmls(config, run_name):
    html = load_html(config, run_name)
    if html is None:
        return "No visualization found for this run"
    
    html = "<br>".join(html.split("<br>")[:100])
    
    return f"<h1>{run_name}</h1>\n{html}"

def update_runs(config):
    runs = get_available_runs(config)
    if not runs:
        return gr.Dropdown(choices=[], value=None)
    return gr.Dropdown(choices=runs, value=runs[0])

configs = [c for c in os.listdir("cache") if c.endswith("-0") and "attn-1" in c]
with gr.Blocks(title="Attention Probe Visualization") as demo:
    gr.Markdown("# Attention Probe Visualization")
    gr.Markdown("View attention probe results for different configurations and runs")
    
    config_dropdown = gr.Dropdown(choices=configs, value=configs[0], label="Configuration")
    run_dropdown = gr.Dropdown(choices=get_available_runs(configs[0]), value=None, label="Run")
    output = gr.HTML()
    
    config_dropdown.change(
        fn=update_runs,
        inputs=[config_dropdown],
        outputs=[run_dropdown]
    )
    
    run_dropdown.change(
        fn=show_htmls,
        inputs=[config_dropdown, run_dropdown],
        outputs=[output]
    )

if __name__ == "__main__":
    demo.launch(share=True)