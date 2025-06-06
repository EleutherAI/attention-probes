from pathlib import Path
import json
import gradio as gr

def get_available_runs(config):
    cache_dir = Path("cache")
    runs = []
    for run in (cache_dir / config).glob("*"):
        html_path = run / "htmls.json"
        if html_path.exists():
            runs.append(run.stem)
    return sorted(runs)

def load_htmls(config, run_name):
    cache_dir = Path("cache")
    html_path = cache_dir / config / run_name / "htmls.json"
    if not html_path.exists():
        return None
    return json.load(open(html_path))

def show_htmls(config, run_name):
    htmls = load_htmls(config, run_name)
    if htmls is None:
        return "No htmls found for this run"
    
    html_output = [f"<h1>{run_name}</h1>"]
    html_output.extend(htmls)
    return "\n".join(html_output)

def update_runs(config):
    runs = get_available_runs(config)
    if not runs:
        return gr.Dropdown(choices=[], value=None)
    return gr.Dropdown(choices=runs, value=runs[0])

configs = ["v1", "v1-tanh"] # Add other configs as needed
with gr.Blocks(title="Attention Probe Visualization") as interface:
    gr.Markdown("# Attention Probe Visualization")
    gr.Markdown("View attention probe results for different configurations and runs")
    
    config_dropdown = gr.Dropdown(choices=configs, value="v1", label="Configuration")
    run_dropdown = gr.Dropdown(choices=get_available_runs("v1"), value=None, label="Run")
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
    interface.launch(share=True)