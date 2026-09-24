"""Gradio interface for CreepPFN deployment inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .deployment import forecast_ensemble, forecast_figure, repository_root


def _parse_queries(text: str):
    values = [float(value.strip()) for value in text.replace(";", ",").split(",") if value.strip()]
    return np.asarray(values, dtype=float)


def create_demo(root: str | Path | None = None):
    import gradio as gr
    bundle = repository_root(root)
    figures = bundle / "assets/paper"

    def run(rho, fc, e28, stress_ratio, context, query_text, fold):
        try:
            frame = pd.DataFrame(context, columns=["t_day", "compliance"]).dropna(how="all")
            modulus = float("nan") if e28 in (None, "", 0, 0.0) else float(e28)
            ratio = float("nan") if stress_ratio in (None, "", 0, 0.0) else float(stress_ratio)
            result = forecast_ensemble(
                frame.t_day,
                frame.compliance,
                _parse_queries(query_text),
                float(rho),
                float(fc),
                modulus,
                stress_ratio=ratio,
                fold=fold,
                device="auto",
                root=root,
            )
            figure = forecast_figure(frame.t_day.to_numpy(float), frame.compliance.to_numpy(float), result)
            note = (
                f"Prediction completed with **{int(result.ensemble_members.iloc[0])} members**. "
                "Intervals are marginal at each requested time. They are not simultaneous bounds for the whole curve."
            )
            return result, figure, note
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="CreepPFN") as demo:
        gr.Markdown(
            "# CreepPFN\n"
            "Forecast concrete creep compliance and a 90% range from a short creep test. "
            "Times are elapsed days since loading. Compliance is in microstrain/MPa."
        )
        with gr.Tabs():
            with gr.Tab("Predict"):
                with gr.Row():
                    with gr.Column(scale=2):
                        rho = gr.Number(value=2400, label="Density, rho (kg/m^3)")
                        fc = gr.Number(value=40, label="Compressive strength, fc (MPa)")
                        e28 = gr.Number(value=30000, label="28-day elastic modulus, E28 (MPa; 0 = missing)")
                        stress_ratio = gr.Number(value=0.4, label="Stress ratio, applied stress / strength at loading (0 = missing)")
                        context = gr.Dataframe(
                            headers=["t_day", "compliance"],
                            datatype=["number", "number"],
                            value=[[1, 10], [3, 15], [7, 20], [10, 23]],
                            row_count=(4, "dynamic"),
                            col_count=(2, "fixed"),
                            label="Early measurements through day 10",
                        )
                        query = gr.Textbox(value="14, 28, 56, 90, 120, 160",
                                           label="Future elapsed days (comma-separated)")
                        fold = gr.Dropdown(
                            choices=["all", "fold_01", "fold_02", "fold_03", "fold_04", "fold_05"],
                            value="all",
                            label="Frozen ensemble",
                            info="all pools 15 members; one fold uses its three paper members",
                        )
                        submit = gr.Button("Forecast", variant="primary")
                    with gr.Column(scale=3):
                        plot = gr.Plot(label="Forecast")
                        status = gr.Markdown()
                output = gr.Dataframe(label="Predictions", interactive=False)
                submit.click(run, [rho, fc, e28, stress_ratio, context, query, fold], [output, plot, status])
                gr.Markdown(
                    "The model predicts the compliance increment relative to the first reading and then adds the "
                    "first reading back for the displayed compliance. Use at least two measurements through day 10. "
                    "The validated horizon ends at day 160. This research model is not a structural design check."
                )

            with gr.Tab("Model card"):
                gr.Markdown(
                    "## What the model does\n"
                    "CreepPFN is a transformer-based prior-data fitted network. It forecasts later creep compliance, "
                    "with a 90% range, from density, compressive strength, optional 28-day modulus, optional stress "
                    "ratio and the readings up to day 10. The first reading is the anchor: its compliance is subtracted "
                    "before prediction, and the predicted increment is added back for display.\n\n"
                    "## How it learns\n"
                    "In each source-disjoint fold, a hierarchical Bayesian model with an identifiable Kelvin-chain curve "
                    "family (Kelvin4) is fitted to the measured training curves. It separates material, source, specimen "
                    "and reading variation. The network is pretrained on synthetic creep tests drawn from this model "
                    "(1,230,000 in total over the 15 networks) and then fine-tuned on measured training curves. "
                    "Validation sources select settings; test sources are never used for fitting."
                )
                gr.Image(value=str(figures / "model_workflow.png"), interactive=False,
                         label="Paper Figure 1: CreepPFN flow")
                gr.Image(value=str(figures / "model_architecture.png"), interactive=False,
                         label="Paper Figure 2: network architecture")
                gr.Markdown(
                    "## Evidence\n"
                    "On 610 curves from 65 held-out NU literature sources, source-averaged NMAE was 15.94% "
                    "(95% interval 13.78-18.22%), the lowest of all models tested under the same protocol. "
                    "The 90% ranges covered 93.6% of later readings, the only model at nominal coverage. "
                    "These values describe the five fold ensembles, not the pooled 15-member option.\n\n"
                    "## Main limits\n"
                    "The model is evaluated only through day 160. It omits humidity, mixture composition, loading age, "
                    "stress history and geometry. It does not enforce monotonic output, and intervals are marginal at "
                    "each time. Recycled-aggregate concrete is rare in the training data and was poorly forecast. "
                    "The model is a research forecast tool, not a structural design check."
                )

            with gr.Tab("Paper figures"):
                for name, label, text in [
                    ("data_characteristics.png", "Data characteristics",
                     "NU, KMUTT (47) and literature (19) cohorts, modulus availability, early readings and source sizes."),
                    ("nu_results.png", "Test error and coverage",
                     "Source-averaged NMAE with 95% intervals and coverage of the 90% range for all models."),
                    ("forecast_examples.png", "Forecast examples",
                     "Black points are model inputs, orange crosses later measurements, the line the forecast and "
                     "shading the 90% range."),
                    ("test_duration.png", "Effect of test duration",
                     "Forecast error and coverage for 3 to 28 days of readings."),
                    ("shap_descriptors.png", "SHAP contributions of the material descriptors",
                     "Contributions with each specimen's early readings held fixed; model behaviour, not causal effects."),
                ]:
                    gr.Markdown(f"## {label}")
                    gr.Image(value=str(figures / name), interactive=False, label=label)
                    gr.Markdown(text)
    return demo


def main():
    create_demo().launch()


if __name__ == "__main__":
    main()
