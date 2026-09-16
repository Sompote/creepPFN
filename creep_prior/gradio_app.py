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

    def run(rho, fc, e28, context, query_text, fold):
        try:
            frame = pd.DataFrame(context, columns=["t_day", "compliance"]).dropna(how="all")
            modulus = float("nan") if e28 in (None, "", 0, 0.0) else float(e28)
            result = forecast_ensemble(
                frame.t_day,
                frame.compliance,
                _parse_queries(query_text),
                float(rho),
                float(fc),
                modulus,
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
            "Forecast concrete compliance after an early measurement history. "
            "Times are elapsed days since loading. Compliance is in microstrain/MPa."
        )
        with gr.Tabs():
            with gr.Tab("Predict"):
                with gr.Row():
                    with gr.Column(scale=2):
                        rho = gr.Number(value=2400, label="Density, rho (kg/m^3)")
                        fc = gr.Number(value=40, label="Compressive strength, fc (MPa)")
                        e28 = gr.Number(value=30000, label="28-day elastic modulus, E28 (MPa; 0 = missing)")
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
                submit.click(run, [rho, fc, e28, context, query, fold], [output, plot, status])
                gr.Markdown(
                    "The model predicts the compliance increment relative to the first reading and then adds the "
                    "first reading back for the displayed compliance. Use at least two measurements through day 10. "
                    "The validated horizon ends at day 160. This research model is not a structural design check."
                )

            with gr.Tab("Model card"):
                gr.Markdown(
                    "## What the model does\n"
                    "CreepPFN is a material-conditioned prior-data fitted network. It forecasts later concrete "
                    "creep compliance from density, compressive strength, optional 28-day modulus, and an irregular "
                    "early measurement history. The first reading is the anchor: its compliance is subtracted before "
                    "prediction, and the predicted increment is added back for display.\n\n"
                    "## How it learns\n"
                    "Each source-disjoint training fold fits a material-linked generator to real training curves. "
                    "The generator samples plausible curve shapes and irregular schedules for synthetic pretraining. "
                    "The network is then fine-tuned on measured training curves. Validation sources select checkpoints. "
                    "Test sources do not calibrate the generator, scaler, imputation, or network."
                )
                gr.Image(value=str(figures / "model_workflow.png"), interactive=False,
                         label="Paper Figure 1: CreepPFN workflow")
                gr.Markdown(
                    "The frozen network encodes early time-response pairs, adds the material representation, and "
                    "compares each future query with the encoded history. Three independently trained members form "
                    "each paper ensemble."
                )
                gr.Image(value=str(figures / "model_architecture.png"), interactive=False,
                         label="Attention architecture")
                gr.Markdown(
                    "## Evidence\n"
                    "The source-disjoint NU evaluation included 610 eligible curves from 65 literature sources and "
                    "5,646 future readings. Source-macro NMAE was 17.40% (95% source-bootstrap interval 14.96–20.14%). "
                    "Aggregate coverage of the nominal 90% marginal interval was 94.51%. These values describe the "
                    "five evaluated fold ensembles. They are not a validation result for the pooled 15-member option.\n\n"
                    "## Main limits\n"
                    "The model is evaluated only through day 160. It omits humidity, mixture composition, loading age, "
                    "stress history, and geometry as explicit application inputs. It does not enforce monotonic output. "
                    "Intervals are marginal at each time. The model is a research forecast tool, not a structural design check."
                )

            with gr.Tab("Paper figures"):
                gr.Markdown("## Data characteristics")
                gr.Image(value=str(figures / "data_characteristics.png"), interactive=False,
                         label="Paper Figure 2")
                gr.Markdown(
                    "The panels compare cohort distributions, modulus availability, early-reading counts, and source sizes. "
                    "Unequal source sizes motivate source-level evaluation."
                )
                gr.Markdown("## NU prediction results")
                gr.Image(value=str(figures / "nu_results.png"), interactive=False,
                         label="Paper Figure 3")
                gr.Markdown(
                    "Points show source-macro normalized mean absolute error (NMAE); whiskers are source-bootstrap intervals."
                )
                gr.Markdown("## Architecture ablations")
                gr.Image(value=str(figures / "architecture_ablation.png"), interactive=False,
                         label="Paper Figure 4")
                gr.Markdown(
                    "Positive differences indicate higher error after removing a component. Removing both attention paths "
                    "or the query-gap input produced intervals above zero under the tested design."
                )
                gr.Markdown("## Grouped SHAP attribution")
                gr.Image(value=str(figures / "shap_attribution.png"), interactive=False,
                         label="Grouped SHAP analysis")
                gr.Markdown(
                    "Early measurement history plus its anchor has the largest attribution magnitude. These values "
                    "explain fitted predictions relative to selected backgrounds; they are not causal material effects."
                )
                gr.Markdown("## Forecast examples")
                gr.Image(value=str(figures / "forecast_examples.png"), interactive=False,
                         label="NU and external forecast examples")
                gr.Markdown(
                    "Black points are observed context, orange crosses are later measurements, the blue line is the "
                    "ensemble forecast, and shading is the central 90% marginal interval."
                )
    return demo


def main():
    create_demo().launch()


if __name__ == "__main__":
    main()
