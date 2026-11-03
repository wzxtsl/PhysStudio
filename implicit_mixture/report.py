"""Summarize the three demos and compare tea stirring against diffusion alone."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .audit import audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["tea", "control", "separation", "collision", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name in ["tea", "control", "separation", "collision"]:
        run = getattr(args, name)
        path = run / "audit.json"
        reports[name] = json.loads(path.read_text()) if path.exists() else audit(run)

    def speeds(run):
        cfg = json.loads((run / "config.json").read_text())
        nf = cfg["fluid_particles"]
        result = []
        for i in range(cfg["frames"]):
            with np.load(run / "states" / f"{i:05d}.npz") as state:
                v = state["velocities"][:nf].astype(float)
                result.append(float(np.sqrt(np.mean(np.sum(v * v, axis=1)))))
        return np.array(result)

    tea_speed = speeds(args.tea)
    control_speed = speeds(args.control)
    tea = reports["tea"]
    control = reports["control"]
    a = tea["summary"]
    b = control["summary"]
    result = dict(
        tea=a,
        separation=reports["separation"]["summary"],
        collision=reports["collision"]["summary"],
        stirring_control=dict(
            final_cv_with_rod=a["concentration_cv_final"],
            final_cv_without_rod=b["concentration_cv_final"],
            cv_reduction_fraction=1
            - a["concentration_cv_final"] / b["concentration_cv_final"],
            final_rms_speed_with_rod=float(tea_speed[-1]),
            final_rms_speed_without_rod=float(control_speed[-1]),
            final_rms_speed_ratio=float(tea_speed[-1] / control_speed[-1]),
        ),
    )
    (args.output / "validation.json").write_text(json.dumps(result, indent=2))
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.7), layout="constrained")
    times = [r["time"] for r in tea["frames"]]
    for report, label, color in [
        (tea, "With glass rod", "#aa531b"),
        (control, "Diffusion only", "#58666e"),
    ]:
        axes[0].plot(
            times,
            [r["concentration_cv"] for r in report["frames"]],
            label=label,
            color=color,
            lw=2,
        )
    axes[0].set(
        xlim=(0.3, 10),
        ylim=(0, 6),
        xlabel="Time (s)",
        ylabel="Concentration coefficient of variation",
        title="Tea becomes more uniform",
    )
    axes[0].axvspan(5.5, 10, color="#ba9061", alpha=0.10)
    axes[0].legend(frameon=False)
    axes[1].plot(times, tea_speed, color="#aa531b", label="With glass rod")
    axes[1].plot(times, control_speed, color="#58666e", label="Diffusion only")
    axes[1].set(
        xlabel="Time (s)",
        ylabel="RMS mixture speed",
        title="Physical stirring response",
    )
    axes[1].axvspan(5.5, 10, color="#ba9061", alpha=0.10)
    rows = reports["separation"]["frames"]
    times = [r["time"] for r in rows]
    for k, (label, color) in enumerate(
        [("Light", "#cb512a"), ("Medium", "#b59448"), ("Heavy", "#225b7e")]
    ):
        axes[2].plot(
            times,
            [r["phase_centers"][k][2] for r in rows],
            label=label,
            color=color,
            lw=2,
        )
    axes[2].set(
        xlabel="Time (s)", ylabel="Phase height centroid", title="Density inversion"
    )
    axes[2].legend(frameon=False)
    fig.savefig(args.output / "diagnostics.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result["stirring_control"], indent=2))


if __name__ == "__main__":
    main()
