"""Audit cached trajectories, phase budgets and visible mixing statistics."""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(run):
    cfg = json.loads((run / "config.json").read_text())
    records = json.loads((run / "diagnostics.json").read_text())
    nf = cfg["fluid_particles"]
    rho = np.asarray(cfg["densities"])
    vol = cfg["dx"] ** 3
    assert len(records) == cfg["frames"], "Incomplete trajectory"
    initial = None
    rows = []
    for index in range(cfg["frames"]):
        with np.load(run / "states" / f"{index:05d}.npz") as data:
            assert all(np.isfinite(data[k]).all() for k in data.files)
            a = data["fractions"].astype(float)
            x = data["positions"].astype(float)
            mass = a.sum(0) * vol * rho
            if initial is None:
                initial = mass.copy()
            if cfg.get("tea_bag"):
                assert (
                    "bag" in data and "rod" in data
                ), "Moving geometry must be cached with the fluid"
            center = (a[:nf].T @ x[:nf]) / np.maximum(a[:nf].sum(0)[:, None], 1.0e-30)
            row = dict(
                frame=index,
                time=index / cfg["fps"],
                phase_mass=mass.tolist(),
                phase_centers=center.tolist(),
                partition_error=float(abs(a.sum(1) - 1).max()),
                phase_min=float(a.min()),
                phase_max=float(a.max()),
                max_relative_mass_error=float(
                    np.max(abs(mass - initial) / np.maximum(initial, 1.0e-9))
                ),
            )
            if cfg["scene"] == "tea":
                c = a[:nf, 1]
                mu = c.mean()
                row.update(
                    tea_in_water=float(c.sum() * vol * rho[1]),
                    tea_in_bag=float(a[nf:, 1].sum() * vol * rho[1]),
                    concentration_cv=float(c.std() / max(mu, 1.0e-12)),
                )
            rows.append(row)
    stats = dict(
        frames=len(rows),
        fluid_particles=nf,
        max_relative_phase_mass_error=max(r["max_relative_mass_error"] for r in rows),
        max_partition_error=max(r["partition_error"] for r in rows),
        min_fraction=min(r["phase_min"] for r in rows),
        max_fraction=max(r["phase_max"] for r in rows),
        maximum_contact_correction=max(r["contact_correction"] for r in records),
        final_phase_centers=rows[-1]["phase_centers"],
        initial_phase_centers=rows[0]["phase_centers"],
    )
    assert stats["max_relative_phase_mass_error"] < 5.0e-4, stats
    assert stats["max_partition_error"] < 5.0e-4, stats
    assert stats["min_fraction"] >= -1.0e-5 and stats["max_fraction"] < 1.0005, stats
    if cfg["scene"] == "tea":
        removed = [r for r in rows if r["time"] >= 4.6]
        if removed:
            stats["tea_mass_after_bag_removed_drift"] = max(
                r["tea_in_water"] for r in removed
            ) - min(r["tea_in_water"] for r in removed)
        stats["final_tea_in_water"] = rows[-1]["tea_in_water"]
        stats["final_tea_in_bag"] = rows[-1]["tea_in_bag"]
        stats["concentration_cv_before_stirring"] = min(
            rows, key=lambda r: abs(r["time"] - 5.5)
        )["concentration_cv"]
        stats["concentration_cv_final"] = rows[-1]["concentration_cv"]
    output = dict(scene=cfg["scene"], passed=True, summary=stats, frames=rows)
    (run / "audit.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(stats, indent=2))
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    audit(p.parse_args().run)
