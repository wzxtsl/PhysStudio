"""Compare phase increments and consecutive-frame changes between two trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np


def phase_stats(folder):
    rows = json.loads((folder / 'audit.json').read_text())
    cfg = json.loads((folder / 'config.json').read_text())
    assert len(rows) == cfg['frames']
    fraction = np.array([r['ice_fraction'] for r in rows])
    time = np.array([r['time'] for r in rows])
    changes = np.diff(fraction)
    peak = int(np.argmax(changes)) + 1
    result = dict(simulation=str(folder.resolve()), frames=len(rows),
                  playback_fps=cfg['playback_fps'], mechanical_seconds=cfg['seconds'],
                  final_solid_fraction=float(fraction[-1]),
                  peak_increment_percentage_points=float(changes.max() * 100),
                  peak_increment_frame=peak,
                  peak_phase_rate_per_mechanical_second=float(np.max(changes / np.diff(time))),
                  peak_phase_rate_per_playback_second=float(changes.max() * cfg['playback_fps']))
    return result, fraction, time


def image_stats(folder, fraction):
    from PIL import Image
    files = sorted(folder.glob('*.png'))
    assert len(files) == len(fraction), (folder, len(files), len(fraction))
    assert [f.stem for f in files] == [f'{i:05d}' for i in range(len(fraction))]
    previous = None
    changes = []
    for file in files:
        with Image.open(file) as image:
            frame = np.asarray(image.convert('RGB').resize((320, 200), Image.Resampling.LANCZOS),
                               np.float32) / 255.
        if previous is not None:
            changes.append(float(np.abs(frame - previous).mean()))
        previous = frame
    changes = np.asarray(changes)
    active = (fraction[1:] > .005) & (fraction[:-1] < .995)
    indices = np.flatnonzero(active)
    peak = int(indices[np.argmax(changes[active])])
    return dict(render_frames=str(folder.resolve()), comparison_resolution=[320, 200],
                active_solid_fraction_range=[.005, .995],
                mean_rgb_difference_during_freezing=float(changes[active].mean()),
                maximum_rgb_difference_during_freezing=float(changes[peak]),
                maximum_rgb_difference_frame=peak + 1), changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--before-frames', type=Path)
    parser.add_argument('--after-frames', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.before_frames) != bool(args.after_frames):
        parser.error('Supply both rendered-frame directories together')
    args.output.mkdir(parents=True, exist_ok=False)
    report, curves = {}, {}
    for label, simulation, frames in [('before', args.before, args.before_frames),
                                      ('after', args.after, args.after_frames)]:
        stats, fraction, time = phase_stats(simulation)
        curves[label] = dict(fraction=fraction, time=time, fps=stats['playback_fps'])
        if frames:
            image, difference = image_stats(frames, fraction)
            stats.update(image); curves[label]['image_difference'] = difference
        report[label] = stats
    report['peak_phase_increment_reduction_percent'] = 100 * (
        1 - report['after']['peak_increment_percentage_points']
        / report['before']['peak_increment_percentage_points'])
    report['peak_mechanical_phase_rate_reduction_percent'] = 100 * (
        1 - report['after']['peak_phase_rate_per_mechanical_second']
        / report['before']['peak_phase_rate_per_mechanical_second'])
    if args.before_frames:
        report['image_metric_description'] = (
            'Consecutive RGB mean absolute difference includes camera motion, fluid motion, '
            'and sampling noise. It is a temporal-change diagnostic, not a physical accuracy metric.')
    (args.output / 'comparison.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(args.output / 'curves.npz', **{
        f'{label}_{key}': value for label, row in curves.items() for key, value in row.items()})
    print(json.dumps(report, indent=2))
    try:
        import matplotlib
    except ImportError:
        return
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    count = 3 if args.before_frames else 2
    fig, axes = plt.subplots(1, count, figsize=(5 * count, 3.5), layout='constrained')
    for (label, row), color in zip(curves.items(), ['#777777', '#15977f']):
        f, fps = row['fraction'], row['fps']
        t = np.arange(len(f)) / fps
        axes[0].plot(t, 100 * f, color=color, label=label.title())
        axes[1].plot(t[1:], 100 * np.diff(f), color=color, label=label.title())
        if count == 3:
            axes[2].plot(t[1:], row['image_difference'], color=color, label=label.title())
    axes[0].set_ylabel('Solid fraction (%)')
    axes[1].set_ylabel('Solid fraction increment (percentage points / frame)')
    if count == 3:
        axes[2].set_ylabel('Consecutive-frame RGB absolute difference')
    for ax in axes:
        ax.set_xlabel('Playback time (s)'); ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.savefig(args.output / 'comparison.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
