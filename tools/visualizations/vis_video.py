import argparse
import os
import numpy as np

from openstl.utils import show_video_gif_multiple, show_video_gif_single


def _select_sample(arr, index):
    if arr.ndim == 5:
        return arr[index]
    if arr.ndim in (3, 4):
        return arr
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def _select_channel(arr, channel):
    if arr.ndim == 4:
        if channel < 0 or channel >= arr.shape[1]:
            raise ValueError(f"Channel index {channel} out of range for shape {arr.shape}")
        return arr[:, channel, :, :]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unsupported array shape for channel selection: {arr.shape}")


def _resolve_saved_dir(work_dir) :
    if os.path.isfile(os.path.join(work_dir, 'preds.npy')):
        return work_dir
    saved_dir = os.path.join(work_dir, 'saved')
    if os.path.isdir(saved_dir):
        return saved_dir
    raise FileNotFoundError(f"Cannot find saved dir under: {work_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='ParFlow GIF visualization (single method)')
    parser.add_argument('--work_dir', '-w', required=True, type=str,
                        help='Work directory that contains saved/inputs.npy, trues.npy, preds.npy')
    parser.add_argument('--index', '-i', default=0, type=int,
                        help='The index of a video sequence to show')
    parser.add_argument('--vis_channel', '-vc', required=True, type=int,
                        help='Select a channel to visualize')
    parser.add_argument('--save_dir', '-s', default='vis_figures', type=str,
                        help='The path to save visualization results')
    return parser.parse_args()


def main():
    args = parse_args()
    saved_dir = _resolve_saved_dir(args.work_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    inputs = np.load(os.path.join(saved_dir, 'inputs.npy'))
    trues = np.load(os.path.join(saved_dir, 'trues.npy'))
    preds = np.load(os.path.join(saved_dir, 'preds.npy'))

    inputs = _select_sample(inputs, args.index)
    trues = _select_sample(trues, args.index)
    preds = _select_sample(preds, args.index)

    inputs = _select_channel(inputs, args.vis_channel)
    trues = _select_channel(trues, args.vis_channel)
    preds = _select_channel(preds, args.vis_channel)

    suffix = f"parflow_C{args.vis_channel}_{args.index}"
    show_video_gif_single(inputs.copy(), out_path=os.path.join(args.save_dir, f'{suffix}_input'))
    show_video_gif_single(trues.copy(), out_path=os.path.join(args.save_dir, f'{suffix}_true'))
    show_video_gif_multiple(inputs, trues, preds,
                            out_path=os.path.join(args.save_dir, f'{suffix}_compare'))
    show_video_gif_single(preds, out_path=os.path.join(args.save_dir, f'{suffix}_pred'))


if __name__ == '__main__':
    main()
