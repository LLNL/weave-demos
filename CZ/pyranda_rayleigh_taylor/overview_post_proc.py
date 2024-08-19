import argparse
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler
import numpy as np


from rich.pretty import pprint

from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def setup_argparse():
    parser = argparse.ArgumentParser(
        prog="RT_Overview_Plotter",
        description="Simple plotting tools for the 'overview' rayleigh taylor studies"
    )

    parser.add_argument('workspaces', nargs='+',
                        help='Paths to workspaces containing ".dat" data files '
                        'output by the pyranda rayleigh taylor model '
                        '(rayleigh_taylor.py)')

    parser.add_argument(
        '--headless',
        action='store_true',
        help="Turn on headless mode to hide plots and only generate png's"
    )

    return parser


def main():
    parser = setup_argparse()

    args = parser.parse_args()

    if args.headless:
        import matplotlib
        matplotlib.use('Agg')

    data_sets = []
    col_name_re = re.compile(r"['](?P<column_name>[^\"^\']+)[']")
    for workspace in args.workspaces:
        record_path = Path(workspace) / 'sim_record.yaml'

        with open(record_path, 'r') as record_file:
            record = load(record_file, Loader=Loader)

            pprint(record)

        datafile_path = Path(workspace) / 'RAYLEIGH_TAYLOR_2D.dat'
        with open(datafile_path, 'r') as data_file:
            data_lines = data_file.readlines()
            data = np.loadtxt(data_lines)
            col_str = data_lines[0].strip('#').strip()

            try:
                columns = [col_match.group(1) for col_match in re.finditer(col_name_re,col_str)]
            except Exception:
                pprint(f"ERROR reading columns from '{datafile_path}'")
                pprint(f"  Header: {col_str}")
                pprint(f"  Regex: {col_name_re}")
                for col_match in re.finditer(col_name_re, col_str):
                    pprint(f'  Column match: {col_match}')
                    pprint(f'  match groups: {col_match.groups()}')
                raise

        rdata = {}
        for idx, col in enumerate(columns):
            rdata[col] = data[idx,:]
        record['data'] = rdata

        data_sets.append(record)

    plt.style.use('seaborn-v0_8')     # For non-white background
    fig = plt.figure(layout='tight', figsize=(9, 6))
    ax = plt.subplot(111)
    filter_leg_keys = ['data', 'ranvel', 'seed']
    linestyle_cycle = cycler(lstyle=['-', '--', '-.', ':'])

    vel_groups = set(rec['velmag']['value'] for rec in data_sets)
    vel_line_styles = {vg: lstyle['lstyle'] for lstyle, vg in zip(linestyle_cycle, vel_groups)}



    at_groups = set(rec['atwood']['value'] for rec in data_sets)
    cmap = plt.get_cmap(name='plasma') # 'cividis')
    vmin = 0.0                          # New Bottom percentage for cmap lookup
    vmax = 0.9                          # New top percentage for cmap lookup
    cmap_trunc = lambda val: cmap(vmin + (vmax - vmin)*val)

    # LinearSegmentedColormap.from_list(
        
    # plt.rcParams['axes.prop_cycle'] = plt.cycler("color", cmap(np.linspace(0, 1, len(list(at_groups)))))
    plt.rcParams['axes.prop_cycle'] = plt.cycler("color", cmap_trunc(np.linspace(0, 1, len(list(at_groups)))))
    color_cycle = cycler(color=plt.rcParams['axes.prop_cycle'].by_key()['color'])    
    at_color_styles = {at: color['color'] for color, at in zip(color_cycle, at_groups)}

    for record in data_sets:
        rlabel = ', '.join([f"{value['label']}={value['value']:.2g}" for key, value in record.items() if key not in filter_leg_keys])
        ax.plot(record['data']['time'], record['data']['mixing width'],
                label=rlabel,
                linestyle=vel_line_styles[record['velmag']['value']],
                lw=3,
                color=at_color_styles[record['atwood']['value']])

    # fig.legend()
    # Create separate legends for color and line style for better readability
    vel_lines = []
    vel_labels = []
    for vel_value, lstyle in vel_line_styles.items():
        pprint(f"{lstyle=}, {vel_value=}")
        vel_lines.append(Line2D([0], [0],
                                color='k',
                                lw=3,
                                linestyle=lstyle,
                                label=f'{vel_value:.1f}'))
        vel_labels.append(vel_value)

    color_lines = []
    color_labels = []
    for at_value, color in at_color_styles.items():
        color_lines.append(Line2D([0], [0],
                                  color=color,
                                  marker='o',
                                  markersize=15,
                                  linestyle=None,
                                  label=f'{at_value:.2f}'))
        color_labels.append(at_value)

    
    vel_legend = ax.legend(handles=vel_lines,
                           title='Single mode velocity magnitude',
                           bbox_to_anchor=(0.0, 1.02, 1.0, 0.1),
                           loc='lower left')
    ax.add_artist(vel_legend)
    at_legend = ax.legend(handles=color_lines,
                          title='Atwood Number',
                          bbox_to_anchor=(1.0, 1.02),
                          loc='lower right')
              
    ax.set_xlabel('Time')
    ax.set_ylabel('Mixing Width')

    fig.savefig('mixing_width_vs_time.png')

    if not args.headless:
        plt.show()


if __name__ == "__main__":
    sys.exit(main())
