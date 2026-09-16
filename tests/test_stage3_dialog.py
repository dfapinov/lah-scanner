"""Exercise the actual Tk/Matplotlib dialog in an isolated, hidden process."""
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('tail_only', [True, False])
@pytest.mark.parametrize('with_spl', [True, False])
@pytest.mark.parametrize('unresolved', [True, False])
def test_stage3_dialog_reference_change_and_send(tail_only, with_spl, unresolved):
    script = r'''
import sys
sys.path.insert(0, 'src')
import tkinter as tk
from tkinter import ttk
import hals_post_ui_core as ui
import stage3_optimize_she_settings as stage3

tail_only = sys.argv[1] == 'True'
with_spl = sys.argv[2] == 'True'
root = tk.Tk()
root.withdraw()
errors = []
root.report_callback_exception = lambda *args: errors.append(args)
original = tk.Toplevel
class HiddenTop(original):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.withdraw()
tk.Toplevel = HiddenTop
root.stage4_vars = {'target_n_max': tk.StringVar(value='99')}
orders = [2, 3, 4, 5]
ratios = [10, 9, 8, 7] if tail_only else [25, 24, 23, 22]
power = [-19, -22, -27, float('-inf')]
options = stage3.stage3_order_choices(orders, ratios, [1]*4, None, power, 5, tail_only=tail_only)
if tail_only:
    options = {'spl':dict(n=3,ratio=9,err=1,tail_db=float('nan'),spl_db=1,label='SPL plateau')}
result = dict(options=options, recommended_key=stage3.recommended_stage3_choice(options),
              tail_only=tail_only, mode_note='Tail-only test', step1=dict(
    orders=orders, ratios=ratios, residuals=[1]*4, internal_tail_power_db=power,
    tail_reference=dict(n=5, ratio=ratios[-1], fallback=False, tail_only=tail_only),
    tail_by_reference={
        '5': dict(n=5, ratio=ratios[-1], power_db=power),
        '4': dict(n=4, ratio=ratios[-2], power_db=[-21,-26,float('-inf'),float('nan')])}))
if sys.argv[3] == 'True':
    result['options'] = {}
    result['recommended_key'] = None
    result['step1']['tail_reference']['n'] = None
    result['step1']['internal_tail_power_db'] = [float('nan')]*4
if with_spl:
    result['spl_change'] = dict(orders=orders, max_change_db=[3, 1, .5, 0],
        p99_change_db=[2.5, .8, .4, 0],
        radius_m=1, sphere_points=1000, floor_db=-40,
        valid_frequency_counts=[2]*4, added_degree_frequency_counts=[2,2,2,0])
if with_spl and not tail_only and sys.argv[3] != 'True':
    result['options']['knee'] = dict(n=3, ratio=24, tail_db=-22, label='Roll-off knee')
    result['options']['spl'] = dict(n=4, ratio=23, tail_db=float('nan'), spl_db=.4, label='Directivity change')
    result['recommended_key'] = 'knee'
    result['step1']['rolloff_knee'] = {'n': 3}
try:
    ui.SpkrScannerApp._show_stage3_choice_popup(root, result)
    root.update()
    figures = [ui.plt.figure(n) for n in ui.plt.get_fignums()]
    figure = next(fig for fig in figures if fig.axes[0].get_ylabel() == 'Int/Ext ratio (dB)')
    assert len(figures) == 1 + int(with_spl)
    assert len(figure.axes) == 2
    if tail_only:
        for ax in figure.axes[:2]:
            assert any('Unavailable for order selection' in text.get_text() for text in ax.texts)
    assert all(ax.get_figure() is figure for ax in figure.axes)
    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)
    widgets = list(descendants(root))
    if with_spl:
        spl_figure = next(fig for fig in figures if fig is not figure)
        assert len(spl_figure.axes[0].lines) == 1
        assert spl_figure.axes[0].get_title() == 'Change to Directivity Pattern'
        assert spl_figure.axes[0].get_xlabel() == figure.axes[0].get_xlabel()
        assert figure.axes[1].get_xlabel().startswith(figure.axes[0].get_xlabel())
        assert not any(isinstance(w, ttk.Notebook) for w in widgets)
    if not tail_only:
        combo = next(w for w in widgets if isinstance(w, ttk.Combobox))
        combo.current(1)
        combo.event_generate('<<ComboboxSelected>>')
        root.update()
        assert figure.axes[0].get_title() == 'Solve Stability'
        assert figure.axes[1].get_title() == 'Sound Power Discarded'
        assert 'Reference N=4' in figure.axes[1].get_xlabel()
        assert figure.axes[1].get_xlim() == figure.axes[0].get_xlim()
        assert not any(line.get_linestyle() == ':' for line in figure.axes[1].lines)
    else:
        assert not any(isinstance(w, ttk.Combobox) for w in widgets)
    if with_spl and not tail_only and sys.argv[3] != 'True':
        radios = [w for w in descendants(root) if isinstance(w, ttk.Radiobutton)]
        assert {w.cget('value') for w in radios} == {'knee', 'tail', 'spl'}
        assert any('Recommended by Directivity Change' in str(w.cget('text')) for w in descendants(root) if isinstance(w, ttk.Label))
        assert spl_figure.axes[0].collections
    assert not errors, errors
    button = next(w for w in widgets if isinstance(w, ttk.Button) and w.cget('text') == 'Use in Stage 4')
    if tail_only and sys.argv[3] == 'True':
        assert str(button.cget('state')) == 'disabled'
        next(w for w in widgets if isinstance(w, ttk.Button) and w.cget('text') == 'Cancel').invoke()
        assert root.stage4_vars['target_n_max'].get() == '99'
    else:
        button.invoke()
        assert root.stage4_vars['target_n_max'].get() == '3'
    root.update()
    assert not ui.plt.get_fignums()
    assert not errors, errors
finally:
    ui.plt.close('all')
    root.destroy()
'''
    result = subprocess.run([sys.executable, '-c', script, str(tail_only), str(with_spl), str(unresolved)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True,
                            text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
