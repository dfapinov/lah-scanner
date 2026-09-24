import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent

from viewers import CloudBrowser3DView, FrequencyBrowser3DView


CFG = {
    'x_bounds': (-100.0, 100.0),
    'y_bounds': (-100.0, 100.0),
    'z_bounds': (-100.0, 100.0),
}


def test_stage2_viewers_use_roll_free_turntable_rotation():
    viewers = [FrequencyBrowser3DView(CFG), CloudBrowser3DView(CFG)]
    try:
        assert plt.rcParams['axes3d.mouserotationstyle'] == 'azel'
        for viewer in viewers:
            assert viewer.ax.roll == 0
            assert viewer.ax._axis_names[viewer.ax._vertical_axis] == 'z'
    finally:
        for viewer in viewers:
            plt.close(viewer.fig)


def test_stage2_cloud_viewer_does_not_lock_elevation():
    viewer = CloudBrowser3DView(CFG)
    try:
        viewer.ax.view_init(elev=55, azim=20, roll=0, vertical_axis='z')
        event = MouseEvent('motion_notify_event', viewer.fig.canvas, 0, 0)
        viewer.fig.canvas.callbacks.process('motion_notify_event', event)
        assert viewer.ax.elev == 55
        assert viewer.ax.azim == 20
        assert viewer.ax.roll == 0
    finally:
        plt.close(viewer.fig)
