import matplotlib.pyplot as plt

from viewers import Stage5Viewer


def test_stage5_viewer_uses_roll_free_turntable_rotation():
    viewer = Stage5Viewer()
    try:
        assert plt.rcParams['axes3d.mouserotationstyle'] == 'azel'
        assert viewer.ax.roll == 0
        assert viewer.ax._axis_names[viewer.ax._vertical_axis] == 'z'
    finally:
        plt.close(viewer.fig)
