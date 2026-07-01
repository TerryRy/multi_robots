import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import cos, sin, pi
from geometry import Point


AGENT_COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9a6324', '#800000', '#aaffc3',
]


class SimulatorRenderer:
    def __init__(self, img_size=224, padding=5):
        self.img_size = img_size
        self.padding = padding
        self._fig = None
        self._ax = None

    def _ensure_fig(self):
        if self._fig is None:
            dpi = 100
            figsize = self.img_size / dpi
            self._fig, self._ax = plt.subplots(
                1, 1, figsize=(figsize, figsize), dpi=dpi,
                frameon=False,
            )
            self._fig.subplots_adjust(0, 0, 1, 1)
            self._ax.set_xlim(0, self.img_size)
            self._ax.set_ylim(0, self.img_size)
            self._ax.invert_yaxis()
            self._ax.axis('off')

    def _world_to_img(self, wx, wy, env_width, env_height):
        """Convert world coordinates to image coordinates with padding."""
        usable = self.img_size - 2 * self.padding
        ix = self.padding + (wx / env_width) * usable
        iy = self.padding + (wy / env_height) * usable
        return ix, iy

    def _world_to_img_scale(self, w, env_width):
        usable = self.img_size - 2 * self.padding
        return (w / env_width) * usable

    def render(self, simulator):
        self._ensure_fig()
        self._ax.clear()
        self._ax.set_xlim(0, self.img_size)
        self._ax.set_ylim(0, self.img_size)
        self._ax.invert_yaxis()
        self._ax.axis('off')

        env_width = simulator.environment.width_in_meters
        env_height = simulator.environment.height_in_meters

        bg = patches.Rectangle((0, 0), self.img_size, self.img_size,
                               linewidth=0, facecolor='#1a1a1a')
        self._ax.add_patch(bg)

        obs_color = '#8B4513'
        if hasattr(simulator, 'environment') and hasattr(simulator.environment, 'obstacles'):
            for obs in simulator.environment.obstacles.values():
                x, y = obs.location.x, obs.location.y
                w, h = obs.dimension[0], obs.dimension[1]
                ix, iy = self._world_to_img(x, y, env_width, env_height)
                iw = self._world_to_img_scale(w, env_width)
                ih = self._world_to_img_scale(h, env_height)
                rect = patches.Rectangle(
                    (ix, iy), iw, ih,
                    linewidth=1, edgecolor='#5a2d0c', facecolor=obs_color, alpha=0.9,
                )
                self._ax.add_patch(rect)

        port_colors = {'loading': '#1a6bff', 'unloading': '#ff3333'}
        for port_list_name in ['loading_ports', 'unloading_ports']:
            port_type = 'loading' if 'loading' in port_list_name else 'unloading'
            port_list = getattr(simulator, port_list_name, [])
            for port in port_list:
                x, y = port.location.x, port.location.y
                w, h = port.dimension[0], port.dimension[1]
                ix, iy = self._world_to_img(x, y, env_width, env_height)
                iw = self._world_to_img_scale(w, env_width)
                ih = self._world_to_img_scale(h, env_height)
                rect = patches.Rectangle(
                    (ix, iy), iw, ih,
                    linewidth=1, edgecolor='white', facecolor=port_colors[port_type],
                    alpha=0.85,
                )
                self._ax.add_patch(rect)
                pid = getattr(port, 'identifier', '?')
                self._ax.text(ix + iw/2, iy + ih/2, str(pid),
                              ha='center', va='center', fontsize=4, color='white',
                              fontweight='bold')

        for agent in simulator.agents:
            body = simulator.b2_objects.get(agent.id)
            if body is None:
                continue
            pos = body.position
            angle = body.angle
            ix, iy = self._world_to_img(pos.x, pos.y, env_width, env_height)
            agent_size = self._world_to_img_scale(0.4, env_width)
            color = AGENT_COLORS[agent.id % len(AGENT_COLORS)]

            circle = patches.Circle(
                (ix, iy), agent_size,
                linewidth=1, edgecolor='white', facecolor=color, alpha=0.9,
            )
            self._ax.add_patch(circle)

            arrow_len = agent_size * 2.5
            dx = cos(angle) * arrow_len
            dy = sin(angle) * arrow_len
            self._ax.arrow(ix, iy, dx, dy,
                           head_width=agent_size*0.6, head_length=agent_size*0.8,
                           fc='white', ec='white', alpha=0.9, width=agent_size*0.2)

            self._ax.text(ix, iy - agent_size - 1.5, f'{agent.id}',
                          ha='center', va='top', fontsize=4, color='white',
                          fontweight='bold')

        buf = io.BytesIO()
        self._fig.savefig(buf, format='raw', dpi=100)
        buf.seek(0)
        w, h = self._fig.canvas.get_width_height()
        img = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape((h, w, 4))
        img = img[:, :, :3].copy()
        buf.close()
        return img

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
