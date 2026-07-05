import numpy as np
from math import cos, sin, pi


AGENT_COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (67, 99, 216),
    (245, 130, 49), (145, 30, 180), (66, 212, 244), (240, 50, 230),
    (191, 239, 69), (250, 190, 212), (70, 153, 144), (220, 190, 255),
]


class SimulatorRenderer:
    def __init__(self, img_size=224, padding=5):
        self.img_size = img_size
        self.padding = padding

    def _world_to_img(self, wx, wy, env_width, env_height):
        usable = self.img_size - 2 * self.padding
        ix = self.padding + (wx / env_width) * usable
        iy = self.padding + (wy / env_height) * usable
        return int(ix), int(iy)

    def _world_to_img_scale(self, w, env_width):
        usable = self.img_size - 2 * self.padding
        return max(1, int((w / env_width) * usable))

    def render(self, simulator):
        img = np.full((self.img_size, self.img_size, 3), 26, dtype=np.uint8)
        env_width = simulator.environment.width_in_meters
        env_height = simulator.environment.height_in_meters

        obstacles = getattr(simulator.environment, 'obstacles', {})
        if obstacles:
            for obs in obstacles.values():
                x, y = obs.location.x, obs.location.y
                w, h = obs.dimension[0], obs.dimension[1]
                ix, iy = self._world_to_img(x, y, env_width, env_height)
                iw = self._world_to_img_scale(w, env_width)
                ih = self._world_to_img_scale(h, env_height)
                img[iy:iy+ih, ix:ix+iw] = (139, 69, 19)

        port_color_map = {'loading': (30, 107, 255), 'unloading': (255, 51, 51)}
        for port_list_name in ['loading_ports', 'unloading_ports']:
            port_type = 'loading' if 'loading' in port_list_name else 'unloading'
            port_list = getattr(simulator, port_list_name, [])
            for port in port_list:
                x, y = port.location.x, port.location.y
                w, h = port.dimension[0], port.dimension[1]
                ix, iy = self._world_to_img(x, y, env_width, env_height)
                iw = self._world_to_img_scale(w, env_width)
                ih = self._world_to_img_scale(h, env_height)
                color = port_color_map[port_type]
                img[iy:iy+ih, ix:ix+iw] = color
                cx, cy = ix + iw // 2, iy + ih // 2
                pid = str(getattr(port, 'identifier', '?'))
                self._put_text(img, pid, cx, cy, (255, 255, 255), 4)

        for agent in simulator.agents:
            body = simulator.b2_objects.get(agent.id)
            if body is None:
                continue
            pos = body.position
            angle = body.angle
            ix, iy = self._world_to_img(pos.x, pos.y, env_width, env_height)
            r = max(3, self._world_to_img_scale(0.35, env_width))
            color = AGENT_COLORS[agent.id % len(AGENT_COLORS)]
            self._fill_circle(img, ix, iy, r, color)
            ax = ix + int(cos(angle) * r * 2.5)
            ay = iy + int(sin(angle) * r * 2.5)
            self._draw_line(img, ix, iy, ax, ay, (255, 255, 255), 2)
            self._put_text(img, str(agent.id), ix, iy - r - 2, (255, 255, 255), 4)

        return img

    def _fill_circle(self, img, cx, cy, r, color):
        ys, xs = np.ogrid[:self.img_size, :self.img_size]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
        img[mask] = color

    def _draw_line(self, img, x1, y1, x2, y2, color, thickness=1):
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        steps = max(dx, dy) or 1
        for t in np.linspace(0, 1, steps + 1):
            px = int(round(x1 + t * (x2 - x1)))
            py = int(round(y1 + t * (y2 - y1)))
            if 0 <= px < self.img_size and 0 <= py < self.img_size:
                img[py, px] = color

    def _put_text(self, img, text, x, y, color, size=4):
        cx, cy = x, y
        offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        for i, ch in enumerate(text):
            bx = cx + i * (size + 1)
            for dx in range(size):
                for dy in range(size):
                    px, py = bx + dx, cy + dy
                    if 0 <= px < self.img_size and 0 <= py < self.img_size:
                        img[py, px] = color

    def close(self):
        pass
