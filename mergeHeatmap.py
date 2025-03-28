from manim import *
import numpy as np

def slope_to_color(slope, vmin=-1, vmax=1):
    if np.isnan(slope):
        return BLACK
    slope_clamped = max(min(slope, vmax), vmin)
    alpha = (slope_clamped - vmin) / (vmax - vmin)
    if alpha < 0.5:
        return interpolate_color(BLUE, WHITE, alpha / 0.5)
    else:
        return interpolate_color(WHITE, RED, (alpha - 0.5) / 0.5)

def create_triangular_heatmap(data, cell_size=0.15):
    n = len(data)
    x_vals = np.arange(n)
    heatmap = VGroup()

    for j in range(n):
        for i in range(j):
            x_window = x_vals[i:j+1]
            y_window = data[i:j+1]
            if len(x_window) < 2:
                continue
            try:
                slope, _ = np.polyfit(x_window, y_window, 1)
            except np.linalg.LinAlgError:
                slope = 0

            color = slope_to_color(slope)
            square = Square(side_length=cell_size)
            square.set_fill(color, opacity=1)
            square.set_stroke(width=0)

            pos = np.array([i + 0.5, -(j + 0.5), 0]) * cell_size
            square.move_to(pos)
            heatmap.add(square)

    return heatmap

class MergeHeatmaps(MovingCameraScene):
    def construct(self):
        n = 6 # trials per heatmap
        n_maps = 30
        cell_size = 0.15
        rng = np.random.default_rng(42)

        # Step 1: Create 30 heatmaps
        heatmaps = VGroup()
        for _ in range(n_maps):
            y_vals = np.sin(0.3 * np.arange(n)) + 2 * rng.normal(size=n)
            hm = create_triangular_heatmap(y_vals, cell_size=cell_size)
            heatmaps.add(hm)

        # Step 2: Tile them in a 3x10 grid
        grid = VGroup(*heatmaps).arrange_in_grid(
            rows=3, cols=10, buff=0.8, align_rows=True
        ).move_to(ORIGIN)
        # Create an array of delays linearly decreasing
        delays = np.linspace(0.1, 0.02, n_maps)
        for hm, delay in zip(heatmaps, delays):
            self.play(FadeIn(hm, run_time=delay))
            self.wait(delay)
        self.wait(1)

        # Step 3: Create a large composite heatmap
        merged_y_vals = np.mean([
            np.sin(0.3 * np.arange(n)) + 2 * rng.normal(size=n)
            for _ in range(n_maps)
        ], axis=0)
        merged_heatmap = create_triangular_heatmap(merged_y_vals, cell_size=0.5)
        merged_heatmap.scale_to_fit_width(config.frame_width / 2 - 1)
        merged_heatmap.to_edge(LEFT, buff=0.5)

        # Step 4: Transfrom the merged map to the big heatmap
        n_big = 50
        big_y_vals = np.sin(0.3 * np.arange(n_big)) + 2 * rng.normal(size=n_big)
        big_heatmap = create_triangular_heatmap(big_y_vals, cell_size=0.5)
        big_heatmap.match_width(merged_heatmap)
        big_heatmap.match_height(merged_heatmap)
        big_heatmap.move_to(merged_heatmap)

        # Add arrows to show trial direction
        top_left = big_heatmap.get_corner(UP + LEFT)
        bottom_left = big_heatmap.get_corner(DOWN + LEFT)
        bottom_right = big_heatmap.get_corner(DOWN + RIGHT)
        # Arrow for "Starting trial →"
        arrow_start = Arrow(
            start=bottom_left + DOWN * 0.3,
            end=bottom_right + DOWN * 0.3,
            buff=0.1,
            stroke_width=3,
            color=WHITE
        )
        label_start = Text("Starting trial", font_size=20).next_to(arrow_start, DOWN)
        # Arrow for "Ending trial ↓"
        arrow_end = Arrow(
            start=top_left + LEFT * 0.3,
            end=bottom_left + LEFT * 0.3,
            buff=0.1,
            stroke_width=3,
            color=WHITE
        )
        label_end = Text("Ending trial", font_size=20).rotate(90 * DEGREES).next_to(arrow_end, LEFT)
        # Add all heatmap elements to the scene
        heatmap_all = VGroup(big_heatmap, arrow_start, label_start, arrow_end, label_end)

        # Step 5: Animate all small heatmaps transforming into the big one
        transforms = [
            Transform(hm, merged_heatmap, run_time=1)
            for hm in heatmaps
        ]
        self.play(
            self.camera.frame.animate
                .set(width=heatmap_all.width * 0.5, height=heatmap_all.height * 1.2)
                .move_to(heatmap_all.get_right())
        )
        self.play(*transforms)
        merged_heatmap = VGroup(*heatmaps)

        self.play(ReplacementTransform(merged_heatmap, heatmap_all), run_time=3)
        self.remove(merged_heatmap)
        
        # Add the texts
        DA_vs_EI_text = Text("DA vs EI map", font_size=40)
        DA_vs_EI_text.next_to(heatmap_all, RIGHT, buff=2)

        DA_heatmap_text = Text("DA slope heatmap", font_size=40)
        DA_heatmap_text.next_to(DA_vs_EI_text, UP, buff=1)
        
        EI_text = Text("EP-LHb sign", font_size=40)
        EI_text.next_to(DA_vs_EI_text, DOWN, buff=1)

        # Animate texts
        self.play(Write(DA_heatmap_text))
        self.wait(5)
        self.play(Write(EI_text))
        self.wait(1)
        text_transform = [ReplacementTransform(EI_text, DA_vs_EI_text, run_time=1), 
                            ReplacementTransform(DA_heatmap_text, DA_vs_EI_text, run_time=1)]
        self.play(*text_transform)
        self.wait(2)