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

def create_big_heatmap(n, cell_size=0.5, side=LEFT):
    rng = np.random.default_rng(42)
    big_y_vals = np.sin(0.3 * np.arange(n)) + 2 * rng.normal(size=n)
    big_heatmap = create_triangular_heatmap(big_y_vals, cell_size=cell_size)
    big_heatmap.scale_to_fit_width(config.frame_width / 2 - 1)
    big_heatmap.to_edge(side, buff=0.5)

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
    heatmap_group = VGroup(big_heatmap, arrow_start, label_start, arrow_end, label_end)
    return heatmap_group

class MergeHeatmaps(MovingCameraScene):
    def construct(self):
        n_maps = 18
        rng = np.random.default_rng(42)

        # Step 1: Create 30 heatmaps
        heatmaps = VGroup()
        heatmap_sizes = []
        for _ in range(n_maps):
            n = rng.integers(6, 16)
            y_vals = np.sin(0.3 * np.arange(n)) + 2 * rng.normal(size=n)
            hm = create_triangular_heatmap(y_vals, cell_size=0.15)
            heatmaps.add(hm)
            heatmap_sizes.append(n)

        # Step 2: Tile them in a grid
        grid = VGroup(*heatmaps).arrange_in_grid(
            rows=3, cols=6, buff=0.8, align_rows=True
        ).move_to(self.camera.frame.get_center()) 
        # Create an array of delays linearly decreasing
        delays = np.linspace(0.1, 0.02, n_maps)
        for hm, delay in zip(heatmaps, delays):
            self.play(FadeIn(hm, run_time=delay))
            self.wait(delay)
        self.wait(1)

        # Step 3: Create a large composite heatmap
        left_big_heatmap = create_big_heatmap(50, cell_size=0.5, side=LEFT)
        right_big_heatmap = create_big_heatmap(50, cell_size=0.5, side=RIGHT)
        self.add(left_big_heatmap[1:])
        self.add(right_big_heatmap[1:])
        self.play(
            self.camera.frame.animate
                .set(width=left_big_heatmap.width * 0.5, height=left_big_heatmap.height * 1.2)
                .move_to(left_big_heatmap.get_right())
        )
        self.wait(2)

        # Step 4: Animate all small heatmaps move to the big heatmap corner one by one
        top_left_corner = left_big_heatmap[0].get_corner(UP + LEFT)
        bottom_right_corner = right_big_heatmap[0].get_corner(DOWN + RIGHT)
        scaling_factor = left_big_heatmap[0].width / (np.max(heatmap_sizes) * 0.15)

        # Move the heatmaps to the left and right big heatmap corners
        delays = np.linspace(1, 0.1, n_maps)
        for hm, delay in zip(heatmaps, delays):
            hm_copy = hm.copy().scale(scaling_factor)
            hm = hm.scale(scaling_factor)
            self.play(
                AnimationGroup(
                    hm.animate.move_to(top_left_corner,aligned_edge=UP+LEFT).set_opacity(0.5),
                    hm_copy.animate.move_to(bottom_right_corner, aligned_edge=DOWN+RIGHT).set_opacity(0.5),
                    lag_ratio=0
                ),
                run_time=delay
            )
            self.wait(delay)
        self.wait(2)

        # Step 5: Add texts and move camera
        DA_vs_EI_text = Text("DA vs EI map", font_size=40)
        DA_vs_EI_text.next_to(right_big_heatmap, RIGHT, buff=2)
        DA_heatmap_text = Text("DA slope heatmap", font_size=40)
        DA_heatmap_text.next_to(DA_vs_EI_text, UP, buff=1)
        EI_text = Text("EP-LHb sign", font_size=40)
        EI_text.next_to(DA_vs_EI_text, DOWN, buff=1)

        self.play(
            self.camera.frame.animate.move_to(right_big_heatmap.get_right())
        )
        self.play(FadeOut(left_big_heatmap))
        self.wait(2)
        self.play(Write(DA_heatmap_text))

        # Step 6: Show pixel contents in the big heatmap
        highlight_x = [5, 3]#, 17, 18, 19, 20, 21, 22, 23, 24]
        highlight_y = [10, 14]#, 29, 29, 29, 29, 29, 29, 29, 29]
        # Keep references to previous elements to update them smoothly
        prev_highlight = highlight_sq
        prev_arrow = curved_arr
        prev_slope_vec = slopeDA_vec
        prev_scatter_group = scatter_group
        prev_points = scatter_points
        prev_fit_group = fit_group

        for i, j in zip(highlight_x, highlight_y):
            # --- 1. New Target Square ---
            target_square = next((sq for sq in right_big_heatmap if getattr(sq, "indices", None) == (i, j)), None)
            if target_square is None:
                continue

            # --- 2. New DA Vector ---
            new_slopeDA = getDAtrend(DAtrend, t1=i-50, t2=j-50, data_type='smooth')
            new_vector_tex = generate_vector_tex(new_slopeDA_clean)
            new_slope_vec = MathTex(new_vector_tex, font_size=25).next_to(DA_label.get_right(), RIGHT, buff=0.2)

            # --- 3. New Highlight Square + Arrow ---
            new_highlight_sq = Square(
                side_length=target_square.width + 0.1,
                color=GREEN,
                stroke_width=4
            ).move_to(target_square.get_center()).set_fill(opacity=0)

            new_arrow = CurvedArrow(
                start_point=new_highlight_sq.get_right(),
                end_point=DA_label.get_left() - 0.2 * RIGHT,
                angle=-PI/4,
                tip_length=0.2,
                color=GREEN
            )

            # --- 4. Animate vector and square updates ---
            self.play(
                ReplacementTransform(prev_highlight, new_highlight_sq),
                ReplacementTransform(prev_arrow, new_arrow),
                FadeOut(prev_slope_vec),
                Write(new_slope_vec),
                run_time=1
            )

            # --- 5. Scatter Plot Update ---
            if np.all(np.isnan(slopeDA_clean)):
                x_min, x_max = -1, 1
            else:
                x_min, x_max = np.nanmin(slopeDA_clean), np.nanmax(slopeDA_clean)
                if x_min == x_max:  # avoid zero range
                    x_min -= 0.5
                    x_max += 0.5
            y_min, y_max = -1, 1
            x_padding = (x_max - x_min) * 0.1
            y_padding = (y_max - y_min) * 0.1
            x_ticks = np.linspace(x_min - x_padding, x_max + x_padding, 3)
            y_ticks = np.linspace(y_min - y_padding, y_max + y_padding, 3)

            new_scatter_axes = Axes(
                x_range=[x_ticks[0], x_ticks[-1]],
                y_range=[y_ticks[0], y_ticks[-1]],
                x_length=4,
                y_length=4,
                tips=False,
                axis_config={"include_ticks": True, "include_numbers": True},
                x_axis_config={"label_direction": DOWN},
                y_axis_config={"label_direction": LEFT},
            )

            x_label = Text("DA slope", font_size=20)
            y_label = Text("Animal\nEI index", font_size=20)
            x_label.next_to(new_scatter_axes.x_axis, RIGHT)
            y_label.next_to(new_scatter_axes.y_axis, UP)

            slopeDA_vec_center = new_slope_vec.get_center()
            mid_y = (slopeDA_vec_center[1] + self.camera.frame.get_bottom()[1]) / 2
            mid_x = (heatmap_all.get_right()[0] + self.camera.frame.get_right()[0]) / 2

            new_scatter_group = VGroup(new_scatter_axes, x_label, y_label).move_to([mid_x, mid_y, 0])
            new_scatter_group.move_to([mid_x, mid_y, 0])
            new_scatter_points = VGroup(*[
                Dot(
                    point=scatter_axes.coords_to_point(np.squeeze(x).item(), np.squeeze(y).item()),
                    radius=0.08,
                    color=BLUE
                )
                for x, y in zip(new_slopeDA_clean, animalEI_clean)
            ])

            # --- 6. Animate Scatter and Fit ---
            self.play(
                ReplacementTransform(prev_scatter_group, new_scatter_group),
                ReplacementTransform(prev_points, new_scatter_points),
                run_time=1
            )
            new_fit_group, _ = draw_fitted_line(self, new_scatter_axes, new_slopeDA_clean, animalEI_clean, run_time=1)

            # --- 7. Update Highlight Color Based on Fitted Slope ---
            new_color = slope_to_color(true_map[i-50-1, j-50-1], vmin=vmin, vmax=vmax)
            new_highlight_sq.set_fill(new_color, opacity=1)
            self.play(Transform(new_fit_group, new_highlight_sq), run_time=1)
            self.wait(2)

            # Update previous elements
            prev_highlight = new_highlight_sq
            prev_arrow = new_arrow
            prev_slope_vec = new_slope_vec
            prev_scatter_group = new_scatter_group
            prev_points = new_scatter_points
            prev_fit_group = new_fit_group
        

        # Animate texts
       
        self.wait(5)
        self.play(Write(EI_text))
        self.wait(1)
        text_transform = [ReplacementTransform(EI_text, DA_vs_EI_text, run_time=1), 
                            ReplacementTransform(DA_heatmap_text, DA_vs_EI_text, run_time=1)]
        self.play(*text_transform)
        self.wait(2)