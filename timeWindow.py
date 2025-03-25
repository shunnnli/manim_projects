from manim import *
import numpy as np

class TimeSeriesAndHeatmapSynchronized(MovingCameraScene):
    def construct(self):
        #################################################################
        #                           DATA SETUP                           #
        #################################################################
        n = 6
        x_vals = np.arange(n)
        # Example data: noisy sine
        y_vals = np.sin(0.3 * x_vals) + 0.1 * np.random.randn(n)

        #################################################################
        #                 1) TOP: TIME-SERIES PLOT                      #
        #################################################################
        # Create a 6-unit wide Axes for the top plot
        axes_top = Axes(
            x_range=[0, n, 1],
            y_range=[-1.5, 1.5, 0.5],
            x_length=6,  
            y_length=3,
            axis_config={
                "font_size": 14,
                "include_tip": False,
            },
            x_axis_config={
                "include_numbers": False,
                "include_ticks": False,  # no tick marks
            },
            y_axis_config={
                "include_numbers": False,
                "include_ticks": False,  # no tick marks
            },
        )
        axes_top.to_edge(UP, buff=0.5)
        self.play(Create(axes_top))

        # Labels for the top axes
        x_label_top = Text("Trials", font_size=20)
        x_label_top.next_to(axes_top.x_axis, DOWN, buff=0.2)

        y_label_top = Text("Avg DA during cue", font_size=20)
        y_label_top.rotate(90 * DEGREES)
        y_label_top.next_to(axes_top.y_axis, LEFT, buff=0.2)

        self.play(Write(x_label_top), Write(y_label_top))

        # Scatter dots for the time series
        dots = VGroup(*[
            Dot(axes_top.coords_to_point(x, y), radius=0.04, color=BLUE)
            for x, y in zip(x_vals, y_vals)
        ])
        self.play(FadeIn(dots, lag_ratio=0.05))

        #################################################################
        #           2) BOTTOM: TRIANGULAR HEATMAP (NO AXES)             #
        #################################################################
        # We want the heatmap also to be ~6 units wide, matching the top.
        # i = "Starting trial" (x-axis, left → right)
        # j = "Ending trial"   (y-axis, top → bottom)
        # We'll place squares only where j > i, so end > start.
        cell_size = 6 / n  # so that n columns => 6 units wide

        squares = {}
        heatmap_group = VGroup()

        # For each pair (i, j) with j > i, place a square.
        # We interpret row j as moving downward. So if j=0 is top, j=1 is below it, etc.
        # We'll invert the y-coordinate by using -(j+0.5).
        for j in range(n):
            for i in range(j):
                sq = Square(side_length=cell_size)
                sq.set_fill(GREY, opacity=0.3)
                sq.set_stroke(width=0)
                # Position each square so:
                # x = i + 0.5,   y = -(j + 0.5)
                pos = np.array([i + 0.5, -(j + 0.5), 0]) * cell_size
                sq.move_to(pos)
                squares[(i, j)] = sq
                heatmap_group.add(sq)

        # Place the heatmap group below the top axes, left-aligned
        heatmap_group.next_to(axes_top, DOWN, buff=1, aligned_edge=LEFT)

        # Add labels: "Starting trial" (below), "Ending trial" (left, top→bottom)
        start_label = Text("Starting trial", font_size=20)
        end_label = Text("Ending trial", font_size=20)
        # Rotate by -90° so text reads from top to bottom
        end_label.rotate(-90 * DEGREES)

        start_label.next_to(heatmap_group, DOWN, buff=0.5)
        end_label.next_to(heatmap_group, RIGHT, buff=0.5)

        heatmap_all = VGroup(heatmap_group, start_label, end_label)
        self.play(FadeIn(heatmap_all, lag_ratio=0.01))

        #################################################################
        #                3) ADJUST THE CAMERA VIEWPORT                  #
        #################################################################
        all_mobjects = VGroup(
            axes_top, x_label_top, y_label_top, dots,
            heatmap_all
        )
        self.play(
            self.camera.frame.animate
                .set(width=all_mobjects.width * 1.2, height=all_mobjects.height * 1.2)
                .move_to(all_mobjects.get_center())
        )
        self.wait(0.5)

        #################################################################
        #         4) HELPER FUNCTION: SLOPE → COLOR MAPPING             #
        #################################################################
        def slope_to_color(slope, vmin=-1, vmax=1):
            if np.isnan(slope):
                return BLACK
            slope_clamped = max(min(slope, vmax), vmin)
            alpha = (slope_clamped - vmin) / (vmax - vmin)
            if alpha < 0.5:
                return interpolate_color(BLUE, WHITE, alpha / 0.5)
            else:
                return interpolate_color(WHITE, RED, (alpha - 0.5) / 0.5)

        #################################################################
        #      5) SYNCHRONIZED ANIMATION OF FITTING WINDOWS             #
        #################################################################
        # Suppose each (i, j) is a "window" from trial i to trial j,
        # with j > i. We'll animate drawing lines on top and
        # transforming them into squares below.
        for i in range(n):
            for j in range(i + 1, n):
                # 1) Draw vertical lines + slope line in the top plot
                p1 = axes_top.coords_to_point(i, y_vals[i])
                p2 = axes_top.coords_to_point(j, y_vals[j])

                line1 = Line(
                    axes_top.coords_to_point(i, -1.5),
                    axes_top.coords_to_point(i, 1.5),
                    color=RED
                )
                line2 = Line(
                    axes_top.coords_to_point(j, -1.5),
                    axes_top.coords_to_point(j, 1.5),
                    color=RED
                )
                slope_line = Line(p1, p2, color=GREEN)

                slope_value = (y_vals[j] - y_vals[i]) / (j - i)
                slope_text = MathTex(rf"\text{{slope}} = {slope_value:.2f}")\
                    .next_to(slope_line, UP)

                fit_group = VGroup(line1, line2, slope_line, slope_text)
                self.play(Create(line1), Create(line2), Create(slope_line), Write(slope_text))
                self.wait(0.5)

                # 2) Fade out the original gray square
                gray_sq = squares[(i, j)]
                self.play(FadeOut(gray_sq), run_time=0.3)

                # 3) Create a new colored square in the same position
                colored_square = Square(side_length=cell_size)
                colored_square.set_stroke(width=0)
                colored_square.set_fill(slope_to_color(slope_value), opacity=1)
                colored_square.move_to(gray_sq.get_center())
                self.add(colored_square)

                # 4) Transform a copy of the top‐fit group into this colored square
                self.play(TransformFromCopy(fit_group, colored_square), run_time=0.5)
                self.play(FadeOut(fit_group))

                # Update squares dictionary
                squares[(i, j)] = colored_square

        self.wait(2)
