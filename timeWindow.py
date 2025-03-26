from manim import *
import numpy as np

#################################################################
#                     HELPER FUNCTION                           #
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

def draw_fitted_line(scene, axes, x_vals, y_vals, 
                    line_color=GREEN, box_color=RED, box_opacity=0.3,
                    font_size=30, run_time=1):
    print(x_vals, y_vals)
    # 1. Fit line using linear regression
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    fit_func = lambda x: slope * x + intercept
    

    # 2. Plot the best-fit line over the window of the data
    x_min, x_max = min(x_vals), max(x_vals)
    line = axes.plot(fit_func, x_range=[x_min, x_max], color=line_color)

    # 3. Create a box to highlight the fitting window
    y_min, y_max = axes.y_range[0], axes.y_range[1]
    ll = axes.coords_to_point(x_min, y_min)
    ur = axes.coords_to_point(x_max, y_max)
    width = ur[0] - ll[0]
    height = ur[1] - ll[1]
    # Create a box with width=0 and grow it horizontally
    box = Rectangle(
        width=0,
        height=height,
        stroke_color=box_color,
        fill_color=box_color,
        fill_opacity=box_opacity,
        stroke_width=2
    )
    # Align the box to the left of the desired window
    box.move_to(axes.coords_to_point(x_min, (y_min + y_max)/2), aligned_edge=LEFT)
    # Target box (full width) for transformation
    target_box = Rectangle(
        width=width,
        height=height,
        stroke_color=box_color,
        fill_color=box_color,
        fill_opacity=box_opacity,
        stroke_width=2
    )
    target_box.move_to((ll + ur) / 2)

    # 4. Add text showing the slope of the line
    slope_text = MathTex(rf"\text{{slope}} = {slope:.2f}",font_size=font_size)\
        .next_to(target_box, UP)
    fit_group = VGroup(box, target_box, line, slope_text)

    # 5. Animate the line and box
    if run_time > 0:
        scene.play(Transform(box, target_box), run_time=run_time)
        scene.play(Create(line), Write(slope_text), run_time=run_time)
    return fit_group, slope


#################################################################
#                           MANIM SCENE                         #
#################################################################
class TimeSeriesToHeatmap(MovingCameraScene):
    def construct(self):
        #################################################################
        #                           DATA SETUP                           #
        #################################################################
        # y_vals = [0.5,1,0.1,-1,-0.3,0.8]
        y_vals = [1.36325003519570, -0.157837602947608, 0.271344196566665, -0.0138622159137982, -0.335586506976710, 1.00239982546129, 0.109018295366464, -0.349335412468967, 0.555618580189778, 0.335257247140798, 1.03937719388420, 0.0148790003825635, 0.251493823288295, 0.506256408522377, 2.99209708379337, 1.98388028132156, 0.565462896568772, 0.778207423142364, 0.283187972787460, 1.92436082324897, 1.16413403816985, 0.846797976865968, 0.655644055633262, -0.0361664745246513, 1.24068560725520, 0.521646168564981, 1.53746532580992, 1.95845288244205, -0.113881322800335, 1.66078721856901, 0.0566985282457796, 1.22128614136409, 0.618051833302253, 1.26657221030343, 0.959853810891455, 1.40047898246035, 0.702610845887428, 1.38958159127824, 1.22903215829783]
        n_trials = len(y_vals)
        x_vals = np.arange(n_trials)

        #################################################################
        #                 1) TOP: TIME-SERIES PLOT                      #
        #################################################################
        # Create a 6-unit wide Axes for the top plot
        axes_top = Axes(
            x_range=[0, n_trials, 1],
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
        # Labels for the top axes
        x_label_top = Text("Trials", font_size=20)
        x_label_top.next_to(axes_top.x_axis, DOWN, buff=0.2)
        y_label_top = Text("Avg DA during cue", font_size=20)
        y_label_top.rotate(90 * DEGREES)
        y_label_top.next_to(axes_top.y_axis, LEFT, buff=0.2)

        timeseries_axes = VGroup(axes_top,x_label_top,y_label_top)

        # Adjust camera to zoom in
        self.play(
            self.camera.frame.animate
                .set(width=timeseries_axes.width * 5
                , height=timeseries_axes.height * 1.2)
                .move_to(timeseries_axes.get_center())
        )
        self.wait(0.5)
        self.play(Create(axes_top))
        self.play(Write(x_label_top), Write(y_label_top))

        # Scatter dots for the time series
        dots = VGroup(*[
            Dot(axes_top.coords_to_point(x, y), radius=0.1, color=BLUE)
            for x, y in zip(x_vals, y_vals)
        ])
        self.play(FadeIn(dots, lag_ratio=0.5))

        #################################################################
        #           2) BOTTOM: TRIANGULAR HEATMAP (NO AXES)             #
        #################################################################
        # We want the heatmap also to be ~6 units wide, matching the top.
        # i = "Starting trial" (x-axis, left → right)
        # j = "Ending trial"   (y-axis, top → bottom)
        # We'll place squares only where j > i, so end > start.
        cell_size = 6 / n_trials  # so that n_trials columns => 6 units wide

        squares = {}
        heatmap_group = VGroup()

        # For each pair (i, col) with col > i, place a square.
        # We interpret row i as moving downward. So if col=0 is top, col=1 is below it, etc.
        # We'll invert the y-coordinate by using -(col+0.5).
        for col in range(n_trials):
            for i in range(col):
                sq = Square(side_length=cell_size)
                sq.set_fill(GREY, opacity=0.3)
                sq.set_stroke(width=0)
                # Position each square so:
                # x = i + 0.5,   y = -(col + 0.5)
                pos = np.array([i + 0.5, -(col + 0.5), 0]) * cell_size
                sq.move_to(pos)
                squares[(i, col)] = sq
                heatmap_group.add(sq)

        # Place the heatmap group below the top axes, left-aligned
        heatmap_group.next_to(axes_top, DOWN, buff=1, aligned_edge=LEFT)

        # # Add labels: "Starting trial" (below), "Ending trial" (left, top→bottom)
        # start_label = Text("Starting trial", font_size=20)
        # end_label = Text("Ending trial", font_size=20).rotate(90 * DEGREES)

        # start_label.next_to(heatmap_group, DOWN, buff=0.5)
        # end_label.next_to(heatmap_group, LEFT, buff=0.5)

        # Add arrows to indicate the direction of the heatmap
        # Compute key corners of the heatmap
        top_left = heatmap_group.get_corner(UP + LEFT)
        bottom_left = heatmap_group.get_corner(DOWN + LEFT)
        bottom_right = heatmap_group.get_corner(DOWN + RIGHT)

        # Arrow for "Starting trial →"
        arrow_start = Arrow(
            start=bottom_left + DOWN * 0.5,
            end=bottom_right + DOWN * 0.5,
            buff=0.1,
            stroke_width=3,
            color=WHITE
        )
        label_start = Text("Starting trial", font_size=20).next_to(arrow_start, DOWN)

        # Arrow for "Ending trial ↓"
        arrow_end = Arrow(
            start=top_left + LEFT * 0.5,
            end=bottom_left + LEFT * 0.5,
            buff=0.1,
            stroke_width=3,
            color=WHITE
        )
        label_end = Text("Ending trial", font_size=20).rotate(90 * DEGREES).next_to(arrow_end, LEFT)

        # Add all heatmap elements to the scene
        heatmap_all = VGroup(heatmap_group, arrow_start, label_start, arrow_end, label_end)

        # Align the entire heatmap_all group horizontally to the center of the time series axes
        dx = axes_top.get_center()[0] - heatmap_all.get_center()[0]
        heatmap_all.shift(RIGHT * dx)

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
                .set(width=all_mobjects.width * 0.5, height=all_mobjects.height * 1.2)
                .move_to(all_mobjects.get_center())
        )
        self.wait(0.5)

        #################################################################
        #      4) SYNCHRONIZED ANIMATION OF FITTING WINDOWS             #
        #################################################################
        # Suppose each (i, j) is a "window" from trial i to trial j,
        # with j > i. We'll animate drawing lines on top and
        # transforming them into squares below.

        # Define run_time heatmap
        run_times = np.concatenate([
            np.ones(2),
            np.linspace(1, 0.2, 5),
            np.ones(n_trials - 8) * 0.02
        ])

        # Randomly select some long windows to plot until run time is less than 0.02
        for run_time in run_times:
            i = np.random.choice(n_trials-30) + 1
            j = i + np.random.choice(15) + 10

            fit_group, slope = draw_fitted_line(self,axes_top, x_vals[i:j], y_vals[i:j],
                                            run_time=run_time)
            self.wait(run_time)

            colored_square = Square(side_length=cell_size)
            colored_square.set_stroke(width=0)
            colored_square.set_fill(slope_to_color(slope), opacity=1)
            colored_square.move_to(squares[(i, j)].get_center())
            self.add(colored_square)
        
            self.play(TransformFromCopy(fit_group, colored_square), run_time=run_time)
            self.play(FadeOut(fit_group), run_time=run_time)
        
        # Fade in the rest of the plot
        animations = []
        for i in range(n_trials):
            for j in range(i + 1, n_trials):
                if j - i < 3: continue  # Skip windows of size 1
                fit_group, slope = draw_fitted_line(self,axes_top, x_vals[i:j], y_vals[i:j],
                                                run_time=0)

                colored_square = Square(side_length=cell_size)
                colored_square.set_stroke(width=0)
                colored_square.set_fill(slope_to_color(slope), opacity=1)
                colored_square.move_to(squares[(i, j)].get_center())
                animations.append(FadeIn(colored_square))

                # Update squares dictionary
                squares[(i, j)] = colored_square

        self.play(AnimationGroup(*animations, lag_ratio=0), run_time=1)
        self.wait(2)