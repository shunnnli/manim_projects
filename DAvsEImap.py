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
            square.indices = (i, j)  # Store indices for later reference.
            heatmap.add(square)
    return heatmap

def generate_vector_tex(data, orientation='horizontal'):
    # Generate n random numbers and format them into a LaTeX column vector.
    random_strs = [str(val) for val in data]
    if len(data) > 10:
        vec_entries = random_strs[:3] + [r"\dots"] + random_strs[-3:]
    else:
        vec_entries = random_strs
    if orientation == 'horizontal':
        vec_str = r"\begin{pmatrix} " + " & ".join(vec_entries) + r" \end{pmatrix}"
    else:
        vec_str = r"\begin{pmatrix} " + r" \\ ".join(vec_entries) + r" \end{pmatrix}"
    return vec_str

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

    # 3. Add text showing the slope of the line
    slope_text = MathTex(rf"\text{{slope}} = {slope:.2f}",font_size=font_size)\
        .next_to(line, LEFT)
    fit_group = VGroup(line, slope_text)

    # 5. Animate the line and box
    if run_time > 0:
        scene.play(Create(line), Write(slope_text), run_time=run_time)
    return fit_group, slope



class DAvsEImap(MovingCameraScene):
    def construct(self):
        # Step 1: Create the big heatmap
        n_big = 50
        rng = np.random.default_rng(42)
        big_y_vals = np.sin(0.3 * np.arange(n_big)) + 2 * rng.normal(size=n_big)
        big_heatmap = create_triangular_heatmap(big_y_vals, cell_size=0.5)
        big_heatmap.scale_to_fit_width(config.frame_width / 2 - 1)
        big_heatmap.to_edge(LEFT, buff=0.5)
        self.add(big_heatmap)
        
        # Zoom the camera so that the heatmap is visible.
        self.play(
            self.camera.frame.animate
                .set(width=big_heatmap.width * 1.2, height=big_heatmap.height * 1.2)
                .move_to(big_heatmap.get_right())
        )
        self.wait(1)

        # Step 2: Show animal EI vector and slope DA
        # Determine the target position: top left corner of the right half of the frame.
        # The right half's left boundary is the center of the frame.
        # So, the top-left corner of the right half is at (frame_center.x, frame_top.y).
        frame_center = self.camera.frame.get_center()
        frame_top = self.camera.frame.get_top()
        middle_corner = np.array([frame_center[0], frame_top[1]-1, 0])

        EI_label = Text("Animal EI index: ", font_size=15)
        EI_label.move_to(middle_corner)
        animalEI = np.round(np.random.uniform(-1, 1, 10), 2)
        vector_latex = generate_vector_tex(animalEI)
        animalEI_vec = MathTex(vector_latex, font_size=22)
        animalEI_vec.next_to(EI_label, RIGHT, buff=0.2)
        self.play(Write(EI_label), Write(animalEI_vec))
        self.wait(1)

        DA_label = Text("DA slope\nduring window: ", font_size=15)
        DA_label.next_to(EI_label, DOWN, buff=0.3)
        slopeDA = np.round(np.random.uniform(-1, 1, 10), 2)
        vector_latex = generate_vector_tex(slopeDA)
        slopeDA_vec = MathTex(vector_latex, font_size=22)
        slopeDA_vec.next_to(animalEI_vec.get_center(), DOWN, buff=0.4)

        # Step 3: Circle a pixel and show the corresponding vector on the side.
        # Choose a target pixel; for example, (i, j) = (30, 40)
        i, j = 15, 29
        target_square = None
        for square in big_heatmap:
            if hasattr(square, "indices") and square.indices == (i, j):
                target_square = square
                break
        if target_square is None:
            print("Target square not found.")
            return

        # Circle the target pixel
        highlight_sq = Square(
            side_length=target_square.get_width() + 0.1,
            color=RED,
            stroke_width=4
        )
        highlight_sq.set_fill(opacity=0)
        highlight_sq.move_to(target_square.get_center())

        # Step 4: Create a curved arrow.
        # Create a curved arrow from the target square to that target position.
        curved_arr = CurvedArrow(
            start_point=highlight_sq.get_right(),
            end_point=DA_label.get_left() - 0.2*RIGHT,
            angle=-PI/4,
            tip_length=0.2,
            color=RED
        )
        self.play(Create(highlight_sq), Create(curved_arr))
        self.wait(0.5)

        # Step 5: Generate a column vector with random values.
        self.play(Write(DA_label), Write(slopeDA_vec))
        self.wait(2)

        # Step 6: Make a scatter plot of slopeDA and animalEI below slopeDA
        # Compute axis limits and paddings
        x_min, x_max = animalEI.min(), animalEI.max()
        y_min, y_max = slopeDA.min(), slopeDA.max()

        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1

        x_ticks = np.linspace(x_min - x_padding, x_max + x_padding, 3)
        y_ticks = np.linspace(y_min - y_padding, y_max + y_padding, 3)

        # Axes where origin is explicitly set at bottom-left
        scatter_axes = Axes(
            x_range=[x_ticks[0], x_ticks[-1]],
            y_range=[y_ticks[0], y_ticks[-1]],
            x_length=4,
            y_length=4,
            tips=False,
            axis_config={"include_ticks": False, "include_numbers": False},
            x_axis_config={"numbers_to_include": x_ticks, "label_direction": DOWN},
            y_axis_config={"numbers_to_include": y_ticks, "label_direction": LEFT},
        )

        # Labels
        x_label = Text("Animal EI index", font_size=15)
        y_label = Text("DA slope", font_size=15)
        x_label.next_to(scatter_axes.x_axis, DOWN)
        y_label.next_to(scatter_axes.y_axis, LEFT)

        # --- Positioning relative to slopeDA_vec and big_heatmap ---
        # Get key positions
        bottom = self.camera.frame.get_bottom()
        heatmap_right = big_heatmap.get_right()
        screen_right = self.camera.frame.get_right()
        slopeDA_vec_center = slopeDA_vec.get_center()
        # Vertical position: halfway between slopeDA_vec and screen bottom
        mid_y = (slopeDA_vec_center[1] + bottom[1]) / 2
        # Horizontal position: halfway between right of heatmap and right of screen
        mid_x = (heatmap_right[0] + screen_right[0]) / 2
        # Apply position
        scatter_group = VGroup(scatter_axes, x_label, y_label)
        scatter_group.move_to([mid_x, mid_y, 0])
        
        scatter_points = VGroup(*[
            Dot(point=scatter_axes.coords_to_point(x, y), radius=0.08, color=BLUE)
            for x, y in zip(animalEI, slopeDA)
        ])
        
        self.play(Create(scatter_axes), Write(x_label), Write(y_label))
        self.play(FadeIn(scatter_points, lag_ratio=1))
        self.wait(1)

        # Step 7: Fit a line to the scatter plot
        fit_group, slope = draw_fitted_line(self, scatter_axes, animalEI, slopeDA, run_time=2)
        self.wait(1)

        # Step 8: Update the slope value to the heatmap highlighted pixel
        highlight_color = slope_to_color(slope)
        highlight_sq.set_fill(highlight_color, opacity=1)
        self.play(Transform(fit_group, highlight_sq), run_time=1)
        self.wait(2)

