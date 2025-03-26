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
            square.indices = (i, j)  # <== assign indices attribute here

            heatmap.add(square)
    return heatmap

def generate_random_vector_tex(n, low=-1, high=1, decimals=2):
    # Generate n random numbers between low and high, rounded to the desired decimals.
    random_vals = np.round(np.random.uniform(low, high, n), decimals)
    # Convert each number to string
    random_strs = [str(val) for val in random_vals]
    
    if n > 10:
        # Keep first 3 and last 3 values, replace the middle with \dots.
        vec_entries = random_strs[:3] + [r"\dots"] + random_strs[-3:]
    else:
        vec_entries = random_strs
    
    # Create a LaTeX formatted column vector string
    vec_str = r"\begin{pmatrix} " + r" \\ ".join(vec_entries) + r" \end{pmatrix}"
    return vec_str


class MergeHeatmaps(MovingCameraScene):
    def construct(self):
        # Step 1: Plot the overall heatmap
        n_big = 50
        rng = np.random.default_rng(42)
        big_y_vals = np.sin(0.3 * np.arange(n_big)) + 2 * rng.normal(size=n_big)
        big_heatmap = create_triangular_heatmap(big_y_vals, cell_size=0.5)
        big_heatmap.scale_to_fit_width(config.frame_width / 2 - 1)
        big_heatmap.to_edge(LEFT, buff=0.5)

        self.play(
            self.camera.frame.animate
                .set(width=big_heatmap.width * 1.2, height=big_heatmap.height * 1.2)
                .move_to(big_heatmap.get_right())
        )
        self.add(big_heatmap)
        self.wait(2)

        # Step 2: Circle a pixel and show the corresponding vector on the side
        i, j = 15, 29
        # Search for the square in big_heatmap that has indices (2,4)
        target_square = None
        for square in big_heatmap:
            if hasattr(square, "indices") and square.indices == (i, j):
                target_square = square
                break
        
        if target_square is None:
            print("Target square not found.")
            return

        # Create a circle that slightly exceeds the square's boundaries
        circ = Circle(radius=target_square.get_width()/2 + 0.05, color=RED)
        circ.move_to(target_square.get_center())
        
        # Create an arrow (vector) pointing rightwards from the pixel
        arr = Arrow(
            start=target_square.get_right(),
            end=target_square.get_right() + RIGHT*5,
            buff=0.1,
            color=WHITE
        )
        
        # Animate the circle and arrow creation
        self.play(Create(circ), Create(arr))
        self.wait(1)

        # Step 3: Create an nx1 column vector (using MathTex) next to the arrow.
        vector_latex = generate_random_vector_tex(n=30, low=-1, high=1, decimals=2)
        vector_tex = MathTex(vector_latex, font_size=25)
        vector_tex.next_to(arr, RIGHT, buff=0.3)

        self.play(Write(vector_tex))
        self.wait(2)