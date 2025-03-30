from manim import *
import numpy as np
from scipy.io import loadmat

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

def remove_nan_values(DA, EI):
    """
    Remove NaN values from x_vals and y_vals.
    Returns cleaned x_vals and y_vals.
    """
    # Convert to NumPy arrays
    DA = np.asarray(DA).flatten()
    EI = np.asarray(EI).flatten()

    # Remove NaNs
    valid = ~np.isnan(DA) & ~np.isnan(EI)
    DA_clean = DA[valid]
    EI_clean = EI[valid]

    return DA_clean, EI_clean

def generate_vector_tex(data, orientation='horizontal'):
    # Generate n random numbers and format them into a LaTeX column vector.
    clean_values = [f"{np.squeeze(val):.2f}" for val in data]
    if len(data) > 6:
        vec_entries = clean_values[:3] + [r"\dots"] + clean_values[-4:]
    else:
        vec_entries = clean_values
    if orientation == 'horizontal':
        vec_str = r"\begin{pmatrix} " + " & ".join(vec_entries) + r" \end{pmatrix}"
    else:
        vec_str = r"\begin{pmatrix} " + r" \\ ".join(vec_entries) + r" \end{pmatrix}"
    return vec_str

def getDAtrend(DAtrend, t1, t2, data_type='smoothed'):
    # Determine dataType field
    if 'smooth' in data_type:
        key = 'slopeMap_smoothed'
    else:
        key = 'slopeMap_raw'

    stats = np.full(len(DAtrend), np.nan)

    for a in range(len(DAtrend)):
        try:
            field_data = DAtrend[a][key]
            if t1 <= 0: t1_final = field_data.shape[0] + t1
            else: t1_final = t1
            if t2 <= 0: t2_final = field_data.shape[1] + t2
            else: t2_final = t2

            if (t1_final > field_data.shape[0] or t2_final > field_data.shape[1]
                or t1_final <= 0 or t2_final <= 0 or t1 > t2_final):
                stats[a] = np.nan
            else:
                stats[a] = field_data[t1_final-1, t2_final-1]  # MATLAB is 1-based, Python is 0-based
        except Exception as e:
            print(f"Error accessing data for animal {a},: {e}")
            stats[a] = np.nan
    
    return stats


class MergeHeatmaps(MovingCameraScene):
    def construct(self):
        n_maps = 18
        rng = np.random.default_rng(42)
        DAtrend = loadmat('/Users/shunli/Desktop/manim_projects/DAtrend_manim.mat')
        DAtrend = DAtrend['DAtrend_manim'].flatten()
        animalEI_mat = loadmat('/Users/shunli/Desktop/manim_projects/animalEIpeaks.mat')
        animalEI = animalEI_mat['animalEIindex_peaks']

        # Step 1: Create heatmaps
        heatmaps = VGroup()
        for _ in range(n_maps):
            # If last iteration, use n = 15
            if _ == n_maps - 1: n = 15
            else: n = rng.integers(6, 16)
            y_vals = np.sin(0.3 * np.arange(n)) + 2 * rng.normal(size=n)
            hm = create_triangular_heatmap(y_vals, cell_size=0.15)
            heatmaps.add(hm)

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
        scaling_factor = left_big_heatmap[0].width / (15 * 0.15)

        # Move the heatmaps to the left and right big heatmap corners
        delays = np.linspace(0.8, 0.1, n_maps)
        for hm, delay in zip(heatmaps, delays):
            hm_copy = hm.copy().scale(scaling_factor)
            hm = hm.scale(scaling_factor)
            self.play(
                AnimationGroup(
                    hm_copy.animate.move_to(top_left_corner,aligned_edge=UP+LEFT).set_opacity(0.5),
                    hm.animate.move_to(bottom_right_corner, aligned_edge=DOWN+RIGHT).set_opacity(0.5),
                    lag_ratio=0
                ),
                run_time=delay
            )
            self.wait(delay)
        self.wait(2)

        # Step 5: Add texts and move camera
        self.play(self.camera.frame.animate.move_to(right_big_heatmap.get_right()), run_time=1)
        self.play(FadeOut(left_big_heatmap),FadeOut(heatmaps), run_time=1)
        # self.play(ReplacementTransform(heatmaps,right_big_heatmap[0]), run_time=2)
        self.play(FadeIn(right_big_heatmap[0]), run_time=2)
        self.wait(2)
        frame_center = self.camera.frame.get_center() + 1*RIGHT
        frame_top = self.camera.frame.get_top() + 0.5*DOWN
        middle_corner = np.array([frame_center[0], frame_top[1]-1, 0])

        DA_vs_EI_text = Text("DA vs EI map", font_size=40)
        DA_vs_EI_text.next_to(right_big_heatmap, RIGHT, buff=2)
        DA_label = Text("DA slope\nduring window", font_size=30)
        DA_label.move_to(middle_corner)
        EI_text = Text("Animal EP-LHb sign", font_size=30)
        EI_text.next_to(DA_label, RIGHT, buff=1)

        # Step 6: Show pixel contents in the big heatmap
        highlight_x = [22, 5, 17, 12]#, 17, 18, 19, 20, 21, 22, 23, 24]
        highlight_y = [48, 40, 29, 23]#, 29, 29, 29, 29, 29, 29, 29, 29]

        # Show DA slope vector
        i, j = 8,10
        slopeDA = getDAtrend(DAtrend, t1=i-50, t2=j-50, data_type='smooth')
        slopeDA_clean, animalEI_clean = remove_nan_values(slopeDA, animalEI)
        vector_latex = generate_vector_tex(slopeDA_clean, orientation='vertical')
        slopeDA_vec = MathTex(vector_latex, font_size=40)
        slopeDA_vec.next_to(DA_label.get_center(), DOWN, buff=1)
        target_square = None
        for square in right_big_heatmap[0]:
            if hasattr(square, "indices") and square.indices == (i, j):
                target_square = square
                break

        # Circle the target pixel
        highlight_sq = Square(
            side_length=target_square.width + 0.1,
            color=GREEN,
            stroke_width=4
        )
        highlight_sq.set_fill(opacity=0)
        highlight_sq.move_to(target_square.get_center())

        # Create a curved arrow from the target square to that target position.
        curved_arr = CurvedArrow(
            start_point=highlight_sq.get_right(),
            end_point=DA_label.get_left() - 0.2*RIGHT,
            angle=-PI/4,
            tip_length=0.2,
            color=GREEN
        )
        self.play(Create(highlight_sq), Create(curved_arr))
        self.wait(0.5)
        self.play(Write(DA_label), Write(slopeDA_vec))
        self.wait(2)

        # Animal EI
        vector_latex = generate_vector_tex(animalEI_clean, orientation='vertical')
        animalEI_vec = MathTex(vector_latex, font_size=40)
        animalEI_vec.next_to(EI_text.get_center(), DOWN, buff=1)
        self.play(Write(EI_text), Write(animalEI_vec))
        self.wait(2)

        # Keep references to previous elements to updat e them smoothly
        prev_highlight = highlight_sq
        prev_arrow = curved_arr
        prev_slope_vec = slopeDA_vec

        for i, j in zip(highlight_x, highlight_y):
            # --- 1. New Target Square ---
            target_square = next((sq for sq in right_big_heatmap[0] if getattr(sq, "indices", None) == (i, j)), None)
            if target_square is None:
                continue

            # --- 2. New DA Vector ---
            slopeDA = getDAtrend(DAtrend, t1=i-50, t2=j-50, data_type='smooth')
            slopeDA_clean, animalEI_clean = remove_nan_values(slopeDA, animalEI)
            new_vector_tex = generate_vector_tex(slopeDA_clean, orientation='vertical')
            new_slope_vec = MathTex(new_vector_tex, font_size=40).next_to(DA_label.get_center(), DOWN, buff=1)

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
            self.wait(1)

            # Update previous elements
            prev_highlight = new_highlight_sq
            prev_arrow = new_arrow
            prev_slope_vec = new_slope_vec
        

        # Animate texts
        self.wait(5)
        text_transform = [ReplacementTransform(EI_text, DA_vs_EI_text, run_time=1), 
                            ReplacementTransform(DA_label, DA_vs_EI_text, run_time=1),
                            ReplacementTransform(prev_slope_vec, DA_vs_EI_text, run_time=1),
                            ReplacementTransform(animalEI_vec, DA_vs_EI_text, run_time=1)]
        self.play(*text_transform)
        self.wait(2)