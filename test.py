from manim import *

class TestCameraFrame(Scene):
    def construct(self):
        print("Camera Frame:", self.camera.frame)
        self.wait(1)