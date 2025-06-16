# 파일명: q_table_playing.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory
from turtlesim.msg import Pose
import threading
import tkinter as tk
from PIL import Image, ImageTk
import math
import os
import time
import numpy as np
import random

class QTableVisualizer:
    def __init__(self, master, grid_size, cell_size=25):
        self.window = tk.Toplevel(master)
        self.window.title("Q-Table Visualization")
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.width = grid_size[0] * cell_size
        self.height = grid_size[1] * cell_size
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height, bg='white')
        self.canvas.pack()

    def _normalize_q_values(self, q_values):
        min_q = np.min(q_values)
        max_q = np.max(q_values)
        if max_q == min_q:
            return np.zeros_like(q_values)
        return (q_values - min_q) / (max_q - min_q)

    def update_q_table(self, q_table):
        self.canvas.delete("all")
        normalized_q = self._normalize_q_values(q_table)
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                cell_x, cell_y = x * self.cell_size, y * self.cell_size
                q_up, q_down, q_left, q_right = normalized_q[x, y, :]
                intensity_up = int(q_up * 255)
                c_up = f'#{intensity_up:02x}00{intensity_up:02x}'
                intensity_down = int(q_down * 255)
                c_down = f'#{intensity_down:02x}00{intensity_down:02x}'
                intensity_left = int(q_left * 255)
                c_left = f'#{intensity_left:02x}00{intensity_left:02x}'
                intensity_right = int(q_right * 255)
                c_right = f'#{intensity_right:02x}00{intensity_right:02x}'
                cx, cy = cell_x + self.cell_size / 2, cell_y + self.cell_size / 2
                self.canvas.create_polygon(cell_x, cell_y, cell_x + self.cell_size, cell_y, cx, cy, fill=c_up, outline='gray')
                self.canvas.create_polygon(cell_x, cell_y + self.cell_size, cell_x + self.cell_size, cell_y + self.cell_size, cx, cy, fill=c_down, outline='gray')
                self.canvas.create_polygon(cell_x, cell_y, cell_x, cell_y + self.cell_size, cx, cy, fill=c_left, outline='gray')
                self.canvas.create_polygon(cell_x + self.cell_size, cell_y, cell_x + self.cell_size, cell_y + self.cell_size, cx, cy, fill=c_right, outline='gray')
        self.window.update_idletasks()


class SchimpanziniPlayer(Node):
    def __init__(self):
        super().__init__("schimpanzini_player")
        self.init_parameters()
        self.score = 0
        enemies_spawn_position = [
            (2.20, 6.30), (8.20, 0.70), (6.66, 3.64), (1.40, 2.40)
        ]
        for e in enemies_spawn_position:
            self.spawn_enemy(e)
        items_spawn_position = [
            (3.0, 3.0), (6.0, 6.6), (2.0, 8.0), (1.5, 4.0), (2.0, 5.0),
            (7.0, 7.0), (5.0, 2.0), (3.3, 1.2), (4.1, 6.6), (8.1, 5.5)
        ]
        for i in items_spawn_position:
            self.spawn_item(i)
        self.load_resources()
        self.init_topic()
        self.init_window()
        
    def init_topic(self):
        self.pose_pub = self.create_publisher(Pose, 'turtle1/pose', 10)
        self.pose_msg = Pose()
        self.create_timer(0.01, self.publish_pose)
        threading.Thread(target=rclpy.spin, args=(self, ), daemon=True).start()

    def init_parameters(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.q_table_path = os.path.join(self.script_dir, 'q-table.npy')

        self.window_width = 500
        self.window_height = 500
        self.pixel_scale = 50.0
        self.world_size = (self.window_width / self.pixel_scale, self.window_height / self.pixel_scale)
        self.position_center_x = self.window_width / 2
        self.position_center_y = self.window_height / 2
        self.position_x = 5.5
        self.position_y = 5.5
        self.rotation = 0.0
        self.enemies = []
        self.items = []
        self.step_distance = 1.25
        self.grid_size = (int(self.world_size[0] / self.step_distance), int(self.world_size[1] / self.step_distance))
        self.actions = [0, 1, 2, 3]
        self.num_actions = len(self.actions)
        
        try:
            self.q_table = np.load(self.q_table_path)
            print(f"--- Successfully loaded {self.q_table_path} ---")
        except FileNotFoundError:
            print(f"[ERROR] {self.q_table_path} not found. Starting with an empty Q-table.")
            self.q_table = np.zeros((self.grid_size[0], self.grid_size[1], self.num_actions))

        # *** UPDATED HERE: Epsilon is now a ROS 2 parameter ***
        self.declare_parameter('epsilon', 0.0)
        self.epsilon = self.get_parameter('epsilon').get_parameter_value().double_value
        self.get_logger().info(f"Play mode started with Epsilon = {self.epsilon}")
        
        self.episode = 0

    def get_state_from_pos(self, x, y):
        state_x = int(x / self.step_distance)
        state_y = int((self.world_size[1] - y) / self.step_distance)
        state_x = max(0, min(self.grid_size[0] - 1, state_x))
        state_y = max(0, min(self.grid_size[1] - 1, state_y))
        return state_x, state_y

    def init_window(self):
        self.window = tk.Tk()
        self.window.title('Schimpanzini Bananini - Play Mode')
        self.canvas = tk.Canvas(self.window, width=self.window_width, height=self.window_height)
        self.canvas.pack()
        self.bg_img_tk = ImageTk.PhotoImage(self.bg_img)
        self.enemy_img_tk = ImageTk.PhotoImage(self.enemy_img)
        self.item_img_tk = ImageTk.PhotoImage(self.item_img)
        self.q_visualizer = QTableVisualizer(self.window, self.grid_size, cell_size=50)
        self.q_visualizer.update_q_table(self.q_table)
        self.window.after(50, self.update)
        self.window.mainloop()

    def load_resources(self):
        resources_path = os.path.join(get_package_share_directory('schimpanzini'), 'resources')
        player_img_path = os.path.join(resources_path, 'schimpanzini.png')
        bg_img_path = os.path.join(resources_path, 'bg.png')
        enemy_img_path = os.path.join(resources_path, 'sahur.png')
        item_img_path = os.path.join(resources_path, 'bananini.png')
        self.player_img = self.resize_image(Image.open(player_img_path), 0.075)
        self.bg_img = Image.open(bg_img_path)
        self.enemy_img = self.resize_image(Image.open(enemy_img_path), 0.06)
        self.item_img = self.resize_image(Image.open(item_img_path), 0.05)

    @staticmethod
    def resize_image(image, scale):
        return image.resize([int(image.width * scale), int(image.height * scale)])
    
    def update(self):
        current_state = self.get_state_from_pos(self.position_x, self.position_y)
        
        # *** UPDATED HERE: Restored Epsilon-Greedy logic for adjustable exploration ***
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions) # Explore
        else:
            action = np.argmax(self.q_table[current_state]) # Exploit
        
        prev_pos = (self.position_x, self.position_y)
        if action == 0:
            self.position_y += self.step_distance
            self.rotation = math.pi / 2
        elif action == 1:
            self.position_y -= self.step_distance
            self.rotation = -math.pi / 2
        elif action == 2:
            self.position_x -= self.step_distance
            self.rotation = math.pi
        elif action == 3:
            self.position_x += self.step_distance
            self.rotation = 0
        self.position_x = max(0, min(self.world_size[0] - self.step_distance, self.position_x))
        self.position_y = max(0, min(self.world_size[1] - self.step_distance, self.position_y))
        
        for enemy in self.enemies:
            if self.check_collision(enemy):
                self.position_x, self.position_y = prev_pos
                break
        pop_idx = -1
        for i, item in enumerate(self.items):
            if self.check_collision(item):
                self.score += 10
                pop_idx = i
                break
        
        if pop_idx != -1:
            self.items.pop(pop_idx)
            if not self.items:
                self.episode += 1
                self.score = 0
                items_spawn_position = [(3.0, 3.0), (6.0, 6.6), (2.0, 8.0), (1.5, 4.0)]
                for i in items_spawn_position: self.spawn_item(i)
        
        self.draw()
        self.window.after(50, self.update)
        
    def draw(self):
        self.canvas.delete('all')
        self.canvas.create_image(self.position_center_x, self.position_center_y, image=self.bg_img_tk)
        for e in self.enemies:
            self.canvas.create_image(e[0] * self.pixel_scale, self.window_height - e[1] * self.pixel_scale, image=self.enemy_img_tk)
        for i in self.items:
            self.canvas.create_image(i[0] * self.pixel_scale, self.window_height - i[1] * self.pixel_scale, image=self.item_img_tk)
        rotated_player = self.player_img.rotate(-math.degrees(self.rotation), expand=True)
        self.tk_img = ImageTk.PhotoImage(rotated_player)
        self.canvas.create_image(self.position_x * self.pixel_scale, self.window_height - self.position_y * self.pixel_scale, image=self.tk_img)
        
        # *** UPDATED HERE: Display the current Epsilon value ***
        score_text = (f"경북대 로봇은 Rottoda (@knu_rottoda)\n"
                      f"너의 로며든 점수: {self.score}점\n"
                      f"에피소드: {self.episode} | Epsilon: {self.epsilon:.2f} (플레이 모드)")
        self.canvas.create_text(20, 20, text=score_text, fill="black", font=("Arial", 9, "bold"), anchor='nw')

    def publish_pose(self):
        self.pose_msg.x = float(self.position_x)
        self.pose_msg.y = float(self.position_y)
        self.pose_msg.theta = float(self.rotation)
        self.pose_pub.publish(self.pose_msg)
    def spawn_item(self, position):
        self.items.append(position)
    def spawn_enemy(self, position):
        self.enemies.append(position)
    def check_collision(self, obj):
        return math.hypot(obj[0] - self.position_x, obj[1] - self.position_y) < (self.step_distance + 0.1)

def main(args=None):
    rclpy.init(args=args)
    player = SchimpanziniPlayer()
    player.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()