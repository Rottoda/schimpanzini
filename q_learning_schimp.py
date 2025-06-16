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
                
                # Q-values for UP, DOWN, LEFT, RIGHT
                q_up, q_down, q_left, q_right = normalized_q[x, y, :]

                intensity_up = int(q_up * 255)
                c_up = f'#{intensity_up:02x}00{intensity_up:02x}'

                intensity_down = int(q_down * 255)
                c_down = f'#{intensity_down:02x}00{intensity_down:02x}'

                intensity_left = int(q_left * 255)
                c_left = f'#{intensity_left:02x}00{intensity_left:02x}'

                intensity_right = int(q_right * 255)
                c_right = f'#{intensity_right:02x}00{intensity_right:02x}'

                # Draw triangles for each action
                cx, cy = cell_x + self.cell_size / 2, cell_y + self.cell_size / 2
                self.canvas.create_polygon(cell_x, cell_y, cell_x + self.cell_size, cell_y, cx, cy, fill=c_up, outline='gray') # UP
                self.canvas.create_polygon(cell_x, cell_y + self.cell_size, cell_x + self.cell_size, cell_y + self.cell_size, cx, cy, fill=c_down, outline='gray') # DOWN
                self.canvas.create_polygon(cell_x, cell_y, cell_x, cell_y + self.cell_size, cx, cy, fill=c_left, outline='gray') # LEFT
                self.canvas.create_polygon(cell_x + self.cell_size, cell_y, cell_x + self.cell_size, cell_y + self.cell_size, cx, cy, fill=c_right, outline='gray') # RIGHT
                
        self.window.update_idletasks()


class Schimpanzini(Node):
    def __init__(self):
        super().__init__("schimpanzini_q_learning")
        
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
        # *** ADDED: Define absolute path for q-table file ***
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.q_table_path = os.path.join(self.script_dir, 'q-table.npy')
        
        # --- Environment & Game Parameters ---
        self.window_width = 500
        self.window_height = 500
        self.pixel_scale = 50.0 # 1 unit = 50 pixels
        self.world_size = (self.window_width / self.pixel_scale, self.window_height / self.pixel_scale) # (10.0, 10.0)

        # --- 아래 두 줄 추가 ---
        self.position_center_x = self.window_width / 2
        self.position_center_y = self.window_height / 2

        # Start position
        self.position_x = 5.5
        self.position_y = 5.5
        self.rotation = 0.0

        self.enemies = []
        self.items = []

        # --- Q-learning Parameters ---
        self.step_distance = 1.25  # 이동 거리 d
        self.grid_size = (int(self.world_size[0] / self.step_distance),
                          int(self.world_size[1] / self.step_distance))
        
        self.actions = [0, 1, 2, 3] # 0: Up, 1: Down, 2: Left, 3: Right
        self.num_actions = len(self.actions)
        
        self.q_table = np.zeros((self.grid_size[0], self.grid_size[1], self.num_actions))
        
        self.learning_rate = 0.1      # Alpha
        self.discount_factor = 0.95   # Gamma
        self.epsilon = 1.0            # Exploration rate
        self.epsilon_decay = 0.9995   # Epsilon decay rate
        self.min_epsilon = 0.1        # Minimum epsilon

        self.episode = 0
        self.update_counter = 0

    def get_state_from_pos(self, x, y):
        state_x = int(x / self.step_distance)
        # Y 좌표를 뒤집어서 변환 (전체 높이에서 현재 y값을 뺌)
        state_y = int((self.world_size[1] - y) / self.step_distance)
        state_x = max(0, min(self.grid_size[0] - 1, state_x))
        state_y = max(0, min(self.grid_size[1] - 1, state_y))
        return state_x, state_y

    def init_window(self):
        self.window = tk.Tk()
        self.window.title('Schimpanzini Bananini - Q-Learning')

        self.canvas = tk.Canvas(self.window, width=self.window_width, height=self.window_height)
        self.canvas.pack()

        self.bg_img_tk = ImageTk.PhotoImage(self.bg_img)
        self.enemy_img_tk = ImageTk.PhotoImage(self.enemy_img)
        self.item_img_tk = ImageTk.PhotoImage(self.item_img)
        
        # Initialize Q-Table Visualizer
        self.q_visualizer = QTableVisualizer(self.window, self.grid_size, cell_size=50)

        self.window.after(50, self.update) # Start update loop
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
        # 1. Get current state
        current_state = self.get_state_from_pos(self.position_x, self.position_y)
        
        # 2. Choose action (Epsilon-Greedy)
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions) # Explore
        else:
            action = np.argmax(self.q_table[current_state]) # Exploit
            
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # 3. Perform action and get next state
        prev_pos = (self.position_x, self.position_y)
        if action == 0: # Up
            self.position_y += self.step_distance
            self.rotation = math.pi / 2
        elif action == 1: # Down
            self.position_y -= self.step_distance
            self.rotation = -math.pi / 2
        elif action == 2: # Left
            self.position_x -= self.step_distance
            self.rotation = math.pi
        elif action == 3: # Right
            self.position_x += self.step_distance
            self.rotation = 0

        # Keep agent within bounds
        self.position_x = max(0, min(self.world_size[0] - self.step_distance, self.position_x))
        self.position_y = max(0, min(self.world_size[1] - self.step_distance, self.position_y))
        
        next_state = self.get_state_from_pos(self.position_x, self.position_y)

        # 4. Observe reward
        reward = -1 # Small negative reward for each step to encourage efficiency
        
        # Check for collision with enemies
        for enemy in self.enemies:
            if self.check_collision(enemy):
                print("Tung x 9 Sahur!!")
                reward = -100 # Large negative reward
                self.score -= 20
                self.position_x, self.position_y = prev_pos # Move back
                next_state = self.get_state_from_pos(self.position_x, self.position_y)
                break
        
        # Check for collision with items
        pop_idx = -1
        for i, item in enumerate(self.items):
            if self.check_collision(item):
                print("Wa wa wa, yammy!")
                reward = 50 # Large positive reward
                self.score += 10
                pop_idx = i
                break
        
        if pop_idx != -1:
            self.items.pop(pop_idx)
            # If all items are collected, reset
            if not self.items:
                self.episode += 1
                self.score = 0  # <--- 점수 초기화 코드 추가
                print(f"--- All Bananas Collected! Starting Episode {self.episode} ---")
                
                # *** UPDATED HERE: Save Q-table on even episodes ***
                if self.episode > 0 and self.episode % 2 == 0:
                    try:
                        # *** MODIFIED: Use the absolute path to save the file ***
                        np.save(self.q_table_path, self.q_table)
                        print(f"--- Q-table saved to {self.q_table_path} ---")
                    except Exception as e:
                        print(f"Error saving Q-table: {e}")
                        
                items_spawn_position = [(3.0, 3.0), (6.0, 6.6), (2.0, 8.0), (1.5, 4.0), (2.0, 5.0), (7.0, 7.0), (5.0, 2.0), (3.3, 1.2), (4.1, 6.6), (8.1, 5.5)]
                for i in items_spawn_position: self.spawn_item(i)

        # 5. Update Q-Table
        old_value = self.q_table[current_state[0], current_state[1], action]
        next_max = np.max(self.q_table[next_state[0], next_state[1]])
        
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[current_state[0], current_state[1], action] = new_value

        # --- Drawing and Visualization ---
        self.draw()
        
        # Update Q-table visualization periodically
        self.update_counter += 1
        if self.update_counter % 5 == 0: # Update visualizer every 5 steps
             self.q_visualizer.update_q_table(self.q_table)

        self.window.after(50, self.update) # Schedule next update
        
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
        
        score_text = (f"경북대 로봇은 Rottoda (@knu_rottoda)\n"
                      f"너의 로며든 점수: {self.score}점\n"
                      f"에피소드: {self.episode} | Epsilon: {self.epsilon:.3f}")
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
    schimpanzini = Schimpanzini()
    
    # mainloop is handled inside the class
    
    schimpanzini.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()


