import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory
from turtlesim.msg import Pose
from turtlesim.srv import Spawn
import threading
import tkinter as tk
from PIL import Image, ImageTk
import math
import os
import time

class Schimpanzini(Node):
    def __init__(self):
        super().__init__("schimpanzini")
        
        self.init_parameters()
        
        # 득점 설계
        self.score = 0

        # Spawn multiple Sahurs
        enemies_spawn_position = [
            (2.20, 6.30),
            (8.20, 0.70),
            (6.66, 3.64),
            (3.30, 5.12),
            (4.40, 2.40),
            (1.40, 2.40)
        ]
        for e in enemies_spawn_position:
            self.spawn_enemy(e)

        # Spawn bananas to eat
        items_spawn_position = [
            (3.0, 3.0),
            (6.0, 6.6),
            (2.0, 8.0),
            (1.5, 4.0),
            (2.0, 6.0),
            (7.0, 7.0),
            (5.0, 2.0),
            (3.3, 1.2),
            (4.1, 6.6),
            (8.1, 5.5)
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

        self.cmd_vel_sub = self.create_subscription(Twist, 'turtle1/cmd_vel', self.cmd_vel_callback, 10)
        self.spawn_item_server = self.create_service(Spawn, 'spawn_item', self.spawn_item)

        threading.Thread(target=rclpy.spin, args=(self, ), daemon=True).start()

    def init_parameters(self):
        self.window_width = 500
        self.window_height = 500
        self.pixel_scale = 50.0
        self.fps = 60.0
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        self.position_center_x = self.window_width / 2
        self.position_center_y = self.window_height / 2
        self.position_x = self.position_center_x / self.pixel_scale
        self.position_y = self.position_center_y / self.pixel_scale

        self.rotation = 0.0
        self.last_command_time = time.time()
        self.enemies = []
        self.items = []
        
        # 충돌 이후 전위치로 돌리기 위하여 췸팬지니 위치 저장
        self.prev_position_x = self.position_x
        self.prev_position_y = self.position_y
        self.prev_rotation = self.rotation
        
        self.enemy_collision_cooldown = 1.0  # 1초 무적
        self.last_enemy_collision_time = 0  # 마지막 충돌 시점 

    def init_window(self):
        self.window = tk.Tk()
        self.window.title('Schimpanzini Bananini')

        self.canvas = tk.Canvas(self.window, width=self.window_width, height=self.window_height)
        self.canvas.pack()

        self.bg_img_tk = ImageTk.PhotoImage(self.bg_img)
        self.enemy_img_tk = ImageTk.PhotoImage(self.enemy_img)
        self.item_img_tk = ImageTk.PhotoImage(self.item_img)

        self.window.after(int(1000 / self.fps), self.update)
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
        width = image.width
        height = image.height

        r_image = image.resize([int(width * scale), int(height * scale)])

        return r_image
        
    
    def update(self):
        dt = 1.0 / self.fps
        if(time.time() - self.last_command_time > 1):
            self.linear_velocity = 0.0
            self.angular_velocity = 0.0
            
        # pose_msg에 충돌 전 위치 저장
        self.prev_position_x = self.position_x
        self.prev_position_y = self.position_y
        self.prev_rotation = self.rotation

        # 위치 갱신
        self.rotation += self.angular_velocity * dt
        self.position_x += self.linear_velocity * math.cos(self.rotation)* dt
        self.position_y += self.linear_velocity * math.sin(self.rotation) * dt

        self.position_x = max(0, min(self.window_width / self.pixel_scale, self.position_x))
        self.position_y = max(0, min(self.window_height / self.pixel_scale, self.position_y))

        self.draw()
        self.window.after(int(1000 / self.fps), self.update)
        #print(f'x : {self.position_x:.3f}, y : {self.position_y:.3f}\tpx : {int(self.position_x * self.pixel_scale)}, py : {int(self.position_y * self.pixel_scale)}', end='\r')

        # enemy 충돌 검사
        current_time = time.time()
        for i in range(len(self.enemies)):
            # if(self.check_collision(self.enemies[i])):
            if self.check_collision(self.enemies[i]) and (current_time - self.last_enemy_collision_time > self.enemy_collision_cooldown):
                print("Tung x 9 Sahur!!")
                self.score -= 20
                print(f"Total score: {self.score}")
                
                # 충돌 전 위치로 복원
                self.position_x = self.prev_position_x
                self.position_y = self.prev_position_y
                self.rotation = self.prev_rotation
                
                self.last_enemy_collision_time = current_time  # 충돌 시간 업데이트
                break  # 한 번만 복원

        pop_idx = []
        for i in range(len(self.items)):
            if(self.check_collision(self.items[i])):
                pop_idx.append(i)
                print("Wa wa wa, yammy!")
                self.score += 10
                print(f"Score: {self.score}")
                
        for i in pop_idx:
            self.items.pop(i)
        
    def draw(self):
        self.canvas.delete('all')

        # 배경 이미지
        self.canvas.create_image(self.position_center_x, self.position_center_y, image=self.bg_img_tk)
        
        # 적(사후르) 그리기
        for e in self.enemies:
            self.canvas.create_image(e[0] * self.pixel_scale, 
                                     self.window_height - e[1] * self.pixel_scale, 
                                     image=self.enemy_img_tk)
        
        # 아이템(바나나) 그리기
        for i in self.items:
            self.canvas.create_image(i[0] * self.pixel_scale, 
                                     self.window_height - i[1] * self.pixel_scale, 
                                     image=self.item_img_tk)

        # 플레이어(췸판지니) 그리기
        rotated = self.player_img.rotate(math.degrees(self.rotation), expand=True)
        self.tk_img = ImageTk.PhotoImage(rotated)
        self.canvas.create_image(self.position_x * self.pixel_scale, 
                                 self.window_height - self.position_y * self.pixel_scale, 
                                 image=self.tk_img)
        
        # 점수 표시
        score_text = f"경북대 로봇은 Rottoda (@knu_rottoda)\n너의 로며든 점수: {self.score}점"
        self.canvas.create_text(20, 20, text=score_text, fill="black", font=("Arial", 13, "bold"), anchor='nw')

    def publish_pose(self):
        self.pose_msg.x = float(self.position_x)
        self.pose_msg.y = float(self.position_y)
        self.pose_msg.theta = self.rotation

        self.pose_msg.linear_velocity = self.linear_velocity
        self.pose_msg.angular_velocity = self.angular_velocity

        self.pose_pub.publish(self.pose_msg)

    def cmd_vel_callback(self, msg:Twist):
        self.last_command_time = time.time()
        self.linear_velocity = msg.linear.x
        self.angular_velocity = msg.angular.z
    
    def spawn_item(self, position):
        self.items.append(position)
        print(f'item was spawned at {position[0]}, {position[1]}')

    def spawn_enemy(self, position):
        self.enemies.append(position)
        print(f'enemy was spawned at {position[0]}, {position[1]}')
    
    def check_collision(self, object):
        distance = math.hypot(object[0] - self.position_x, object[1] - self.position_y)
        if(distance < 0.5):
            return True
        else:
            return False
    

def main(args=None):
    rclpy.init(args=args)
    schimpanzini = Schimpanzini()
    rclpy.spin(schimpanzini)

    schimpanzini.window.destroy()
    schimpanzini.window.quit()
    schimpanzini.destroy_node()
    rclpy.try_shutdown()

if __name__ == "__main__":
    main()