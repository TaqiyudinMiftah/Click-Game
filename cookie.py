import cv2
import numpy as np
import time
from cvzone.HandTrackingModule import HandDetector
import random

# Initialize the camera
cap = cv2.VideoCapture(0)

# Hand detector instance
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Game settings
width, height = 640, 480
box_size = 150  # Size of the main box
corner_size = 30  # Size of the corners
num_corners = 12  # Number of corners
time_limit = 45  # Time limit for the game
start_time = time.time()

# Coordinates for the main box (centered)
box_top_left = (width // 2 - box_size // 2, height // 2 - box_size // 2)
box_bottom_right = (width // 2 + box_size // 2, height // 2 + box_size // 2)

# Randomly generate positions for the corners
corners = [(random.randint(0, width - corner_size), random.randint(0, height - corner_size)) for _ in range(num_corners)]

# Initial obstacle
obstacles = [
    {'position': [100, height//2], 'size': [40, 40], 'speed': 7, 'direction': 'horizontal'}  # Initial obstacle
]

# Flags for visited corners
corner_visited = [False] * num_corners

# Game state
game_over = False
bonus_earned = False  # Bonus points flag

# Activate random corner (initially activate the first one)
current_active_corner = random.randint(0, num_corners - 1)

def add_new_obstacle():
    """Add a new random obstacle with random movement direction."""
    direction = random.choice(['horizontal', 'vertical', 'diagonal'])
    new_obstacle = {
        'position': [random.randint(0, width - 50), random.randint(0, height - 50)],
        'size': [random.randint(20, 50), random.randint(20, 50)],  # Random size
        'speed': random.randint(3, 8),  # Random speed
        'direction': direction
    }
    obstacles.append(new_obstacle)

while True:
    # Read frame from the camera
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip image for mirror effect

    # Detect hands
    hands, img = detector.findHands(img)

    # Draw the main box (optional, center reference)
    cv2.rectangle(img, box_top_left, box_bottom_right, (255, 0, 0), 2)

    # Draw corners and check if they are visited
    for i, corner in enumerate(corners):
        if corner_visited[i]:
            color = (0, 255, 0)  # Green for visited
        elif i == current_active_corner:
            color = (255, 255, 0)  # Yellow for active
        else:
            color = (0, 0, 255)  # Red for inactive
        cv2.rectangle(img, corner, (corner[0] + corner_size, corner[1] + corner_size), color, -1)

    # Draw obstacles and move them
    for obstacle in obstacles:
        obstacle_rect = (obstacle['position'][0], obstacle['position'][1],
                         obstacle['position'][0] + obstacle['size'][0], obstacle['position'][1] + obstacle['size'][1])
        cv2.rectangle(img, (obstacle['position'][0], obstacle['position'][1]),
                      (obstacle['position'][0] + obstacle['size'][0], obstacle['position'][1] + obstacle['size'][1]), 
                      (0, 255, 255), -1)

        # Move obstacle based on its direction
        if obstacle['direction'] == 'horizontal':
            obstacle['position'][0] += obstacle['speed']
            if obstacle['position'][0] + obstacle['size'][0] >= width or obstacle['position'][0] <= 0:
                obstacle['speed'] = -obstacle['speed']
        elif obstacle['direction'] == 'vertical':
            obstacle['position'][1] += obstacle['speed']
            if obstacle['position'][1] + obstacle['size'][1] >= height or obstacle['position'][1] <= 0:
                obstacle['speed'] = -obstacle['speed']
        elif obstacle['direction'] == 'diagonal':
            obstacle['position'][0] += obstacle['speed']
            obstacle['position'][1] += obstacle['speed']
            if obstacle['position'][0] + obstacle['size'][0] >= width or obstacle['position'][0] <= 0:
                obstacle['speed'] = -obstacle['speed']
            if obstacle['position'][1] + obstacle['size'][1] >= height or obstacle['position'][1] <= 0:
                obstacle['speed'] = -obstacle['speed']

    # Check game timer
    elapsed_time = time.time() - start_time
    remaining_time = max(0, time_limit - int(elapsed_time))
    cv2.putText(img, f'Time: {remaining_time}s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if remaining_time <= 0:
        game_over = True
        cv2.putText(img, 'GAME OVER!', (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    if hands:
        lmList = hands[0]['lmList']  # Landmark list
        index_finger = lmList[8]  # Index finger tip coordinates
        cursor_x, cursor_y = index_finger[0], index_finger[1]

        # Draw the cursor (red circle for the fingertip)
        cv2.circle(img, (cursor_x, cursor_y), 10, (0, 0, 255), cv2.FILLED)

        # Check if cursor collides with any obstacle
        for obstacle in obstacles:
            obstacle_rect = (obstacle['position'][0], obstacle['position'][1],
                             obstacle['position'][0] + obstacle['size'][0], obstacle['position'][1] + obstacle['size'][1])
            if (obstacle_rect[0] < cursor_x < obstacle_rect[2]) and (obstacle_rect[1] < cursor_y < obstacle_rect[3]):
                game_over = True
                cv2.putText(img, 'HIT OBSTACLE! GAME OVER!', (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Check if the cursor is inside the active corner
        corner = corners[current_active_corner]
        if (corner[0] < cursor_x < corner[0] + corner_size) and (corner[1] < cursor_y < corner[1] + corner_size):
            corner_visited[current_active_corner] = True
            # Randomly activate the next unvisited corner
            unvisited_corners = [i for i in range(num_corners) if not corner_visited[i]]
            if unvisited_corners:
                current_active_corner = random.choice(unvisited_corners)
                # Add a new obstacle each time a corner is visited
                add_new_obstacle()
            else:
                game_over = True
                cv2.putText(img, 'YOU WON!', (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

    if game_over:
        cv2.imshow("Game", img)
        cv2.waitKey(3000)  # Wait 3 seconds before closing
        break

    # Show the image with game elements
    cv2.imshow("Game", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()