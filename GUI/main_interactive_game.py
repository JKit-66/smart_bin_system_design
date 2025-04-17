import pygame
import random
import time
import numpy as np
import cv2
import json
import textwrap
import os
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import ctypes
from collections import defaultdict
import cvzone
from ultralytics import YOLO
import math
#import serial

# Define colors for categories
category_colors = {
    'Mixed Recycling': (96, 176, 229),  # #60b0e5
    'Paper': (65, 213, 85),             # #41d555
    'Food Waste': (168, 103, 185),      # #a867b9
    'General Waste': (236, 101, 90)     # #ec655a
}


hover_color = {
    "Mixed Recycling": '#7acbf0',
    "Paper": '#67e97b', 
    "Food Waste": '#d6a8e0', 
    "General Waste":'#f08b84'
}

categories = ["Mixed Recycling", "Paper", "Food Waste", "General Waste"]

# Improved trash list structure: dictionary with lists for each category
trash_list = {
    'Mixed Recycling': [
        {"name": "Aluminum Can"},
        {"name": "Glass Bottle"},
        {"name": "Plastic Container"},
        {"name": "Glass Jar"}
    ],
    'Paper': [
        {"name": "Cardboard Box"},
        {"name": "Paper Cup"},
        {"name": "Magazine"},
        {"name": "Paper Bag"},
        {"name": "Notebook"}
    ],
    'Food Waste': [
        {"name": "Banana Peel"},
        {"name": "Eggshells"},
        {"name": "Tea Bag"},
        {"name": "Vegetable Peels"},
        {"name": "Coffee Grounds"}
    ],
    'General Waste': [
        {"name": "Broken Glass"},
        {"name": "Ceramic Plate"},
        {"name": "Used Tissue"},
        {"name": "Cigarette Butt"},
        {"name": "Expired Medicine"}
    ]
}


pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN | pygame.RESIZABLE)
#pygame.mouse.set_visible(False)
clock = pygame.time.Clock()

# Font Management
pixel_sans_path = 'font/PixelifySans-Regular.ttf'
press_start_2p_path = 'font/PressStart2P-Regular.ttf'
lilita_one_path = 'font/LilitaOne-Regular.ttf'

# File managemet
congrat_image = pygame.image.load("one.png")  # Replace with your image file path
C_target_size = (440*1.1, 566*1.1) 
congrat_image = pygame.transform.smoothscale(congrat_image, C_target_size)  #(400,400)
#pygame.transform.scale(congrat_image, (400, 400))  # Resize as needed

sadden_image = pygame.image.load("two.png")  # Replace with your image file path
S_target_size = (550, 454) 
sadden_image  = pygame.transform.smoothscale(sadden_image, S_target_size) #(350, 350)

school_image = pygame.image.load("school_logo.png").convert_alpha()  # Replace with your image file path
original_width, original_height = school_image.get_size()
target_size = (385*1.5, 84*1.5) #(385, 84) 
scaled_school  = pygame.transform.smoothscale(school_image, target_size)  # Resize as needed

twotouch_image = pygame.image.load("TwoTouch.png").convert_alpha()  # Replace with your image file path
twotouch_image_target_size = (1147*0.06, 1962*0.06) #(1147, 1962) 
scaled_twotouch  = pygame.transform.smoothscale(twotouch_image, twotouch_image_target_size)  # Resize as needed
scaled_twotouch.set_alpha(180)


STATS_FILE = "json_file/stats.json"
RECYCLE_FILE = "json_file/detectable.json"

def load_facts(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return "File not found."
    except json.JSONDecodeError:
        return "Error decoding JSON file."

trash_facts = load_facts(RECYCLE_FILE)    


'''JSON FILE RELATED FUNCTIONS AND OPERATIONS'''
def load_stats():
    """Load statistics from a JSON file."""
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, "r") as f:
            return json.load(f)
    else:
        return {
            "sessions": 0,
            "total_questions": 0,
            "categories": {
                "Mixed Recycling": {"correct": 0, "incorrect": 0, "times_shown": 0},
                "Paper": {"correct": 0, "incorrect": 0, "times_shown": 0},
                "Food Waste": {"correct": 0, "incorrect": 0, "times_shown": 0},
                "General Waste": {"correct": 0, "incorrect": 0, "times_shown": 0}
            },
            "total_correct": 0,
            "total_incorrect": 0
        }

def save_stats(stats):
    """Save statistics to a JSON file."""
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

def update_stats(category, is_correct):
    stats = load_stats()
    stats["total_questions"] += 1
    stats["categories"].setdefault(category, {"correct": 0, "incorrect": 0, "times_shown": 0})
    stats["categories"][category]["times_shown"] += 1
    
    if is_correct:
        stats["categories"][category]["correct"] += 1
        stats["total_correct"] += 1
    else:
        stats["categories"][category]["incorrect"] += 1
        stats["total_incorrect"] += 1
    return stats

def rel_x(x_percent):  # Convert percentage to X coordinate
    return int(SCREEN_WIDTH * x_percent / 100)

def rel_y(y_percent):  # Convert percentage to Y coordinate
    return int(SCREEN_HEIGHT * y_percent / 100)


# State management
states = ['main', 'loading', 'result', 'menu']
popup_confirmed = False
PHOTO_FOLDER = "captured"
if not os.path.exists(PHOTO_FOLDER):
    os.makedirs(PHOTO_FOLDER)
captured_filename = "captured/loading.png"
lilcurrent_state = 'main'
user_guess = None
actual_category = None
correct_count = 0
total_guesses = 0
history = deque(maxlen=5)
state_start_time = pygame.time.get_ticks()
loading_page_time = 6000
result_page_time = 5000



'''
NEW IMPORTANT
'''
model = YOLO('YOLO_weight/v7_18_3/best.pt')  #v7_18_3/best.pt
classNames = ['shoe', 'paperBox', 'pastry', 
              'penPencil', 'milkCarton', 'cutlery', 
              'crumpledPaper', 'eggShell', 'glassBottle', 
              'plasticContainer','paperEnvelope', 'paperCup', 
              'fruit', 'noodlePasta', 'plasticLid', 
              'plasticMilkBottle', 'sandwich','tissueCore', 
              'vape', 'vegeScraps', 'glassJar', 
              'meat', 'book', 'plasticBottle', 
              'alCan', 'softPlastic']

formal_names = {
    'shoe': 'Shoe',
    'paperBox': 'Paper Box',
    'pastry': 'Pastry',
    'penPencil': 'Pen Pencil',
    'milkCarton': 'Milk Carton',
    'cutlery': 'Cutlery',
    'crumpledPaper': 'Crumpled Paper',
    'eggShell': 'Egg Shell',
    'glassBottle': 'Glass Bottle',
    'plasticContainer': 'Plastic Container',
    'paperEnvelope': 'Paper Envelope',
    'paperCup': 'Paper Cup',
    'fruit': 'Fruit',
    'noodlePasta': 'Noodle Pasta',
    'plasticLid': 'Plastic Lid',
    'plasticMilkBottle': 'Plastic Milk Bottle',
    'sandwich': 'Sandwich',
    'tissueCore': 'Tissue Core',
    'vape': 'Vape',
    'vegeScraps': 'Vege Scraps',
    'glassJar': 'Glass Jar',
    'meat': 'Meat',
    'book': 'Book',
    'plasticBottle': 'Plastic Bottle',
    'alCan': 'Aluminium Can',
    'softPlastic': 'Soft Plastic'
}


def find_item_category(json_file_path, item_to_find):
    try:
        # Open and read the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Iterate through each category in the JSON data
        for category, content in data.items():
            # Check if the item is in the "items" list of the current category
            if "items" in content and item_to_find in content["items"]:
                return category
        
        # If item is not found in any category
        return f"Item '{item_to_find}' not found in any category."

    except FileNotFoundError:
        return f"Error: The file '{json_file_path}' was not found."
    except json.JSONDecodeError:
        return "Error: Failed to decode the JSON file."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


def obtain_results_YOLO(image_path):
    parent_dir = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    img = cv2.imread(image_path)
    results = model(image_path)

    for i in results:
        boxes = i.boxes
        for box in boxes:
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf > 0.2:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                bbox = (x1, y1, w, h)
                cvzone.cornerRect(img, bbox)
                # Class name
                cls = int(box.cls[0])
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(40, y1)))
    
    output_path = os.path.join(parent_dir, "labeled", filename)
    # Ensure the "labeled" directory exists
    os.makedirs(os.path.join(parent_dir, "labeled"), exist_ok=True)
    cv2.imwrite(output_path, img)

    # Extract predictions and confidence values from the first result
    each_img_detected = results[0].boxes.cls.tolist()
    each_img_conf = results[0].boxes.conf.tolist()
    result_clss_name = [classNames[int(i)] for i in each_img_detected]

    return output_path, result_clss_name, each_img_conf


def choose_class(class_list, confidence_list):
    """
    Returns the final class based on majority vote weighted by confidence.
    If both lists are empty, defaults to "General Waste".
    
    Parameters:
      class_list (list of str): The predicted class names.
      confidence_list (list of float): The confidence scores corresponding to each prediction.
    
    Returns:
      str: The formal class name based on weighted vote, or "General Waste" if no predictions.
    """
    # Handle empty lists: return default category "General Waste"
    if not class_list and not confidence_list:
        return "General Waste"
    
    if len(class_list) != len(confidence_list):
        raise ValueError("Input lists must be of the same length.")
    
    # Sum the confidence scores for each class
    class_confidence = defaultdict(float)
    for cls, conf in zip(class_list, confidence_list):
        class_confidence[cls] += conf

    # Determine the class with the maximum total confidence
    max_class = max(class_confidence, key=class_confidence.get)
    # Return the formal name; if not found in mapping, return the raw class name
    return max_class


def get_recycling_category(max_class, recycling_data):
    """
    Returns the recycling category based on the predicted class.
    If max_class is None or not found, defaults to "General Waste".
    """
    if not max_class:
        return "General Waste"
    for category, info in recycling_data.items():
        if max_class in info.get("items", []):
            return category
    return "General Waste"



'''
END
'''



# UI/UX Tools
def draw_text(text, color, x, y, font_size=40, center=True, font_style=None):
    """Helper function to draw text on the screen."""
    font_txt = pygame.font.Font(font_style, font_size)
    text_surface = font_txt.render(text, True, color)
    text_rect = text_surface.get_rect(center=(x, y)) if center else (x, y)
    screen.blit(text_surface, text_rect)


def draw_trash_item_txt(text, color, x, y, font_size=40, center=True, item_font_name=None, special_style=None):
    if "Trash Item: " in text:
        prefix = "Trash Item: "
        random_item = text[len(prefix):]  # Extract the random_item part
    else:
        # If the text doesn't match the expected format, render it as is
        font_txt = pygame.font.Font(None, font_size)
        text_surface = font_txt.render(text, True, color)
        text_rect = text_surface.get_rect(center=(x, y)) if center else (x, y)
        screen.blit(text_surface, text_rect)
        return

    # Font for "Trash Item: " (default system font)
    font_txt = pygame.font.Font(special_style, font_size-7)
    # Font for random_item (e.g., Comic Sans MS or a custom font)
    item_font = pygame.font.SysFont(None, font_size)
    # If using a custom font file (e.g., AnimeAce.ttf), uncomment the line below
    # item_font = pygame.font.Font("AnimeAce.ttf", font_size)

    # Render the two parts separately
    prefix_surface = item_font.render(prefix, True, color)
    item_surface = font_txt.render(random_item, True, color)

    # Calculate the total width of the combined text
    total_width = prefix_surface.get_width() + item_surface.get_width()
    #print(item_surface.get_width())
    #print(prefix_surface.get_width()    )
    if center:
        # If centering, calculate the starting x position so the entire text is centered
        start_x = x - (total_width // 2)
        prefix_rect = prefix_surface.get_rect(topleft=(start_x, y - font_size // 2))
        item_rect = item_surface.get_rect(topleft=(start_x + prefix_surface.get_width(), y - font_size // 1.33))
    else:
        # If not centering, start at the given x position
        prefix_rect = prefix_surface.get_rect(topleft=(x, y))
        item_rect = item_surface.get_rect(topleft=(x + prefix_surface.get_width(), y))

    # Draw both parts on the screen
    screen.blit(prefix_surface, prefix_rect)
    screen.blit(item_surface, item_rect)

    
def draw_wrapped_text(text, color, x, y, font_size, max_width):
    """Draws text wrapped to a specified width."""
    font_path = 'font/WorkSans-Medium.ttf'
    font2 = pygame.font.Font(font_path, font_size)#pygame.font.SysFont(None, font_size)

    wrapped_lines = textwrap.wrap(text, width=max_width)
    line_height = font_size + 3

    for i, line in enumerate(wrapped_lines):
        text_surface = font2.render(line, True, color)
        text_rect = text_surface.get_rect(center=(x, y + i * line_height))
        screen.blit(text_surface, text_rect)

def create_rounded_mask(size, radius):
    """Create a rounded rectangle mask."""
    mask = pygame.Surface(size, pygame.SRCALPHA)  # Enable alpha channel
    rect = pygame.Rect(0, 0, *size)
    pygame.draw.rect(mask, (255, 255, 255, 255), rect, border_radius=radius)
    return mask

def draw_text_with_highlight(
    text, highlight_text, color, highlight_color, highlight_bg, x, y, 
    font_size=40, center=True, border_radius=10
):
    """Draws text with a highlighted portion that has a rounded background."""
    # Split the text into parts
    parts = text.split(highlight_text)
    
    # Base font for the normal text
    base_font = pygame.font.Font(None, font_size)
    
    # Highlight font
    highlight_font_size = int(font_size * 1.2)  # Make it bigger
    highlight_font = pygame.font.Font(None, highlight_font_size)
    
    # Render each part
    rendered_parts = [base_font.render(part, True, color) for part in parts]
    highlight_rendered = highlight_font.render(highlight_text, True, highlight_color)
    
    # Calculate positions
    total_width = sum(part.get_width() for part in rendered_parts) + highlight_rendered.get_width()
    total_height = max(part.get_height() for part in rendered_parts + [highlight_rendered])
    
    if center:
        x -= total_width // 2
        y -= total_height // 2
    
    # Draw normal text parts
    for part in rendered_parts:
        screen.blit(part, (x, y))
        x += part.get_width()
    
    # Draw highlighted text with rounded background
    highlight_width = highlight_rendered.get_width() + 10  # Add padding
    highlight_height = highlight_rendered.get_height() + 6  # Add padding
    highlight_rect = pygame.Rect(x - 5, y - 3, highlight_width, highlight_height)
    pygame.draw.rect(screen, highlight_bg, highlight_rect, border_radius=border_radius)
    screen.blit(highlight_rendered, (x, y))


def draw_text_with_shadow(text, color, x, y, font_size=45, shadow_color='#b3b2b2', shadow_offset=(3, 3), font_style=False):

    if font_style == False:
        text_font = pygame.font.SysFont(None, font_size)
    else:
        text_font = pygame.font.Font(font_style, font_size)
    
    # Draw shadow
    shadow_surface = text_font.render(text, True, shadow_color)
    shadow_rect = shadow_surface.get_rect(center=(x + shadow_offset[0], y + shadow_offset[1]))
    screen.blit(shadow_surface, shadow_rect)

    # Draw main text
    text_surface = text_font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(x, y))
    screen.blit(text_surface, text_rect)


def draw_rounded_button(screen, color, hover_color, x, y, width, height, radius, text, text_color, font_size=50):
    SHADOW_COLOR = '#dcdcdc'
    rect = pygame.Rect(x, y, width, height)
    # Check hover state
    mouse_pos = pygame.mouse.get_pos()
    hover = rect.collidepoint(mouse_pos)
    draw_color = hover_color if hover else color
    
    shadow_offset = 7
    shadow_rect = pygame.Rect(x + shadow_offset, y + shadow_offset, width, height)
    pygame.draw.rect(screen, SHADOW_COLOR, shadow_rect, border_radius=radius)
    
    # Draw button with rounded corners
    pygame.draw.rect(screen, draw_color, rect, border_radius=radius)
    # Draw text with dynamic font size
    #font_size = int(height * 0.5)  # Adjust multiplier (0.5) as needed
    font = pygame.font.SysFont(None, font_size)
    text_surface = font.render(text, True, text_color)
    text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
    screen.blit(text_surface, text_rect)


def draw_loading_bar_2(progress):
    width = 700  # Total width of the bar
    height = 75
    LIGHT_GRAY = (200, 200, 200)
    BAR_COLOR = (154, 155, 152)
    SHADOW_COLOR = '#dcdcdc'
    BAR_BG_COLOR = LIGHT_GRAY

    # Draw shadow
    shadow_offset = 9
    #x = rel_x(30.5)
    #y = rel_y(34)
    x = (SCREEN_WIDTH - width) // 2
    y = SCREEN_HEIGHT // 2 - 100
    
    pygame.draw.rect(screen, SHADOW_COLOR, (x + shadow_offset, y + shadow_offset, width, height), border_radius=20)
    
    # Draw background
    pygame.draw.rect(screen, BAR_BG_COLOR, (x, y, width, height), border_radius=20)
    
    # Draw progress bar with gradient effect
    fill_width = int((progress / 100) * width)
    gradient_color = (BAR_COLOR[0] - progress, BAR_COLOR[1] , BAR_COLOR[2]- progress)  # Dynamic color
    pygame.draw.rect(screen, gradient_color, (x, y, fill_width, height), border_radius=20)

def draw_progress_bar(surface, x, y, width, height, progress, color, border_radius=6):
    # Calculate the filled width
    fill_width = max(border_radius * 2, width * progress)  # Prevent width from being too small

    # Draw the progress portion with rounded edges
    progress_rect = pygame.Rect(x, y, fill_width, height)
    pygame.draw.rect(surface, color, progress_rect, border_radius=border_radius)

    # Draw circles on both ends to ensure roundness
    pygame.draw.circle(surface, color, (x + border_radius, y + height // 2), border_radius)
    pygame.draw.circle(surface, color, (x + fill_width - border_radius, y + height // 2), border_radius)


# Button positions and sizes
button_width, button_height = 260, 95
x_1 = rel_x(60) #x_1 = rel_x(45) #rel_x(50)
y_1 = rel_y(39) #y_1 = rel_y(30) #rel_y(30)
x_2 = rel_x(80) #x_2 = rel_x(62) #rel_x(72)
y_2 = rel_y(62) #y_2 = rel_y(48) #rel_y(54)


buttons = {
    'Mixed Recycling': (x_1, y_1),
    'Paper': (x_2, y_1),
    'Food Waste': (x_1, y_2),
    'General Waste': (x_2, y_2)
}
menu_button = pygame.Rect(rel_x(91.25), rel_y(6), 130, 80)
back_button_pos = (rel_x(2), rel_y(6))


'''START: CONGRATULATORY PAGE RELATED FUNCTIONS'''
# Ribbon and confetti setup
GOLD = (255, 215, 0)
RED2 = '#ff3e38'
GREEN2 = '#16f86b'
BLUE2 = '#377af8'
class Ribbon:
    def __init__(self):
        self.x = random.randint(0, SCREEN_WIDTH)
        self.y = random.randint(-100, -10)
        self.speed = random.randint(8, 11)
        self.color = random.choice([RED2, GREEN2, BLUE2, GOLD])
        self.width = random.randint(5, 15)
        self.height = random.randint(15, 30)

    def fall(self):
        self.y += self.speed
        if self.y > SCREEN_HEIGHT:
            self.y = random.randint(-100, -10)
            self.x = random.randint(0, SCREEN_WIDTH)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))

class Confetti:
    def __init__(self):
        self.x = random.randint(0, SCREEN_WIDTH)
        self.y = random.randint(-100, -10)
        self.speed = random.randint(6, 10)
        self.radius = random.randint(3, 6)
        self.color = random.choice([RED2, GREEN2, BLUE2, GOLD])

    def fall(self):
        self.y += self.speed
        if self.y > SCREEN_HEIGHT:
            self.y = random.randint(-100, -10)
            self.x = random.randint(0, SCREEN_WIDTH)

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)


NUM_RIBB = np.random.choice(np.arange(90, 130))
ribbons = [Ribbon() for _ in range(NUM_RIBB)]

NUM_CONFET = np.random.choice(np.arange(120, 160))
confetti_list = [Confetti() for _ in range(NUM_CONFET)]

LIGHT_BLUE = (242, 255, 255)
LIGHT_BLUE2 = (13, 251, 255)

class Raindrop:
    def __init__(self):
        self.x = random.randint(0, SCREEN_WIDTH)
        self.y = random.randint(-SCREEN_HEIGHT, 0)
        self.speed = random.uniform(8, 11)
        self.color = random.choice([LIGHT_BLUE, LIGHT_BLUE2])

    def fall(self):
        self.y += self.speed
        if self.y > SCREEN_HEIGHT:  # Reset raindrop position
            self.y = random.randint(-SCREEN_HEIGHT, 0)
            self.x = random.randint(0, SCREEN_WIDTH)

    def draw(self, screen):
        pygame.draw.line(screen, self.color, (self.x, self.y), (self.x, self.y + 5), 1)

NUM_DROPS = np.random.choice(np.arange(130, 180))
raindrops = [Raindrop() for _ in range(NUM_DROPS)]

def simulate_ai_classification():
    category = random.choice(list(trash_list.keys()))
    random_item = random.choice(trash_list[category])["name"]
    return category, random_item

def get_random_fact_or_item(facts, category, selection_type='fact'):
    if category not in facts:
        return f"Category '{category}' not found. Available categories: {', '.join(facts.keys())}"
    
    if selection_type not in ['fact', 'item']:
        return "Invalid selection type. Choose 'fact' or 'item'."

    key = 'facts' if selection_type == 'fact' else 'items'
    
    if not facts[category][key]:
        return f"No {selection_type}s available for category '{category}'."
    
    return random.choice(facts[category][key])

'''START: MATPLOTLIB ON MENU PAGE'''
# Colors and styles
#category: ["Mixed Recycling", "Paper", "Food Waste", "General Waste"]

def plot_class_stats(ax, category, correct, incorrect, color_corr, color_incorr, hatch):
    """Plots a single donut chart on the given axes (ax)."""
    # Handle zero values
    if correct == 0 and incorrect == 0:
        # Skip plotting if both values are zero
        ax.text(0, 0, "No Data", ha='center', fontsize=14, weight='bold', color='gray')
        ax.set_aspect('equal')
        ax.axis('off')
        return

    # Add a small epsilon to avoid division by zero
    epsilon = 0.001
    correct = max(correct, epsilon)
    incorrect = max(incorrect, epsilon)

    # Create donut chart (correct vs incorrect)
    wedges, _ = ax.pie([correct, incorrect], radius=1.2, colors=[color_corr, color_incorr],
                        startangle=90, wedgeprops={'width': 0.3, 'edgecolor': 'white'})

    # Display 0 instead of 0.001 in labels
    def format_value(value):
        return int(value) if abs(value - epsilon) < 1e-5 else int(round(value))
    
    # Apply hatching for incorrect responses
    wedges[1].set_hatch(hatch)
    #wedges[1].set_facecolor(self.bckg_color)

    # Fun and engaging labels
    ax.text(0, 0.0, category, ha='center', fontsize=26, weight='bold', color=color_corr)
    
    if correct >= incorrect:
        ax.text(0, 0.4, f"{format_value(correct)} ✓", ha='center', fontsize=33, color='#0B9E0D', weight='bold')
        ax.text(0, -0.5, f"{format_value(incorrect)} ✗", ha='center', fontsize=33, color='#e94141', weight='bold')
    else:
        ax.text(0, -0.5, f"{format_value(correct)} ✓", ha='center', fontsize=33, color='#0B9E0D', weight='bold')
        ax.text(0, 0.4, f"{format_value(incorrect)} ✗", ha='center', fontsize=33, color='#e94141', weight='bold')

    ax.set_aspect('equal')
    ax.axis('off')

def plot_all_stats(stats):
    bckg_color = '#ffffff' 

    colors_correct = ['#60b0e5', '#41d555', '#a867b9', '#ec655a']
    colors_incorrect = ['#E0E0E0'] * len(categories)  # Gray for incorrect sections
    hatch_patterns = ['*', '\\\\', '..', '-']  # Patterns for incorrect sections
    correct_list = [stats["categories"][category]["correct"] for category in stats["categories"]]
    incorrect_list = [stats["categories"][category]["incorrect"] for category in stats["categories"]]

    """Plot all categories in a 2x2 subplot layout and save the plot to a file."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Set custom background color for the entire figure
    fig.patch.set_facecolor(bckg_color)  # Light gray background for the entire plot


    # Plot each category in its respective position
    for i, ax in enumerate(axes.flat):
        plot_class_stats(ax, categories[i], correct_list[i], incorrect_list[i], 
                                colors_correct[i], colors_incorrect[i], hatch_patterns[i])

    # Adjust layout and save the plot
    plt.tight_layout()
    # Render the figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)

    # Close the figure to free up memory
    plt.close(fig)

    # Load the buffer into a Pygame surface
    surface = pygame.image.load(buf)
    return surface

def plot_single_accuracy(stats):
    bckg_color = '#ffffff'
    
    # Calculate total correct and incorrect across all categories
    total_correct = sum(stats["categories"][category]["correct"] for category in stats["categories"])
    total_incorrect = sum(stats["categories"][category]["incorrect"] for category in stats["categories"])
    accuracy = total_correct / (total_correct + total_incorrect) * 100 if (total_correct + total_incorrect) > 0 else 0

    # Create figure with single subplot
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(bckg_color)
    
    # Handle zero values
    if total_correct == 0 and total_incorrect == 0:
        ax.text(0, 0, "No Data", ha='center', fontsize=26, weight='bold', color='gray')
        ax.set_aspect('equal')
        ax.axis('off')
    else:
        # Add small epsilon to avoid division by zero
        epsilon = 0.001
        correct = max(total_correct, epsilon)
        incorrect = max(total_incorrect, epsilon)

        # Create donut chart
        wedges, _ = ax.pie([correct, incorrect], 
                          radius=1.2,
                          colors=['#0B9E0D', '#e94141'],  # Green for correct, red for incorrect
                          startangle=90,
                          wedgeprops={'width': 0.3, 'edgecolor': 'white'})

        # Add hatching to incorrect portion
        wedges[1].set_hatch('||')
        
        # Central accuracy percentage
        ax.text(0, 0.1, f"{accuracy:.1f}%", ha='center', fontsize=42, color='#2c3e50', weight='bold')
        ax.text(0, -0.15, "ACCURACY", ha='center', fontsize=28, color='#2c3e50', weight='bold')

        # Bottom labels
        ax.text(0, -0.4, f"Correct: {int(total_correct)}", 
               ha='center', fontsize=21, color='#0B9E0D', weight='bold')
        ax.text(0, -0.55, f"Incorrect: {int(total_incorrect)}", 
               ha='center', fontsize=21, color='#e94141', weight='bold')

    ax.set_aspect('equal')
    ax.axis('off')

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    plt.close(fig)

    # Load into pygame surface
    surface = pygame.image.load(buf)
    return surface
'''END: MATPLOTLIB ON MENU PAGE'''

# Add this function to draw the settings/menu page
def print_stats_on_menu_page(stats):
    screen.fill((255,255,255))
    #(rel_x(40), rel_y(7))
    
    draw_text(f'View Statistic', "#242d33", rel_x(50), rel_y(7), font_size=85, font_style=lilita_one_path)  #font_style
    BLACK = (0, 0, 0)
    #STATS_LEFT_MARGIN = 55
    A, B, C, D = stats['sessions'], stats['total_questions'], stats['total_correct'], round(stats['total_correct']*100/(stats['total_questions'] or 1),1)
    LIST_X = [B,C]
    #print(LIST_X)

    #UP_X = SCREEN_WIDTH // 6.8
    #UP_X_R  = rel_x(60)
    #UP_X_PB = rel_x(22.8)

    UP_X = SCREEN_WIDTH // 6.4
    UP_X_R  = SCREEN_WIDTH - (SCREEN_WIDTH // 4.9)
    UP_X_PB = SCREEN_WIDTH // 3.85

    #draw_text(f"Total Sessions", BLACK, UP_X, (SCREEN_HEIGHT // 2 - 370), font_size=30, center=False) #370, 320, 270, 220
    draw_text(f"Total Questions", BLACK, UP_X, (SCREEN_HEIGHT // 2 - 370), font_size=30, center=False)
    draw_text(f"Correct Answers", BLACK, UP_X, (SCREEN_HEIGHT // 2 - 320), font_size=30, center=False)
    draw_text(f"Overall Accuracy", BLACK, UP_X, (SCREEN_HEIGHT // 2 - 270), font_size=30, center=False)
    
    #draw_progress_bar(screen, UP_X_PB, (SCREEN_HEIGHT // 2 - 370), 10, 18, A*100/(max(LIST_X) or 1), '#f7568e')
    draw_progress_bar(screen, UP_X_PB, (SCREEN_HEIGHT // 2 - 370), 10, 18, B*100/(max(LIST_X) or 1), '#776eec')
    draw_progress_bar(screen, UP_X_PB, (SCREEN_HEIGHT // 2 - 320), 10, 18, C*100/(max(LIST_X) or 1), '#6e9cb7')
    draw_progress_bar(screen, UP_X_PB, (SCREEN_HEIGHT // 2 - 270), 10, 18, D, '#ecb67e')

    #draw_text(f"{A}", BLACK, UP_X_R, (SCREEN_HEIGHT // 2 - 370), font_size=30, center=False)
    draw_text(f"{B}", BLACK, UP_X_R, (SCREEN_HEIGHT // 2 - 370), font_size=30, center=False)
    draw_text(f"{C}", BLACK, UP_X_R, (SCREEN_HEIGHT // 2 - 320), font_size=30, center=False)
    draw_text(f"{D}%", BLACK, UP_X_R, (SCREEN_HEIGHT // 2 - 270), font_size=30, center=False)

    # Create the plot surface
    plot_surface = plot_all_stats(stats)
    scaled_width = plot_surface.get_width() // 1.5 #//1.9
    scaled_height = plot_surface.get_height() // 1.5 #//1.9
    scaled_surface = pygame.transform.smoothscale(plot_surface, (scaled_width, scaled_height))

    # Display the plot surface at the center of the screen
    screen.blit(scaled_surface, (SCREEN_WIDTH//2 + 90, SCREEN_HEIGHT//2 - 190)) #screen.blit(scaled_surface, (rel_x(40), rel_y(28)))   ### change line 848 also, (SCREEN_WIDTH//2 + 50, SCREEN_HEIGHT//2 - 50)
    
    plot_acc_surface = plot_single_accuracy(stats)
    scaled_acc_width = plot_acc_surface.get_width() // 0.92 #// 1.2  # Adjust scale factor as needed
    scaled_acc_height = plot_acc_surface.get_height() // 0.92 #// 1.2
    scaled_acc_surface = pygame.transform.smoothscale(plot_acc_surface, (scaled_acc_width, scaled_acc_height))

    screen.blit(scaled_acc_surface, (195, SCREEN_HEIGHT//2 - 200)) #screen.blit(scaled_acc_surface, (rel_x(8), rel_y(28)))


def sorter_for_esp32(ans):
    if "General" in ans:
        return int(0)
    elif "Paper" in ans:
        return int(1)
    elif "Mixed" in ans:
        return int(2)
    elif "Food" in ans:
        return int(3)
    else:
        return int(0)
            
'''ser = serial.Serial(
              
    port='/dev/ttyAMA0',
    baudrate = 115200,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)'''

    
'''MAIN START FUNCTION'''
if __name__ == "__main__":

    CAM_W, CAM_H = 214*4, 214*3  #4:3
    
    '''NEW IMPORT'''
    recognized_ITEM = None
    category_random = None
    pup_up_show_take_photo = False
    '''END NEW IMPORT'''

    '''NEW IMPORT 2'''
    prohibited_popup_active = False #must be False  # Flag to control the prohibited items pop-up
    prohibited_popup_start_time = 0  # Time when the pop-up starts
    '''END NEW IMPORT 2'''
    
    running = True
    current_state = 'main'
    category_random, _ = simulate_ai_classification()  # Generate once at startcurrent_state = 'main'
    factual = random.choice(trash_facts[category_random]['facts'])
    
    circle_x = random.randint(300, SCREEN_WIDTH//2)
    circle_y = random.randint(300, SCREEN_HEIGHT - 300)
    circle_dx, circle_dy = 7, 7
    color_index = 0
    GREEN_GRADIENT = [(34, 177, 76), (46, 204, 113), (60, 220, 133), (85, 237, 153), (110, 255, 173)]

    NUM_FALL = np.random.choice(np.arange(10, 19))
    falling_objects = [{"x": random.randint(0, SCREEN_WIDTH), "y": random.randint(-50, -10)} for _ in range(NUM_FALL)]
    falling_speed = 7.5  # Speed at which objects fall
    curr_stats = load_stats()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if current_state == 'main':
                    if not popup_confirmed:
                        # Handle pop-up buttons
                        if yes_button_rect.collidepoint(mouse_pos):
                            '''START: CAMERA SETUP VARIABLES'''
                            cap = cv2.VideoCapture(0)  # Use default camera (0)
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    
                            cap.set(3, CAM_W)  # Width of camera frame
                            cap.set(4, CAM_H)
                            '''END: CAMERA SETUP VARIABLES'''

                            ret, frame = cap.read()
                            # OpenCV Camera Setup
                            
                            if ret:
                                # Generate a unique filename using a timestamp
                                timestamp = int(time.time())
                                filename = f"photo_{timestamp}.jpg"
                                filepath = os.path.join(PHOTO_FOLDER, filename)
                                # Save the photo
                                cv2.imwrite(filepath, frame)
                                # Store the filename
                                #captured_filename = filepath
                                #print(captured_filename)
                                # Confirm the popup
                                popup_confirmed = True
                                #print(f"Photo saved as {captured_filename}")
                                cap.release()
                                '''NEW IMPORT'''
                                captured_filename, ans, ans2 = obtain_results_YOLO(filepath)
                                recognized_item = choose_class(ans, ans2)
                                recognized_category = find_item_category(RECYCLE_FILE, recognized_item)
                                recognized_ITEM = formal_names.get(recognized_item)
                                print(recognized_ITEM, recognized_category)
                                
                                category_random = recognized_category
                                #factual = random.choice(trash_facts[recognized_category]['facts'])

                                '''END NEW IMPORT'''

                                # Check if the recognized category is "Prohibited Items"
                                if recognized_category == "Prohibited Item":
                                    prohibited_popup_active = True
                                    print('Ohhh Nooo!!')
                                    prohibited_popup_start_time = pygame.time.get_ticks()
                                else:
                                    # Normal operation for non-prohibited items
                                    category_random = recognized_category
                                    factual = random.choice(trash_facts[recognized_category]['facts'])
                                    bin_index = sorter_for_esp32(recognized_category)
                                    
                            else:
                                pass
                                
                    elif not prohibited_popup_active: 
                        if menu_button.collidepoint(mouse_pos):
                            current_state = 'menu'
                            state_start_time = pygame.time.get_ticks()
                        else:
                            for category, pos in buttons.items():
                                button_rect = pygame.Rect(pos[0], pos[1], button_width, button_height)
                                if button_rect.collidepoint(mouse_pos):
                                    ser.write((str(bin_index)+'/n').encode())
                                    user_guess = category
                                    current_state = 'loading'
                                    print(bin_index, 'success')
                                    state_start_time = pygame.time.get_ticks()
                                    break
                elif current_state == 'result':
                    back_button = pygame.Rect(back_button_pos[0], back_button_pos[1], 130, 80)
                    if back_button.collidepoint(mouse_pos):
                        current_state = 'main'  
                        category_random, random_item = simulate_ai_classification()  # Generate new trash item **only here**
                        factual = random.choice(trash_facts[category_random]['facts'])
                        state_start_time = pygame.time.get_ticks()
                elif current_state == 'menu':
                    back_button = pygame.Rect(back_button_pos[0], back_button_pos[1], 130, 80)
                    if back_button.collidepoint(mouse_pos):
                        current_state = 'main'
                        state_start_time = pygame.time.get_ticks()

        screen.fill('#fcf8f5')

        if current_state == 'main':

            vert_x = rel_x(28)
            image_rect_twoTouch = scaled_twotouch.get_rect(center=(rel_x(5.5), rel_y(10)))
            screen.blit(scaled_twotouch, image_rect_twoTouch)

            image_rect_school = scaled_school.get_rect(center=(rel_x(50), rel_y(9)))
            screen.blit(scaled_school, image_rect_school)

            draw_text(f'Waste Item:', "#242d33", vert_x, rel_y(21), font_size=40, font_style=lilita_one_path)

            draw_text_with_highlight(
                        text='Dual Finger Activation',
                        highlight_text= 'Dual Finger Activation',
                        color=(255, 255, 255),
                        highlight_color=(255,255,255),
                        highlight_bg='#ef6767',
                        x=rel_x(14),
                        y=rel_y(10.3),
                        font_size=25,
                        center=True,
                        border_radius=8
                    )
            
            # Draw OpenCV frame (existing logic)
            #ret, frame = cap.read()
            
            '''if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (CAM_W, CAM_H))
                #frame = cv2.flip(frame, 1)
                frame_surface = pygame.image.frombuffer(frame.tobytes(), (CAM_W, CAM_H), "RGB")
                mask = create_rounded_mask((CAM_W, CAM_H), 15)  #corner radius
                temp_surface = pygame.Surface((CAM_W, CAM_H), pygame.SRCALPHA)
                temp_surface.blit(frame_surface, (0, 0))
                temp_surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
                screen.blit(temp_surface, (rel_x(6.04), rel_y(24)))

            else:'''
            #show_processing_text = False
            #print(captured_filename)
            placeholder_image = pygame.image.load(captured_filename) 
            # Resize the image to match the webcam dimensions
            placeholder_image = pygame.transform.smoothscale(placeholder_image, (CAM_W, CAM_H))

            frame_surface = placeholder_image  # Directly use the resized image
            mask = create_rounded_mask((CAM_W, CAM_H), 15)  # Corner radius
            temp_surface = pygame.Surface((CAM_W, CAM_H), pygame.SRCALPHA)
            temp_surface.blit(frame_surface, (0, 0))
            temp_surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            screen.blit(temp_surface, (rel_x(6.04), rel_y(24)))




            #draw_text_with_shadow("What's your Guess", "#2e2d2d", rel_x(60) , rel_y(22), font_size=50)
            draw_text_with_shadow("What's your Guess?", "#2e2d2d", SCREEN_WIDTH //1.28 , SCREEN_HEIGHT // 3.2, font_size=60)

            # Draw buttons
            class_button_width, class_button_height = 300, 120
            for category, pos in buttons.items():
                base_color = category_colors[category]
                # Lighten color for hover (add 20 to each component, cap at 255)
                hovered = hover_color[category]
                #draw_rounded_button(screen, base_color, hovered, pos[0], pos[1], 20, category, (0, 0, 0))
                draw_rounded_button(screen, base_color, hovered, pos[0], pos[1], class_button_width, class_button_height, 20, category, (0, 0, 0))
            
            # Menu button
            mouse_pos = pygame.mouse.get_pos()
            hover = menu_button.collidepoint(mouse_pos)
            draw_color = (93, 93, 93) if hover else (0, 0, 0)
            pygame.draw.rect(screen, draw_color, menu_button)
            font = pygame.font.SysFont(None, 24)
            menu_text = font.render('Menu', True, (255, 255, 255))
            menu_text_rect = menu_text.get_rect(center=menu_button.center)
            screen.blit(menu_text, menu_text_rect)

            # Add pop-up if not confirmed
            if not popup_confirmed:
                # Draw semi-transparent overlay
                overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 128))  # 128 alpha for half-transparent black
                screen.blit(overlay, (0, 0))

                # Draw pop-up rectangle
                popup_rect = pygame.Rect((SCREEN_WIDTH-650) // 2, (SCREEN_HEIGHT - 200) // 2, 650, 230)
                pygame.draw.rect(screen, (255, 255, 255), popup_rect, border_radius=20)

                # Draw pop-up text
                draw_text("Have you placed your item into the bin?", (0, 0, 0), popup_rect.centerx, popup_rect.y + 50, font_size=40)
                draw_text("Taking and Processing Photo...", (0, 0, 0), popup_rect.centerx, popup_rect.y + 90, font_size=30)
                # Draw Yes and No buttons
                yes_button_rect = pygame.Rect(popup_rect.x + 253, popup_rect.y + 120, 100, 130)
                #no_button_rect = pygame.Rect(popup_rect.x + 225, popup_rect.y + 120, 100, 50)
                draw_rounded_button(screen, (109, 198, 100), (155, 220, 149), yes_button_rect.x, yes_button_rect.y, 150, 80, 10, "Yes", (255, 255, 255), font_size=40)
                #draw_rounded_button(screen, (255, 0, 0), (255, 50, 50), no_button_rect.x, no_button_rect.y, 100, 50, 10, "No", (255, 255, 255))

                    
                
            elif prohibited_popup_active:
                # Draw semi-transparent overlay
                overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 128))
                # Draw pop-up rectangle
                popup_rect = pygame.Rect((SCREEN_WIDTH-790) // 2, (SCREEN_HEIGHT - 400) // 2, 790, 430)
                popup_rect_shadow = pygame.Rect(((SCREEN_WIDTH-790) // 2)+7, ((SCREEN_HEIGHT - 400) // 2)+7, 790, 430)
                pygame.draw.rect(screen, '#c58080', popup_rect_shadow, border_radius=30)
                pygame.draw.rect(screen, '#f7e1e1', popup_rect, border_radius=30)
                
                # Calculate remaining time
                elapsed = (pygame.time.get_ticks() - prohibited_popup_start_time) / 1000  # Convert to seconds
                remaining = 10 - elapsed
                
                if remaining > 0:
                    # Draw pop-up text with countdown
                    draw_text("WARNING !!!", (255, 0, 0), popup_rect.centerx, popup_rect.y + 60, font_size=100)
                    draw_text("Identified Prohibited Items!", (255, 0, 0), popup_rect.centerx, popup_rect.y + 130, font_size=75)
                    draw_text("Please remove it...", (0, 0, 0), popup_rect.centerx, popup_rect.y + 210, font_size=75)
                    draw_text(f"{int(remaining)}", (0, 0, 0), popup_rect.centerx, popup_rect.y + 350, font_size=200)
                else:
                    # After 10 seconds, reset pop-up flags and proceed
                    prohibited_popup_active = False
                    popup_confirmed = False
                    # Here, "proceed to normal operation" means continuing after the warning
                    category_random = recognized_category
                    factual = random.choice(trash_facts[recognized_category]['facts'])


            
        elif current_state == 'loading':

            screen.fill('#fcf8f5')
            circle_radius = 35
            circle_boundary_x = 25
            circle_boundary_y = 25
            pygame.draw.circle(screen, GREEN_GRADIENT[color_index], (circle_x, circle_y), circle_radius)
            circle_x += circle_dx
            circle_y += circle_dy



            # Bounce logic and color change
            if circle_x <= circle_boundary_x or circle_x >= SCREEN_WIDTH - circle_boundary_x:
                circle_dx *= -1
                color_index = (color_index + 1) % len(GREEN_GRADIENT)
            if circle_y <= circle_boundary_y or circle_y >= SCREEN_HEIGHT - circle_boundary_y:
                circle_dy *= -1
                color_index = (color_index + 1) % len(GREEN_GRADIENT)

            # --- Animation: Falling Objects ---
            for obj in falling_objects:
                pygame.draw.circle(screen, '#517ede', (obj["x"], obj["y"]), 6.5)
                obj["y"] += falling_speed
                if obj["y"] > (SCREEN_HEIGHT):
                    obj["y"] = random.randint(-50, -10)
                    obj["x"] = random.randint(0, SCREEN_WIDTH)
            
            font_path = "font/PixelifySans-Regular.ttf"
            draw_text_with_shadow("Did You Know!", "#2e2d2d", rel_x(50), rel_y(64), font_size=45, font_style=font_path)
            draw_wrapped_text(factual, '#2e2d2d', rel_x(50), rel_y(70), font_size=35, max_width=35)
            font = pygame.font.SysFont(None, 160)
            text = font.render('Processing...', True, (0, 0, 0))
            text_rect = text.get_rect(center=(rel_x(50), rel_y(33)))
            screen.blit(text, text_rect)
            elapsed_time = pygame.time.get_ticks() - state_start_time
            progress = (elapsed_time / loading_page_time) * 100
            draw_loading_bar_2(progress)
            if elapsed_time >= loading_page_time:
                actual_category, _ = simulate_ai_classification()  # Don't change random_item here
                actual_category = category_random
                is_correct = user_guess == actual_category
                
                curr_stats = update_stats(actual_category, is_correct)
                save_stats(curr_stats)
                #print(stats)
                feedback = "Congratulations!" if is_correct else "Oh ohh... try that again"
                total_guesses += 1
                history.append((user_guess, actual_category, feedback))
                current_state = 'result'
                state_start_time = pygame.time.get_ticks()

            # Update the display
            pygame.display.flip()
            clock.tick(60)  # Limit to 60 FPS
            '''END: INFO PAGE RELATED FUNCTIONS'''

        elif current_state == 'result':

            '''mouse_pos = pygame.mouse.get_pos()
            font = pygame.font.SysFont(None, 32)
            text = font.render(feedback, True, (0, 0, 0))
            text_rect = text.get_rect(center=(360, 240))
            back_text = font.render('Back', True, (255, 255, 255))
            back_text_rect = back_text.get_rect(center=back_button.center)'''

            if feedback == "Congratulations!":
                screen.fill('#fcf8f5')
                #print(SCREEN_HEIGHT)
                
                image_rect = congrat_image.get_rect(center=(rel_x(50), rel_y(54)))
                screen.blit(congrat_image, image_rect)
                correct_count += 1  # Increment correct count for "congrats"
                for ribbon in ribbons:
                    ribbon.fall()
                    ribbon.draw(screen)
                for confetti in confetti_list:
                    confetti.fall()
                    confetti.draw(screen)
                
                draw_text_with_shadow("Congratulations!", "#2e2d2d", rel_x(50), rel_y(22), font_size=120)

            elif feedback ==  "Oh ohh... try that again":  # If it's a normal result, apply the rain effect
                screen.fill((36, 36, 36))
                # Draw dark overlay
                for drop in raindrops:
                    drop.fall()
                    drop.draw(screen)
                
                draw_text_with_shadow("Uh Ohhh...", (200, 0, 0), rel_x(51), rel_y(29), shadow_color='#ed5959', font_size=75, font_style=press_start_2p_path)
                draw_text_with_highlight(
                        text=f'It is {category_random}',
                        highlight_text=category_random,
                        color=(255, 255, 255),
                        highlight_color=(255,255,255),
                        highlight_bg=category_colors[category_random],
                        x=rel_x(50),
                        y=rel_y(47),
                        font_size=120,
                        center=True,
                        border_radius=13
                    )
                image_rect_sad = sadden_image.get_rect(center=(240, SCREEN_HEIGHT-230))
                screen.blit(sadden_image, image_rect_sad)
                
                overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA) 
                overlay.fill((0, 0, 0, 110) )
                screen.blit(overlay, (0, 0))

                
                
                
            if pygame.time.get_ticks() - state_start_time >= result_page_time:
                current_state = 'main'
                popup_confirmed = False  # Reset pop-up
                state_start_time = pygame.time.get_ticks()
                category_random, random_item = simulate_ai_classification()  # Generate new trash item when returning to main
                factual = random.choice(trash_facts[category_random]['facts'])

        elif current_state == 'menu':
            #
            font = pygame.font.SysFont(None, 24)
            #stats_text = f"Correct: {correct_count}/{total_guesses}"
            #stats = font.render(stats_text, True, (0, 0, 0))
            #stats_rect = stats.get_rect(center=(400, 100))
            #screen.blit(stats, stats_rect)
            #history_text = "History (last 5):"
            #history_label = font.render(history_text, True, (0, 0, 0))
            #history_rect = history_label.get_rect(center=(400, 150))
            #screen.blit(history_label, history_rect)
            #y = 180
            '''for entry in list(history)[-5:]:
                guess, actual, feedback = entry
                line = f"Guess: {guess}, Actual: {actual}, {feedback}"
                line_text = font.render(line, True, (0, 0, 0))
                line_rect = line_text.get_rect(left=50, top=y)
                screen.blit(line_text, line_rect)
                y += 20
            '''

            print_stats_on_menu_page(curr_stats)
            mouse_pos2 = pygame.mouse.get_pos()
            back_button = pygame.Rect(back_button_pos[0], back_button_pos[1], 130, 80)
            hover2 = back_button.collidepoint(mouse_pos2)
            draw_color_back = (93, 93, 93) if hover2 else (0, 0, 0)
            pygame.draw.rect(screen, draw_color_back, back_button)
            back_text = font.render('Back', True, (255, 255, 255))
            back_text_rect = back_text.get_rect(center=back_button.center)
            screen.blit(back_text, back_text_rect)


        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

