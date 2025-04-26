import numpy as np
from collections import defaultdict
from PIL import Image
import json
import os
import re
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

all_list = ['trunk', 'hoodie', 'railroad track', 'beach', 'bone', 'tag', 'water', 'baseball player', 'toy', 'thong', 'angelfish',
 'helm', 'trailer', 'owl', 'man', 'clothing', 'rack', 'bag', 'pearl', 'tank', 'blazer', 'ground', 'tugboat', 'tie',
 'fire ball', 'skier', 'excavator', 'bike', 'cage', 'pad', 'fingernail', 'center', 'grill', 'fire', 'hat', 'baby',
 'landing gear', 'tunnel', 'eel', 'bale', 'sand', 'lobby', 'student', 'bison', 'mouse', 'strawberry', 'chihuahua',
 'hamburger', 'tusk', 'nose', 'tug boat', 'necklace', 'crowd', 'tv', 'glasses', 'goat', 'car', 'fork', 'adult',
 'blanket', 'rv', 'subway', 'construction site', 'card', 'surfer', 'arm', 'redwood', 'cloth', 'fall', 'sun', 'mouth',
 'golf ball', 'event', 'ski', 'ball', 'edge', 'garage', 'leaf', 'cake', 'tank top', 'pirate', 'airport', 'donut',
 'kick', 'plate', 'step', 'pose', 'wrist', 'house', 'door', 'wire', 'cap', 'hair', 'boulder', 'bridge', 'pant',
 'scissor', 'court', 'phone', 'soccer ball', 'belt', 'color', 'art', 'sign', 'sailboat', 'headphone', 'woman', 'salad',
 'bird', 'coffee cup', 'background', 'basketball', 'fighter jet', 'wheelchair', 'railroad', 'tray', 'cruise ship',
 'airplane', 'substance', 'position', 'cheek', 'slice', 'fireplace', 'column', 'parking lot', 'paintbrush',
 'whale shark', 'stick', 'mask', 'field', 'cupcake', 'truck', 'dock', 'dirt', 'mirror', 'tarp', 'bowl', 'firework',
 'test tube', 'face', 'leather', 'dolphin', 'horse', 'bear', 'desk', 'back', 'hospital', 'printer', 'spoon', 'exercise',
 'character', 'frisbee', 'screen', 'doorway', 'tennis ball', 'combine', 'tree', 'pencil', 'palm tree', 'microphone',
 'valley', 'bra', 'drawing', 'meal', 'rock', 'deer', 'hill', 'coral reef', 'jetliner', 'brain', 'microscope', 'street',
 'server rack', 'stocking', 'harvester', 'skyline', 'jacket', 'ship', 'motion', 'cell phone', 'knee', 'branch', 'bed',
 'floor', 'swamp', 'lab coat', 'stack', 'air', 'brush', 'astronaut', 'locker', 'anemone', 'town', 'reef', 'overall',
 'push up', 'collar', 'breakfast', 'hand', 'cherry blossom', 'roller coaster', 'road', 'light', 'mustache', 'rubble',
 'dreadlock', 'shirt', 'steering wheel', 'coral', 'cow', 'weed', 'apple', 'gymnasium', 'child', 'bull', 'bun', 'bow',
 'panda cub', 'instrument', 'leg', 'base', 'city', 'revolver', 'legging', 'doll', 'cart', 'vegetable', 'fountain',
 'pen', 'dog', 'jean', 'game', 'sheep', 'paw', 'whale', 'sunflower', 'shuttle', 'golf', 'lipstick', 'seaweed', 'ice',
 'antelope', 'living room', 'writing', 'hole', 'forest', 'tablet', 'paint', 'gazelle', 'bouquet', 'mangrove', 'camera',
 'harbor', 'panda', 'path', 'rapid', 'train', 'compass', 'sky', 'woods', 'alga', 'turkey', 'box', 'number', 'dumbbell',
 'stingray', 'surface', 'statue', 'vest', 'room', 'umbrella', 'balloon', 'precinct', 'nurse', 'fish', 'bat', 'beak',
 'shark', 'duck', 'short', 'head', 'painting', 'karate', 'ice cream', 'vehicle', 'labrador retriever', 'speedometer',
 'earth', 'guitar', 'machine', 'lake', 'dinner', 'pillar', 'blossom', 'shotgun', 'eye', 'range', 'world', 'golf course',
 'shack', 'night', 'aquarium', 'counter top', 'couch', 'platter', 'bamboo', 'rose', 'liquid', 'iguana', 't - shirt',
 'scuba diver', 'roof', 'shower', 'cattle', 'darkness', 'text', 'boat', 'tongue', 'fishing rod', 'juice', 'ocean',
 'sweater', 'gun', 'mud', 'stable', 'cloud', 'shoulder', 'people', 'chicken', 'baseball field', 'melon', 'wing',
 'hillside', 'paper', 'candle', 'housewife', 'wood', 'sling', 'apron', 'chair', 'station', 'finger', 'flame',
 'comforter', 'push', 'dark', 'croissant', 'puppy', 'body', 'bridle', 'coffee maker', 'clock', 'plant', 'site', 'grass',
 'skirt', 'syringe', 'egg', 'something', 'dresser', 'liberty', 'cliff', 'counter', 'suv', 'neck', 'market', 'towel',
 'island', 'cigarette', 'sweatshirt', 'backpack', 'bush', 'yard', 'panel', 'hut', 'monkey', 'zoo', 'book', 'toilet',
 'lip', 'butterfly', 'tail', 'elephant', 'shop', 'dress', 'jeep', 'explosion', 'beard', 'word', 'river', 'shelf',
 'rifle', 'shore', 'stream', 'stomach', 'area', 'motorcycle', 'row', 'badge', 'mother', 'stand', 'piano', 'luggage',
 'cup', 'track', 'hallway', 'wine', 'crane', 'television', 'sponge', 'wind', 'mountain', 'parrot fish', 'golf club',
 'bill', 'girl', 'wheat', 'suspender', 'rod', 'sink', 'family', 'snow', 'pastry', 'hallows', 'stair', 'person',
 'player', 'arrow', 'animal', 'tooth', 'father', 'table', 'slope', 'space station', 'parrotfish', 'combine harvester',
 'boy', 'elevator', 'hammock', 'video', 'outfit', 'bench', 'top', 'bunker', 'bottle', 'server', 'rug', 'grain', 'tower',
 'image', 'van', 'buffalo', 'restaurant', 'banana', 'soccer', 'jellyfish', 'penguin', 'heel', 'knife', 'gear', 'coffee',
 'gravel', 'ice cream cone', 'smile', 'park', 'object', 'dollar', 'smoke', 'suitcase', 'flower', 'highway', 'tent',
 'beanie', 'orange', 'pile', 'glove', 'computer', 'dough', 'cave', 'fruit', 'oven', 'ipad', 'tuxedo', 'laptop',
 'footage', 'basketball hoop', 'scarf', 'lettuce', 'moon', 'board', 'present', 'piece', 'meat', 'caribbean', 'shoe',
 'fern', 'mango', 'platform', 'storm', 'pilgrim', 'bottom', 'beer', 'office', 'quote', 'sidewalk', 'desert', 'hay',
 'lion', 'crate', 'figurine', 'classroom', 'trash', 'panda bear', 'arch', 'barn', 'harness', 'flag', 'club', 'turtle',
 'straw', 'marsh', 'wall', 'seat', 'None', 'hay bale', 'building', 'window', 'suit', 'blueberry', 'rain', 'walkway',
 'wheel', 'spacesuit', 'bread', 'mat', 'gym', 'village', 'waterfall', 'glass', 'tractor', 'bandage', 'data center',
 'carriage', 'kitchen', 'bar', 'niagara falls', 'skateboard', 'butterfly fish', 'bookshelf', 'breast', 'bow tie',
 'coat', 'wave', 'fence', 'trail', 'stair case', 'line', 'diver', 'food', 'canada', 'biker', 'hoop', 'espresso machine',
 'cat', 'money', 'foot', 'snowboard', 'ant', 'pond', 'uniform', 'lynx', 'satellite', 'robe', 'ad', 'sunglasses',
 'pizza', 'cobblestone']


BACKGROUND_CATEGORIES = ['railroad track', 'beach', 'bone', 'tag', 'water', 'rack', 'tank', 'ground', 'tugboat', 'fire',
                         'tunnel', 'bale', 'sand', 'lobby', 'student', 'center', 'fireplace', 'column', 'parking lot',
                         'field', 'dock', 'dirt', 'substance', 'position', 'background', 'railroad', 'tray', 'valley',
                         'coral reef', 'server rack', 'skyline', 'motion', 'floor', 'swamp', 'stack', 'air', 'town',
                         'overall', 'push up', 'road', 'light', 'rubble', 'weed', 'gymnasium', 'legging', 'writing',
                         'forest', 'street', 'sky', 'woods', 'surface', 'room', 'precinct', 'crowd', 'tv', 'subway',
                         'construction site', 'redwood', 'fall', 'sun', 'event', 'edge', 'garage', 'leaf', 'airport',
                         'step', 'center', 'doorway', 'valley', 'drawing', 'hill', 'coral reef', 'server rack',
                         'skyline', 'motion', 'back', 'stack', 'air', 'town', 'overall', 'push up', 'collar', 'road',
                         'light', 'rubble', 'background', 'field', 'color', 'sign', 'sailboat', 'market', 'island',
                         'yard', 'panel', 'hut', 'world', 'golf course', 'shack', 'night', 'counter top', 'couch',
                         'platter', 'liquid', 'darkness', 'ocean', 'background', 'mud', 'cloud', 'baseball field',
                         'hillside', 'wood', 'forest', 'area', 'row', 'hall', 'range', 'world', 'sidewalk', 'desert',
                         'classroom', 'trash', 'wall', 'None', 'building', 'highway', 'village', 'street', 'trail',
                         'road', 'beach', 'river', 'shore', 'stream', 'mountain', 'valley', 'hill', 'forest', 'woods',
                         'field', 'park', 'garden', 'playground', 'stadium', 'arena', 'courtyard', 'alley', 'path',
                         'lane', 'route', 'track', 'roadway', 'highway', 'freeway', 'expressway', 'thoroughfare',
                         'drive', 'driveway', 'avenue', 'boulevard', 'street', 'lane', 'road', 'path', 'alley',
                         'avenue', 'boulevard', 'court', 'drive', 'lane', 'path', 'road', 'route', 'street',
                         'track', 'trail', 'way', 'alley', 'avenue', 'boulevard', 'lane', 'road', 'street', 'track',
                         'trail', 'way', 'alley', 'path', 'road', 'street', 'trail', 'lane', 'path', 'route', 'track',
                         'trail', 'way']


FOREGROUND_CATEGORIES = ['trunk', 'hoodie', 'baseball player', 'toy', 'thong', 'angelfish', 'helm', 'trailer', 'owl',
                         'man', 'clothing', 'bag', 'pearl', 'blazer', 'tie', 'fire ball', 'skier', 'excavator', 'bike',
                         'cage', 'pad', 'fingernail', 'grill', 'hat', 'baby', 'landing gear', 'eel', 'bison', 'mouse',
                         'strawberry', 'chihuahua', 'hamburger', 'tusk', 'nose', 'necklace', 'glasses', 'goat', 'car',
                         'fork', 'adult', 'blanket', 'rv', 'card', 'surfer', 'arm', 'cloth', 'mouth', 'golf ball', 'ski',
                         'ball', 'cake', 'tank top', 'pirate', 'donut', 'kick', 'plate', 'pose', 'wrist', 'house', 'door',
                         'cap', 'hair', 'boulder', 'pant', 'scissor', 'phone', 'soccer ball', 'belt', 'art', 'headphone',
                         'woman', 'salad', 'bird', 'coffee cup', 'basketball', 'fighter jet', 'wheelchair', 'cruise ship',
                         'airplane', 'cheek', 'slice', 'paintbrush', 'whale shark', 'stick', 'mask', 'cupcake', 'truck',
                         'mirror', 'tarp', 'bowl', 'firework', 'test tube', 'face', 'leather', 'dolphin', 'horse', 'bear',
                         'desk', 'hospital', 'printer', 'spoon', 'exercise', 'character', 'frisbee', 'screen', 'tennis ball',
                         'combine', 'tree', 'pencil', 'palm tree', 'microphone', 'bra', 'drawing', 'meal', 'rock', 'deer',
                         'jetliner', 'brain', 'microscope', 'stocking', 'harvester', 'jacket', 'ship', 'cell phone', 'knee',
                         'branch', 'bed', 'lab coat', 'brush', 'astronaut', 'locker', 'anemone', 'reef', 'collar', 'breakfast',
                         'hand', 'cherry blossom', 'roller coaster', 'mustache', 'dreadlock', 'shirt', 'steering wheel', 'coral',
                         'cow', 'apple', 'child', 'bull', 'bun', 'bow', 'panda cub', 'instrument', 'leg', 'base', 'city', 'revolver',
                         'legging', 'doll', 'cart', 'vegetable', 'fountain', 'pen', 'dog', 'jean', 'game', 'sheep', 'paw',
                         'whale', 'sunflower', 'shuttle', 'golf', 'lipstick', 'antelope', 'living room', 'hole', 'tablet',
                         'paint', 'gazelle', 'bouquet', 'mangrove', 'camera', 'harbor', 'panda', 'train', 'compass', 'alga',
                         'turkey', 'box', 'dumbbell', 'stingray', 'statue', 'vest', 'umbrella', 'balloon', 'nurse', 'fish',
                         'bat', 'beak', 'shark', 'duck', 'short', 'head', 'painting', 'karate', 'ice cream', 'vehicle',
                         'labrador retriever', 'speedometer', 'guitar', 'machine', 'dinner', 'pillar', 'shotgun', 'eye',
                         'aquarium', 'bamboo', 'rose', 'iguana', 't - shirt', 'scuba diver', 'roof', 'shower', 'cattle',
                         'text', 'boat', 'tongue', 'fishing rod', 'juice', 'sweater', 'gun', 'mud', 'stable', 'cloud',
                         'shoulder', 'people', 'chicken', 'baseball field', 'melon', 'wing', 'paper', 'candle',
                         'housewife', 'wood', 'sling', 'apron', 'chair', 'station', 'finger', 'flame', 'comforter',
                         'push', 'dark', 'croissant', 'puppy', 'body', 'bridle', 'coffee maker', 'clock', 'plant',
                         'skirt', 'syringe', 'egg', 'dresser', 'counter', 'neck', 'towel', 'cigarette', 'sweatshirt',
                         'backpack', 'bush', 'monkey', 'zoo', 'book', 'toilet', 'lip', 'butterfly', 'tail', 'elephant',
                         'shop', 'dress', 'jeep', 'explosion', 'beard', 'rifle', 'stomach', 'motorcycle', 'badge',
                         'mother', 'stand', 'piano', 'luggage', 'cup', 'track', 'hallway', 'wine', 'crane', 'television',
                         'sponge', 'parrot fish', 'golf club', 'bill', 'girl', 'wheat', 'suspender', 'rod', 'sink',
                         'family', 'pastry', 'stair', 'person', 'player', 'arrow', 'animal', 'tooth', 'father', 'table',
                         'slope', 'space station', 'parrotfish', 'combine harvester', 'boy', 'elevator', 'hammock',
                         'video', 'outfit', 'bench', 'top', 'bunker', 'bottle', 'server', 'rug', 'grain', 'tower',
                         'image', 'van', 'buffalo', 'banana', 'soccer', 'jellyfish', 'penguin', 'heel', 'knife', 'gear',
                         'coffee', 'ice cream cone', 'smile', 'park', 'dollar', 'smoke', 'suitcase', 'flower', 'tent',
                         'beanie', 'orange', 'pile', 'glove', 'computer', 'dough', 'cave', 'fruit', 'oven', 'ipad',
                         'tuxedo', 'laptop', 'footage', 'basketball hoop', 'scarf', 'lettuce', 'moon', 'board', 'present',
                         'piece', 'meat', 'caribbean', 'shoe', 'fern', 'mango', 'platform', 'storm', 'pilgrim', 'bottom',
                         'beer', 'office', 'quote', 'hay', 'lion', 'crate', 'figurine', 'classroom', 'trash', 'panda bear',
                         'arch', 'barn', 'harness', 'flag', 'club', 'turtle', 'straw', 'marsh', 'wall', 'seat', 'hay bale',
                         'building', 'window', 'suit', 'blueberry', 'rain', 'walkway', 'wheel', 'spacesuit', 'bread',
                         'mat', 'gym', 'village', 'waterfall', 'glass', 'tractor', 'bandage', 'data center', 'carriage',
                         'kitchen', 'bar', 'niagara falls', 'skateboard', 'butterfly fish', 'bookshelf', 'breast',
                         'bow tie', 'coat', 'wave', 'fence', 'trail', 'stair case', 'line', 'diver', 'food', 'canada',
                         'biker', 'hoop', 'espresso machine', 'cat', 'money', 'foot', 'snowboard', 'ant', 'pond',
                         'uniform', 'lynx', 'satellite', 'robe', 'ad', 'sunglasses', 'pizza', 'cobblestone']


PRIORITY_CATEGORIES = ['man', 'baby', 'woman', 'child', 'bull', 'panda cub', 'dog', 'bear', 'horse', 'sheep', 'cat', 'chicken', 'duck', 'elephant', 'monkey', 'penguin', 'labrador retriever', 'panda', 'bison', 'chihuahua', 'antelope', 'gazelle', 'dolphin', 'whale', 'whale shark', 'tugboat', 'tug boat', 'deer', 'lion', 'buffalo', 'ant', 'hammock', 'butterfly', 'butterfly fish', 'stingray', 'parrot fish', 'parrotfish', 'fish', 'eagle', 'owl', 'turtle']


def load_masks_from_png(mask_dir, json_data):
    """
    Load masks and category information from PNG files and JSON data.
    :param mask_dir: Directory storing mask PNG files.
    :param json_data: JSON data containing category information.
    :return: Masks and category information organized by video and frame.
    """
    masks = {}

    # Iterate through all PNG files in the mask directory
    for mask_file in os.listdir(mask_dir):
        if not mask_file.endswith(".png"):
            continue

        # Parse the filename to extract video_id, frame_id, and label
        match = re.match(r"mask_(\d+)_f(\d+)_(\d+).png", mask_file)
        if not match:
            continue

        video_id = int(match.group(1))
        frame_id = int(match.group(2))
        label = match.group(3)

        # Load the PNG file
        mask_path = os.path.join(mask_dir, mask_file)
        mask_image = Image.open(mask_path)
        mask_array = np.array(mask_image)  # Convert to numpy array

        # Get category name
        mask_key = f"mask_{video_id}_f{frame_id}"
        if mask_key not in json_data:
            continue
        label_dict = json_data[mask_key]
        if label not in label_dict:
            continue
        category_name = label_dict[label]

        # Store mask and category information
        if video_id not in masks:
            masks[video_id] = {}
        if frame_id not in masks[video_id]:
            masks[video_id][frame_id] = {}
        masks[video_id][frame_id][int(label)] = {
            "segmentation": mask_array,
            "category": category_name
        }

    return masks


def calculate_center(segmentation):
    """
    Calculate the center point of an object in the mask.
    :param segmentation: Segmentation mask with shape (H, W).
    :return: Object's center point coordinates (x, y).
    """
    y_indices, x_indices = np.where(segmentation > 0)  # Assume non-zero values in mask represent the object
    if len(y_indices) == 0:
        return None
    center_y = np.mean(y_indices)
    center_x = np.mean(x_indices)
    return (center_x, center_y)


def select_key_objects_for_video(video_masks, top_k=3):
    """
    Select key objects for a video based on inter-frame changes and semantic information.
    :param video_masks: Mask and category information for a video, organized by frame.
    :param top_k: Return the top k key objects.
    :return: List of key object names.
    """
    # Store inter-frame changes for each object
    object_changes = defaultdict(float)
    # Store size (area ratio) for each object
    object_sizes = defaultdict(float)

    frame_ids = sorted(video_masks.keys())

    # Iterate through each frame
    for i in range(1, len(frame_ids)):
        prev_frame_id = frame_ids[i - 1]
        curr_frame_id = frame_ids[i]

        prev_masks = video_masks[prev_frame_id]
        curr_masks = video_masks[curr_frame_id]

        # Iterate through each object in the current frame
        for label, curr_mask_info in curr_masks.items():
            category_name = curr_mask_info["category"]

            # Exclude background categories
            if category_name in BACKGROUND_CATEGORIES:
                continue

            # Calculate the center point of the current frame object
            curr_center = calculate_center(curr_mask_info["segmentation"])
            if curr_center is None:
                continue

            # Calculate the center point of the previous frame object
            if label not in prev_masks:
                continue
            prev_center = calculate_center(prev_masks[label]["segmentation"])
            if prev_center is None:
                continue

            # Calculate inter-frame displacement
            displacement = np.sqrt((curr_center[0] - prev_center[0]) ** 2 + (curr_center[1] - prev_center[1]) ** 2)

            # If it is a high-priority category, assign higher weight
            if category_name in PRIORITY_CATEGORIES:
                displacement *= 2  # Weight factor, can be adjusted as needed

            object_changes[category_name] += displacement

            # Calculate object size (area ratio)
            mask_area = np.sum(curr_mask_info["segmentation"] > 0)
            image_area = curr_mask_info["segmentation"].shape[0] * curr_mask_info["segmentation"].shape[1]
            object_sizes[category_name] = mask_area / image_area

    # Sort by inter-frame change and select the top_k key objects
    sorted_objects = sorted(object_changes.items(), key=lambda x: x[1], reverse=True)

    # Prefer high-priority categories
    priority_objects = [obj for obj, change in sorted_objects if obj in PRIORITY_CATEGORIES]
    if priority_objects:
        key_objects = priority_objects[:top_k]
    else:
        # If there are no high-priority categories, select based on original logic
        filtered_objects = [obj for obj, change in sorted_objects if object_sizes[obj] < 0.5]  # Area ratio less than 50%
        if not filtered_objects:  # If no objects meet the criteria, select the ones with the largest displacement
            filtered_objects = [obj for obj, change in sorted_objects]
        key_objects = filtered_objects[:top_k]

    return key_objects


def select_key_objects_for_all_videos_old(masks, images, top_k=3):
    """
    Select key objects for all videos.
    :param masks: Mask and category information organized by video and frame.
    :param images: Video frame images with shape [num_videos, 3, H, W].
    :param top_k: Return the top k key objects.
    :return: Dictionary where keys are video IDs and values are the key object lists for that video.
    """
    video_key_objects = {}

    # Create directory to save images
    os.makedirs("./vis_key_object", exist_ok=True)

    # Iterate through each video
    for video_id, video_masks in tqdm(masks.items()):
        key_objects = select_key_objects_for_video(video_masks, top_k)
        video_key_objects[video_id] = key_objects

        # Get the second frame image of the current video
        image_idx = video_id
        image = images[image_idx].permute(1, 2, 0).numpy()  # Convert to HWC format

        # Get the key object's mask
        frame_id = 2  # Assume the second frame
        if frame_id in video_masks:
            for label, mask_info in video_masks[frame_id].items():
                if mask_info["category"] in key_objects:
                    mask = mask_info["segmentation"]

                    # Create visualization image
                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    ax[0].imshow(image)
                    ax[0].set_title("Original Image")
                    ax[0].axis("off")

                    ax[1].imshow(mask, cmap="gray")
                    ax[1].set_title(f"Key Object Mask: {mask_info['category']}")
                    ax[1].axis("off")

                    # Save image
                    print(f"\033[92m saving ./vis_key_object/{image_idx}_{mask_info['category']}.jpg\033[0m")
                    plt.savefig(f"./vis_key_object/{image_idx}_{mask_info['category']}.jpg")
                    plt.close()

    return video_key_objects


def select_key_objects_for_all_videos(masks, images, top_k=1, save_vis=False):
    """
    Select key objects for all videos and store each video's key objects and masks for all frames.
    :param masks: Mask and category information organized by video and frame.
    :param images: Video frame images with shape [num_videos, 3, H, W].
    :param top_k: Return the top k key objects.
    :return: Dictionary where keys are video IDs and values are key object information (including category and masks for all frames).
    """
    video_key_objects = {}
    all_masks = np.zeros((4320, 6, 224, 224))  # To store masks for all videos, shape [4320, 6, 224, 224]

    # Create directory to save images
    os.makedirs("./vis_key_object", exist_ok=True)

    # Iterate through each video
    for video_id, video_masks in tqdm(masks.items()):
        # Get the key object categories for the current video
        key_objects = select_key_objects_for_video(video_masks, top_k)
        if not key_objects:
            # If there are no key objects, fill with zero masks
            key_object_masks = np.zeros((6, 224, 224))
            key_object_category = "None"
        else:
            # Select the first key object (assuming top_k=1)
            key_object_category = key_objects[0]

            # Initialize a list of zero masks with shape [6, 224, 224]
            key_object_masks = [np.zeros((224, 224)) for _ in range(6)]

            # Iterate through all frames (0-5)
            for frame_id in range(6):
                if frame_id in video_masks:
                    for label, mask_info in video_masks[frame_id].items():
                        if mask_info["category"] == key_object_category:
                            key_object_masks[frame_id] = mask_info["segmentation"]

            # Convert the list of masks to a numpy array, shape [6, 224, 224]
            key_object_masks = np.stack(key_object_masks, axis=0)

        # Add the current video's masks to the list of all videos' masks
        all_masks[video_id] = key_object_masks

        # Store key object information
        video_key_objects[video_id] = {
            "category": key_object_category
        }

        if save_vis:
            # Visualize and save the key object's mask (for the second frame as an example)
            image_idx = video_id
            image = images[image_idx].permute(1, 2, 0).numpy()  # Convert to HWC format
            frame_id = 2  # Assume the second frame
            if frame_id in video_masks:
                for label, mask_info in video_masks[frame_id].items():
                    if mask_info["category"] == key_object_category:
                        mask = mask_info["segmentation"]

                        # Create visualization image
                        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                        ax[0].imshow(image)
                        ax[0].set_title("Original Image")
                        ax[0].axis("off")

                        ax[1].imshow(mask, cmap="gray")
                        ax[1].set_title(f"Key Object Mask: {key_object_category}")
                        ax[1].axis("off")

                        # Save image
                        plt.savefig(f"./vis_key_object/{image_idx}_{key_object_category}.jpg")
                        plt.close()

    # Convert all videos' masks to torch.tensor, shape [4320, 6, 224, 224]
    all_masks = torch.tensor(all_masks, dtype=torch.float32)

    return video_key_objects, all_masks


if __name__ == '__main__':
    print(f"\033[92m all_list {len(all_list)} \033[0m")
    print(f"\033[92m BACKGROUND_CATEGORIES {len(list(set(BACKGROUND_CATEGORIES)))} \033[0m")
    print(f"\033[92m FOREGROUND_CATEGORIES {len(list(set(FOREGROUND_CATEGORIES)))} \033[0m")

    root_dir = './cc2017_dataset/masks'



    for mode in ["train", "test"]:

        images = torch.load(f'./cc2017_dataset/GT_{mode}_3fps.pt', map_location='cpu')[:, 2, :, :, :]  # [4320, 3, 224, 224]

        masks_json = json.load(open(f'{root_dir}/mask_cls_dict_{mode}_qwen_video.json'))

        mask_dir = f'{root_dir}/mask_cls_{mode}_qwen_video/'  # Replace with actual mask PNG file directory
        masks = load_masks_from_png(mask_dir, masks_json)

        # Select key objects and store
        key_objects, all_masks = select_key_objects_for_all_videos(masks, images, top_k=1)

        # Sort key objects by video ID
        sorted_video_key_objects = dict(sorted(key_objects.items(), key=lambda x: x[0]))

        # Print key object information
        print("Key objects:", sorted_video_key_objects)

        # Save key object information to file
        with open(f"{root_dir}/key_objects_info_{mode}.json", "w") as f:
            json.dump(sorted_video_key_objects, f, indent=4)

        # Save all videos' masks to file
        torch.save(all_masks, f"{root_dir}/key_objects_masks_{mode}.pt")