import cv2
import numpy as np
import os


def replace_transparent_background(image):
    """
    Replace the transparent background of an image with the standard Minecraft empty slot color.
    """
    b, g, r, a = cv2.split(image)
    mask = a < 0.01
    image[mask] = (139, 139, 139, 255)
    return image


def remove_stray_pixels(mask):
    """
    Once the inventory mask has been scaled down, remove any stray single pixels that have black neighbors on all
    four sides, ignoring diagonal neighbors.
    """
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(mask, -1, kernel)
    stray_pixels = (mask == 255) & (neighbor_count == 0)
    mask[stray_pixels] = 0
    return mask


def mask_color(image, color):
    return cv2.inRange(image, color, color)


def mask_color_range(image, lower, upper):
    return cv2.inRange(image, lower, upper)


def vectorize_image(image):
    """
    This function converts an inventory slot image to a feature vector. The feature vector consists of the histograms
    of the red, green, and blue channels of the image, as well as a binary representation of the shape of the object
    in the image. The general shape helps to distinguish between different items that have similar colors.
    """
    if image.shape != (16, 16, 3):
        raise ValueError('Item image must be 16 x 16 pixels to be vectorized')

    # Find the histograms for all three channels
    hist_r = np.histogram(image[:, :, 0].flatten(), bins=256, range=(0, 256))[0]
    hist_g = np.histogram(image[:, :, 1].flatten(), bins=256, range=(0, 256))[0]
    hist_b = np.histogram(image[:, :, 2].flatten(), bins=256, range=(0, 256))[0]

    # Create a binary representation of the shape of the object in the image
    shape = mask_color(image, (139, 139, 139))
    shape[9:, 5:] = 0
    shape = shape.flatten()

    return np.concatenate([hist_r, hist_g, hist_b, shape])


class InventoryViewer:
    """
    Class for viewing the contents of a Minecraft inventory using computer vision.
    """

    def __init__(self, inventory_icons_directory):
        """
        Initialize the InventoryViewer with the directory containing the inventory icons
        of the items to be recognized.
        """
        self.invicon_dir = inventory_icons_directory
        self.db = self.VectorDatabase()
        self.load_invicons()

    def load_invicons(self):
        # Load all inventory icons from the directory and add them to the database
        # The icons taken from the wiki are named 'Invicon_[item name].png', so we look for this pattern.
        for filename in os.listdir(self.invicon_dir):
            if filename.startswith('Invicon_') and filename.endswith('.png'):
                item = self.ItemImage(os.path.join(self.invicon_dir, filename))
                self.db.add_vector(item.as_vector(), filename)

    def process_inventory_image(self, image_path):
        inventory_img = cv2.imread(image_path)
        inventory_img_rgb = cv2.cvtColor(inventory_img, cv2.COLOR_BGR2RGB)

        # Mask out the empty inventory slots and inventory borders
        empty_slot_mask = mask_color(inventory_img_rgb, np.array([139, 139, 139]))  # empty inventory slot color
        border_mask_gray = mask_color(inventory_img_rgb, np.array([198, 198, 198]))  # inventory border color
        border_mask_white = mask_color(inventory_img_rgb, np.array([255, 255, 255]))  # white

        # Crop the image to only the inventory area
        x, y, w, h = cv2.boundingRect(border_mask_gray)
        inventory_img = inventory_img[y:y + h, x:x + w]
        empty_slot_mask = empty_slot_mask[y:y + h, x:x + w]
        border_mask_white = border_mask_white[y:y + h, x:x + w]

        # The GUI scale is an integer which multiplies the size of the pixels in the inventory. We can calculate
        # it by checking the first few pixels from the left edge of the image, since there is a white border around
        # the inventory. At GUI scale 1, this white border is 1 pixel wide, at GUI scale 2, it is 2 pixels wide, etc.
        gui_scale = 0
        for i in range(10):
            white_pixel = border_mask_white[border_mask_white.shape[0] // 2, i] == 255
            if not white_pixel:
                break
            gui_scale += 1

        # Scale down the empty slot mask by the GUI scale to get everything down to the original pixel by pixel size
        # Then remove any stray pixels (there are some in the inventory design between slots)
        # Then scale it back up to the original size
        empty_slot_mask = empty_slot_mask[::gui_scale, ::gui_scale]
        empty_slot_mask = remove_stray_pixels(empty_slot_mask)
        empty_slot_mask = np.repeat(np.repeat(empty_slot_mask, gui_scale, axis=0), gui_scale, axis=1)

        # Find the contours of the empty slots
        contours, _ = cv2.findContours(empty_slot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Remove any contours that don't have a matching height and width, since inventory slots are squares
        contours = [contour for contour in contours if cv2.boundingRect(contour)[2] == cv2.boundingRect(contour)[3]]

        # Find the most common width of the inventory slots
        widths = [cv2.boundingRect(contour)[2] for contour in contours]
        width_counts = {width: widths.count(width) for width in widths}
        most_common_width = max(width_counts, key=width_counts.get)

        # Remove any contours that don't have the most common width, since all inventory slots should have the same size
        contours = [contour for contour in contours if cv2.boundingRect(contour)[2] == most_common_width]

        # Create InventorySlot objects for each slot, then search for the closest item in the database
        slots = [self.InventorySlot(inventory_img, *cv2.boundingRect(contour)) for contour in contours]
        items = [self.db.find_closest_vector(slot.as_vector()) for slot in slots if not slot.is_empty()]

        # Return the list of items in the inventory, removing the 'Invicon_' prefix and '.png' suffix
        return [item.replace('Invicon_', '').replace('.png', '').replace('_', ' ') for item in items]

    class ItemImage:
        """
        Class for representing an item icon image as taken from the Minecraft wiki.
        """

        def __init__(self, item_image_filename):
            self.image = cv2.imread(item_image_filename, cv2.IMREAD_UNCHANGED)
            self.w, self.h = self.image.shape[:2]
            if self.w != self.h or (self.w & (self.w - 1)) != 0:
                raise ValueError('Image must be a square with a size that is a power of 2')
            self.image = self.image[::self.w // 16, ::self.h // 16]
            self.image = replace_transparent_background(self.image)
            self.image = self.image[:, :, :3]

        def as_vector(self):
            return vectorize_image(self.image)

    class InventorySlot:
        """
        Class for representing an inventory slot extracted from the inventory image.
        """

        def __init__(self, image, x, y, w, h):
            self.x = x
            self.y = y
            self.w = w
            self.h = h
            self.image = image[self.y:self.y + self.h, self.x:self.x + self.w]
            if self.w != self.h or (self.w & (self.w - 1)) != 0:
                raise ValueError('Image must be a square with a size that is a power of 2')
            self.image = self.image[::self.w // 16, ::self.h // 16]
            self.empty = np.all(np.logical_or(self.image == (139, 139, 139), self.image == (85, 85, 85)))

        def is_empty(self):
            return self.empty

        def as_vector(self):
            return vectorize_image(self.image)

    class VectorDatabase:
        """
        A very simple vector database for storing vectors and their corresponding values. This is used to store
        and search for the inventory item vectors.
        """

        def __init__(self):
            self.vectors = []
            self.values = []

        def add_vector(self, vector, value):
            self.vectors.append(vector)
            self.values.append(value)

        def find_closest_vector(self, vector):
            min_distance = float('inf')
            closest_value = None
            for i, v in enumerate(self.vectors):
                distance = np.linalg.norm(vector - v)
                if distance < min_distance:
                    min_distance = distance
                    closest_value = self.values[i]
            return closest_value


if __name__ == '__main__':
    viewer = InventoryViewer('images')
    items = viewer.process_inventory_image('images/screenshot.png')
    print(items)
