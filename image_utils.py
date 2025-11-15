from PIL import Image


def concat_images_horizontally(images):
    """
    Concatenate a list of PIL.Image objects horizontally.

    Args:
        images (List[PIL.Image]): List of PIL images.

    Returns:
        PIL.Image: A new image composed of the input images concatenated side-by-side.
    """
    # Resize all images to the same height (optional)
    heights = [img.height for img in images]
    min_height = min(heights)
    resized_images = [
        img if img.height == min_height else img.resize(
            (int(img.width * min_height / img.height), min_height),
            Image.LANCZOS
        ) for img in images
    ]

    # Compute total width and max height
    total_width = sum(img.width for img in resized_images)
    height = min_height

    # Create new blank image
    new_img = Image.new('RGB', (total_width, height))

    # Paste images side by side
    x_offset = 0
    for img in resized_images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_img

def concat_images_vertically(images):
    """
    Concatenate a list of PIL.Image objects vertically.

    Args:
        images (List[PIL.Image]): List of PIL images.

    Returns:
        PIL.Image: A new image composed of the input images stacked top-to-bottom.
    """
    # Resize all images to the same width (optional)
    widths = [img.width for img in images]
    min_width = min(widths)
    resized_images = [
        img if img.width == min_width else img.resize(
            (min_width, int(img.height * min_width / img.width)),
            Image.LANCZOS
        ) for img in images
    ]

    # Compute total height and width
    total_height = sum(img.height for img in resized_images)
    width = min_width

    # Create new blank image
    new_img = Image.new('RGB', (width, total_height))

    # Paste images one below another
    y_offset = 0
    for img in resized_images:
        new_img.paste(img, (0, y_offset))
        y_offset += img.height

    return new_img