from PIL import Image


def crop_image(image, center, crop_size):

    width = 1680
    height = 1050
    crop_half = crop_size // 2
    
    left = max(center[0] - crop_half, 0)
    top = max(center[1] - crop_half, 0)
    right = min(center[0] + crop_half, width)
    bottom = min(center[1] + crop_half, height)

    new_image = Image.new('RGB', (crop_size, crop_size), (0, 0, 0))

    cropped_part = image.crop((left, top, right, bottom))
    new_image.paste(cropped_part, (crop_half - (center[0] - left), crop_half - (center[1] - top)))

    return new_image


if __name__ == "__main__":
    image = Image.open(r"C:\datasets\COCO-Search18\TP\bottle\000000001455.jpg")
    crop_size = 292
    center = (146, 146)
    cropped_image = crop_image(image, crop_size, center)
    cropped_image.show()
