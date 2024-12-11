from PIL import Image


def crop_image(image: Image, center, crop_size) -> Image:
    width = 1680
    height = 1050
    assert len(center) == 2
    assert len(crop_size) == 2

    crop_w, crop_h = crop_size
    crop_w_half = crop_w // 2
    crop_h_half = crop_h // 2
    crop_half = max(crop_w_half, crop_h_half)
    
    left = max(center[0] - crop_w_half, 0)
    top = max(center[1] - crop_h_half, 0)
    right = min(center[0] + crop_w_half, width)
    bottom = min(center[1] + crop_h_half, height)

    crop_image_length = max(crop_w, crop_h)

    new_image = Image.new('RGB', (crop_image_length, crop_image_length), (0, 0, 0))

    cropped_part = image.crop((left, top, right, bottom))
    # cropped_part.show()
    new_image.paste(cropped_part, (crop_half - (center[0] - left), crop_half - (center[1] - top)))

    return new_image


def crop_image_using_bbox(image: Image, bbox: list, mode="padding") -> Image:
    assert len(bbox) == 4
    modes = {
        "padding": lambda w, h: (w, h),
        "fill": lambda w, h: (max([w, h]), max([w, h]))
    }
    if mode not in modes:
        raise ValueError(f"unsupported modes: {mode}")
    
    x, y, w, h = bbox
    center = (round(x + w / 2), round(y + h / 2))

    crop_size = modes[mode](w, h)

    return crop_image(image, center, crop_size)



if __name__ == "__main__":
    image = Image.open(r"C:\datasets\COCO-Search18\TP\bottle\000000001455.jpg")
    crop_size = 292
    center = (146, 146)
    # cropped_image = crop_image(image, (952, 137), (242, 105))
    cropped_image = crop_image_using_bbox(image, [899, 16, 105, 242], "fill")
    cropped_image.show()
