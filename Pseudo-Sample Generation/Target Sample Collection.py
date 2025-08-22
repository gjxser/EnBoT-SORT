import cv2
import numpy as np

# Global variables are used to store the pixel values of the background points clicked by the user.
bg_pixel_value = None
clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global bg_pixel_value, clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        bg_pixel_value = param[y, x]
        print(f"The coordinates of the selected background points: {clicked_point}, Pixel value: {bg_pixel_value}")

def binarize_image(img, bg_rgb, tol=10, blur_kernel=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # First, perform mean blurring
    blurred = cv2.blur(gray, (blur_kernel, blur_kernel))  # blur_kernel 控制模糊程度

    # Convert the selected RGB background to grayscale
    bg_gray = int(np.dot(bg_rgb[:3], [0.114, 0.587, 0.299]))

    # Calculate the difference between each pixel and the background gray level
    diff = np.abs(blurred.astype(np.int16) - bg_gray)

    # If the difference is less than the tolerance, it is considered as background (set to 0); otherwise, it is considered as target (set to 1)
    binary = np.where(diff <= tol, 0, 1).astype(np.uint8)
    return binary

def generate_random_blocks(shape, N, min_value=130):
    h, w = shape
    block_h, block_w = h // N, w // N
    result = np.zeros((h, w), dtype=np.uint8)
    for i in range(N):
        for j in range(N):
            value = np.random.randint(min_value, 256)
            result[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w] = value
    return result

def main():
    global bg_pixel_value
    img = cv2.imread('target_images_vi/0401_0004.png')  # PATH
    scale_factor = 8
    height, width = img.shape[:2]
    new_size = (int(width * scale_factor), int(height * scale_factor))
    img = cv2.resize(img, new_size)
    if img is None:
        print("Unable to load image")
        return

    cv2.namedWindow('Select Background Pixel')
    cv2.setMouseCallback('Select Background Pixel', mouse_callback, img)

    print("Please click on the image to select background pixels...")
    while True:
        cv2.imshow('Select Background Pixel', img)
        if cv2.waitKey(1) & 0xFF == 27 or bg_pixel_value is not None:
            break
    cv2.destroyWindow('Select Background Pixel')

    # generate a binary image
    binary = binarize_image(img, bg_pixel_value)  # 0=BACKGROUND, 1=TARGET
    cv2.namedWindow("window_name", cv2.WINDOW_NORMAL)
    cv2.imshow('Binary Image', binary * 255)

    # Generate an NxN random block image
    N = 16  # The number of blocks can be adjusted according to needs
    rand_blocks = generate_random_blocks(binary.shape, N)
    cv2.imshow('Random Block Image', rand_blocks)

    result = cv2.bitwise_and(rand_blocks, rand_blocks, mask=binary)
    cv2.imshow('AND Result', result)

    # Gaussian blur, simulating infrared targets
    blurred = cv2.GaussianBlur(result, (15, 15), sigmaX=5)
    cv2.imshow('Simulated Infrared Target', blurred)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
