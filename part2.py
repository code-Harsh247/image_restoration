import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
originalImg = plt.imread('./restore.jpg')

# Check if the image is colored (RGB)
originalImg = plt.imread('./restore.jpg')  
img = np.mean(originalImg, axis=2)

# Motion Blur Kernel
def motion_blur_kernel(length):
    kernel = np.zeros(length)
    kernel[:] = 1  # Set all values to 1
    kernel /= length  # Normalize the kernel
    return kernel

# Apply Motion Blur
def apply_motion_blur(image, kernel):
    h, w = image.shape
    blurred_image = np.zeros_like(image, dtype=np.float32)

    # Pad the image to handle borders
    padded_image = np.pad(image, (0, kernel.size - 1), mode='edge') 
    
    for i in range(h):
        for j in range(w):
            # Calculate the sum of the pixel values for the kernel size
            blurred_value = np.sum(padded_image[i, j:j + kernel.size] * kernel)
            blurred_image[i, j] = blurred_value

    return np.clip(blurred_image, 0, 255).astype(np.uint8)

# Create the motion blur kernel and apply it
blur_length = 15
kernel = motion_blur_kernel(blur_length)
blurred_image = apply_motion_blur(img, kernel)

# Adaptive Filtering (Simple Average Filter)
def adaptive_filter(image, filter_size):
    h, w = image.shape
    restored_image = np.zeros_like(image, dtype=np.float32)
    
    pad_size = filter_size // 2
    padded_image = np.pad(image, pad_size, mode='edge')  
    for i in range(h):
        for j in range(w):
            local_region = padded_image[i:i + filter_size, j:j + filter_size]
            restored_image[i, j] = np.mean(local_region)

    return np.clip(restored_image, 0, 255).astype(np.uint8)

#Apply Adaptive Filtering
filter_size = 5  # Size of the average filter
restored_image = adaptive_filter(blurred_image, filter_size)

# Calculate MSE
def calculate_mse(original, restored):
    return np.mean((original - restored) ** 2)

# Calculate PSNR
def calculate_psnr(original, restored):
    mse = calculate_mse(original, restored)
    if mse == 0:
        return float('inf') 
    max_pixel_value = 255.0 
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr

# Calculate MSE and PSNR between original and restored images
mse_value = calculate_mse(img, restored_image)
psnr_value = calculate_psnr(img, restored_image)

print(f'Mean Squared Error (MSE): {mse_value}')
print(f'Peak Signal-to-Noise Ratio (PSNR): {psnr_value:.2f} dB')

plt.imsave("./restored_image_part2.png", restored_image, cmap='gray')
output_txt_path = 'psnr_mse_values_part2.txt'
with open(output_txt_path, 'w') as f:
    f.write(f'Mean Squared Error (MSE): {mse_value}\n')
    f.write(f'Peak Signal-to-Noise Ratio (PSNR): {psnr_value} dB\n')

# Display the original, blurred, and restored images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Blurred Image (Motion Blur)')
plt.imshow(blurred_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Restored Image (Adaptive Filter)')
plt.imshow(restored_image, cmap='gray')
plt.axis('off')

plt.show()


