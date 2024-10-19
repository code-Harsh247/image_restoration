import numpy as np
import matplotlib.pyplot as plt


#Read image restore.jpg in gray scale.
originalImg = plt.imread('./restore.jpg')  
img = np.mean(originalImg, axis=2)
    
    
# Perform 2D Fourier Transform
F_transform = np.fft.fft2(img)

# Shift the zero frequency component (DC component) to the center of the spectrum
F_shifted = np.fft.fftshift(F_transform)

# Get image dimensions
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2  # Center of the frequency domain


# Create a mask with ones in the low frequency area and zeros elsewhere
radius = 100 
mask = np.zeros((rows, cols))


for i in range(rows):
    for j in range(cols):
        distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
        if distance <= radius:
            mask[i, j] = 1
            
# Apply the mask to the shifted Fourier-transformed image
F_filtered = F_shifted * mask


# Shift back the frequencies
F_inverse_shift = np.fft.ifftshift(F_filtered)
# Perform the inverse 2D Fourier transform
image_filtered = np.fft.ifft2(F_inverse_shift)
# Take the real part of the inverse transform (the result might be complex)
image_filtered = np.abs(image_filtered)



#Adding Gaussian Noise to the image
variance = 100
mean = 0  
noise = np.random.normal(mean, np.sqrt(variance), image_filtered.shape)
noisy_image = image_filtered + noise
noisy_image = np.clip(noisy_image, 0, 255)
noisy_image = noisy_image.astype(np.uint8)



def wiener_filter(noisy_image, original_image, noise_var):
    # Estimate the local mean and variance of the original image
    original_mean = np.mean(original_image)
    original_var = np.var(original_image)

    # Compute the Wiener filter
    K = noise_var / original_var
    h, w = noisy_image.shape
    filtered_image = np.zeros_like(noisy_image)

    # Apply Wiener filtering
    for i in range(h):
        for j in range(w):
            # Define the local region
            local_region = noisy_image[max(i-1, 0):min(i+2, h), max(j-1, 0):min(j+2, w)]
            local_mean = np.mean(local_region)
            local_var = np.var(local_region)

            # Wiener filter equation
            if local_var > 0:
                filtered_image[i, j] = local_mean + K * (original_image[i, j] - local_mean)
            else:
                filtered_image[i, j] = noisy_image[i, j]

    return filtered_image

def calculate_mse(original, restored):
    return np.mean((original - restored) ** 2)

def calculate_psnr(original, restored):
    mse = calculate_mse(original, restored)
    if mse == 0:
        return float('inf')  # If MSE is 0, PSNR is infinite
    max_pixel_value = 255.0  # Assuming the image pixel values are in range [0, 255]
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr

restored_image = wiener_filter(noisy_image, img, variance)
plt.imsave("./restored_image_part1.png", restored_image, cmap='gray')
mse_value = calculate_mse(img, restored_image)
psnr_value = calculate_psnr(img, restored_image)

print(f'Mean Squared Error (MSE): {mse_value}')
print(f'Peak Signal-to-Noise Ratio (PSNR): {psnr_value} dB')

output_txt_path = 'psnr_mse_values_part1.txt'
with open(output_txt_path, 'w') as f:
    f.write(f'Mean Squared Error (MSE): {mse_value}\n')
    f.write(f'Peak Signal-to-Noise Ratio (PSNR): {psnr_value} dB\n')

# Plot the original, greyscale, filtered, noisy, and restored images
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(originalImg)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Greyscale Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Filtered Image (Ideal Low-Pass)')
plt.imshow(image_filtered, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Noisy Image (Gaussian Noise)')
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Restored Image (Wiener Filter)')
plt.imshow(restored_image, cmap='gray')
plt.axis('off')

plt.show()
