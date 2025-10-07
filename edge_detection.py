import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional


def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load an image from the specified file path.

    Args:
        image_path (str): Path to the image file

    Returns:
        np.ndarray: Loaded image in BGR format, or None if loading fails
    """
    if not os.path.exists(image_path):
        print(f"Error: Image path does not exist: {image_path}")
        return None

    # Read image using OpenCV
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        return None

    return image


def apply_canny_edge_detection(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    blur_kernel_size: int = 5
) -> np.ndarray:
    """
    Apply Canny edge detection to an input image.

    This function converts the image to grayscale, applies Gaussian blur
    to reduce noise, and then performs Canny edge detection.

    Args:
        image (np.ndarray): Input image in BGR format
        low_threshold (int): Lower threshold for Canny edge detection (default: 50)
        high_threshold (int): Upper threshold for Canny edge detection (default: 150)
        blur_kernel_size (int): Kernel size for Gaussian blur (must be odd, default: 5)

    Returns:
        np.ndarray: Edge-detected binary image where edges are white (255) on black (0)
    """
    # Convert image to grayscale for edge detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve edge detection quality
    blurred_image = cv2.GaussianBlur(gray_image, (blur_kernel_size, blur_kernel_size), 0)

    # Apply Canny edge detection algorithm
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

    return edges


def save_edge_image(edge_image: np.ndarray, output_path: str, original_filename: str) -> str:
    """
    Save the edge-detected image to the specified output directory.

    Args:
        edge_image (np.ndarray): Edge-detected binary image
        output_path (str): Directory path where the image will be saved
        original_filename (str): Original filename (used to create output filename)

    Returns:
        str: Full path of the saved image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Generate output filename by adding '_edges' suffix before file extension
    base_name = os.path.splitext(original_filename)[0]
    extension = os.path.splitext(original_filename)[1]
    output_filename = f"{base_name}_edges{extension}"

    # Construct full output file path
    output_file_path = os.path.join(output_path, output_filename)

    # Save the edge-detected image
    cv2.imwrite(output_file_path, edge_image)

    return output_file_path


def visualize_comparison(original: np.ndarray, edges: np.ndarray, title: str = "Edge Detection") -> None:
    """
    Display side-by-side comparison of original and edge-detected images.

    Args:
        original (np.ndarray): Original image in BGR format
        edges (np.ndarray): Edge-detected binary image
        title (str): Title for the visualization window
    """
    # Convert BGR to RGB for correct color display with matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Create figure with two subplots
    plt.figure(figsize=(12, 5))

    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title("Original Image")
    plt.axis('off')

    # Display edge-detected image
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection (Canny)")
    plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def perform_edge_detection(
    input_path: str,
    output_path: Optional[str] = None,
    show_preview: bool = False,
    low_threshold: int = 50,
    high_threshold: int = 150,
    blur_kernel_size: int = 5
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Main function to perform edge detection on single image or batch of images.

    This function handles both single image files and directories containing
    multiple images. It applies Canny edge detection and optionally saves
    the results and displays previews.

    Args:
        input_path (str): Path to a single image file or directory containing images
        output_path (str, optional): Directory path to save processed images (default: None)
        show_preview (bool): Whether to display side-by-side comparison (default: False)
        low_threshold (int): Lower threshold for Canny algorithm (default: 50)
        high_threshold (int): Upper threshold for Canny algorithm (default: 150)
        blur_kernel_size (int): Gaussian blur kernel size (default: 5)

    Returns:
        Union[np.ndarray, List[np.ndarray]]: Single edge-detected image array or
                                              list of edge-detected image arrays
    """
    # List to store all processed edge images
    processed_edges = []

    # Determine if input is a file or directory
    if os.path.isfile(input_path):
        # Single image processing
        print(f"Processing single image: {input_path}")

        # Load the image
        original_image = load_image(input_path)
        if original_image is None:
            return None

        # Apply edge detection
        edge_image = apply_canny_edge_detection(
            original_image,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            blur_kernel_size=blur_kernel_size
        )

        # Save processed image if output path is specified
        if output_path:
            filename = os.path.basename(input_path)
            saved_path = save_edge_image(edge_image, output_path, filename)
            print(f"Saved edge-detected image to: {saved_path}")

        # Display comparison if preview is enabled
        if show_preview:
            visualize_comparison(original_image, edge_image, title=os.path.basename(input_path))

        return edge_image

    elif os.path.isdir(input_path):
        # Batch processing for directory
        print(f"Processing images in directory: {input_path}")

        # Supported image file extensions
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']

        # Collect all image files in the directory
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_path, ext)))
            # Also check for uppercase extensions
            image_files.extend(glob.glob(os.path.join(input_path, ext.upper())))

        # Remove duplicates and sort
        image_files = sorted(list(set(image_files)))

        if not image_files:
            print(f"No image files found in directory: {input_path}")
            return []

        print(f"Found {len(image_files)} image(s) to process")

        # Process each image in the directory
        for idx, image_path in enumerate(image_files, 1):
            print(f"[{idx}/{len(image_files)}] Processing: {os.path.basename(image_path)}")

            # Load the image
            original_image = load_image(image_path)
            if original_image is None:
                continue

            # Apply edge detection
            edge_image = apply_canny_edge_detection(
                original_image,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                blur_kernel_size=blur_kernel_size
            )

            # Save processed image if output path is specified
            if output_path:
                filename = os.path.basename(image_path)
                saved_path = save_edge_image(edge_image, output_path, filename)
                print(f"  -> Saved to: {saved_path}")

            # Display comparison for first few images if preview is enabled
            if show_preview and idx <= 3:  # Limit preview to first 3 images
                visualize_comparison(original_image, edge_image, 
                                   title=f"[{idx}] {os.path.basename(image_path)}")

            # Add to processed list
            processed_edges.append(edge_image)

        print(f"\nCompleted processing {len(processed_edges)} image(s)")
        return processed_edges

    else:
        print(f"Error: Invalid input path: {input_path}")
        return None



if __name__ == "__main__":
    """
    Example usage demonstrating different use cases of the edge detection module.
    """
    print("=" * 70)
    print("Edge Detection Preprocessing for Plant Disease Detection")
    print("=" * 70)

    print("\n[Example 1] Single Image Processing")
    print("-" * 70)

    single_image_path = "images\image (2).JPG"
    edges = perform_edge_detection(
        input_path=single_image_path,
        output_path="./output_edges",
        show_preview=True,
        low_threshold=50,
        high_threshold=150
    )
    print(f"Output shape: {edges.shape}")

    # print("\n[Example 2] Batch Directory Processing")
    # print("-" * 70)

    # Uncomment and modify the path to test with your own directory
    # input_directory = "path/to/your/plant_images_folder"
    # edge_list = perform_edge_detection(
    #     input_path=input_directory,
    #     output_path="./processed_edges",
    #     show_preview=False,
    #     low_threshold=50,
    #     high_threshold=150,
    #     blur_kernel_size=5
    # )
    # print(f"Processed {len(edge_list)} images")

