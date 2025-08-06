from flask import Flask, request, render_template, send_from_directory
import cv2
import numpy as np
import os
import math
import uuid
from datetime import datetime # Import datetime for current year in footer
from numba import njit, prange


@njit
def find_largest_square(mask_shape):
    h_img, w_img = mask_shape.shape
    dp = np.zeros((h_img, w_img), dtype=np.int32)
    max_side = 0
    best_square = (-1, -1, -1)

    for y in range(h_img):
        for x in range(w_img):
            if mask_shape[y, x] == 255:
                if y == 0 or x == 0:
                    dp[y, x] = 1
                else:
                    dp[y, x] = min(dp[y-1, x], dp[y, x-1], dp[y-1, x-1]) + 1

                if dp[y, x] > max_side:
                    max_side = dp[y, x]
                    best_square = (x - max_side + 1, y - max_side + 1, max_side)

    return best_square

# --- Image processing functions ---
def compute_spiral_length(a, b, theta_max, step=0.05):
    """
    Calculates the length of an Archimedean spiral.
    Parameters:
        a (float): Starting radius.
        b (float): Factor determining how tightly the spiral is coiled.
        theta_max (float): Maximum angle (in radians) to compute the spiral length.
        step (float): Step size for theta to approximate the integral.
    Returns:
        float: The approximated length of the spiral.
    """
    length = 0
    theta = 0
    while theta <= theta_max:
        r = a + b * theta
        r_next = a + b * (theta + step)
        # Approximating arc length using Pythagorean theorem for small steps
        length += np.sqrt((r_next - r) ** 2 + (r * step) ** 2)
        theta += step
    return length

def compute_angle_from_centroid(point, cx, cy):
    """
    Computes the angle of a point relative to a centroid.
    Parameters:
        point (tuple): (x, y) coordinates of the point.
        cx (int): X-coordinate of the centroid.
        cy (int): Y-coordinate of the centroid.
    Returns:
        float: Angle in degrees (0-360).
    """
    dx = point[0] - cx
    dy = point[1] - cy
    angle_rad = math.atan2(dy, dx) # atan2 returns angle in radians (-pi to pi)
    angle_deg = math.degrees(angle_rad) # Convert to degrees
    return angle_deg + 360 if angle_deg < 0 else angle_deg # Normalize to 0-360 degrees

# --- End of image processing functions ---

app = Flask(__name__)
# Configuration for uploaded and result folders
app.config['UPLOAD_FOLDER'] = 'uploaded_images'
app.config['RESULT_FOLDER'] = 'static/results'

# Create necessary directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# New route for the main Home page
@app.route('/')
def home():
    """
    Renders the new Home page.
    Passes the current datetime object to the template for footer year display.
    """
    return render_template('home.html', now=datetime.now())

# Route for the thickness analysis application (formerly '/')
@app.route('/analyzer')
def analyzer_app():
    """
    Renders the main application page for thickness analysis.
    Passes the current datetime object to the template for footer year display.
    """
    return render_template('index.html', now=datetime.now())

# Route to handle image uploads and processing
@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handles file upload, processes the image to calculate coconut thickness,
    and renders the results on the analyzer page.
    """
    # Check if a file part is in the request
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    # If no file is selected
    if file.filename == '':
        return "No selected file"
    
    if file:
        # Generate a unique filename to prevent overwriting
        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath) # Save the uploaded file

        # Read the image using OpenCV
        img_full = cv2.imread(filepath)
        if img_full is None:
            # Handle error if image cannot be read
            return "Error: Could not read image."

        # Convert image to RGB and HSV color spaces
        img_rgb_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
        hsv_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2HSV)

        # Define color ranges for green (coconut husk)
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv_full, lower_green, upper_green)

        # Define specific RGB colors for the edge/border (coconut shell)
        edge_rgbs = [
            (209, 188, 102), (203, 172, 90), (203, 184, 95), (135, 120, 37),
            (130, 101, 16), (150, 100, 23), (136, 113, 32), (140, 108, 27)
        ]
        # Convert RGB to BGR for OpenCV, then to HSV
        edge_bgrs = np.array([(c[2], c[1], c[0]) for c in edge_rgbs], dtype=np.uint8)
        edge_hsvs = cv2.cvtColor(edge_bgrs.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        
        # Determine min/max HSV values for the edge color range
        lower_edge = np.min(edge_hsvs, axis=0)
        upper_edge = np.max(edge_hsvs, axis=0)
        mask_edge = cv2.inRange(hsv_full, lower_edge, upper_edge)

        # Combine green and edge masks
        mask_final = cv2.bitwise_or(mask_green, mask_edge)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((15, 15), np.uint8)
        mask_closed = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours in the processed mask
        contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return "❌ ไม่พบคอนทัวร์" # No contours found

        # Combine all contour points and find the convex hull
        all_points = np.vstack([cnt[:, 0, :] for cnt in contours])
        hull = cv2.convexHull(all_points)

        # Calculate centroid of the overall hull (for full image)
        M = cv2.moments(hull)
        centroid_x = int(M['m10'] / M['m00']) if M['m00'] != 0 else img_full.shape[1] // 2
        centroid_y = int(M['m01'] / M['m00']) if M['m00'] != 0 else img_full.shape[0] // 2

        # Crop image around the centroid
        crop_size = 2000
        start_x = max(0, centroid_x - crop_size // 2)
        start_y = max(0, centroid_y - crop_size // 2)
        
        # Adjust crop boundaries if they exceed image dimensions
        if start_x + crop_size > img_full.shape[1]:
            start_x = img_full.shape[1] - crop_size
        if start_y + crop_size > img_full.shape[0]:
            start_y = img_full.shape[0] - crop_size

        img_crop = img_full[start_y:start_y + crop_size, start_x:start_x + crop_size]
        mask_crop = mask_opened[start_y:start_y + crop_size, start_x:start_x + crop_size]
        img_crop_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)

        # Find contours and convex hull in the cropped image
        contours_crop, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_points_crop = np.vstack([cnt[:, 0, :] for cnt in contours_crop])
        hull_crop = cv2.convexHull(all_points_crop)

        # Prepare output image for drawing
        output = img_crop_rgb.copy()
        cv2.drawContours(output, [hull_crop], -1, (255, 0, 0), 3) # Draw the hull

        # Calculate centroid of the cropped hull
        M_crop = cv2.moments(hull_crop)
        centroid_x_crop = int(M_crop['m10'] / M_crop['m00']) if M_crop['m00'] != 0 else crop_size // 2
        centroid_y_crop = int(M_crop['m01'] / M_crop['m00']) if M_crop['m00'] != 0 else crop_size // 2
        cv2.circle(output, (centroid_x_crop, centroid_y_crop), 7, (255, 0, 255), -1)

        h_img, w_img = output.shape[:2]
        mask_shape = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(mask_shape, [hull_crop], -1, 255, thickness=-1)

        best_square = find_largest_square(mask_shape)

        spiral_img = output.copy() # Image for spiral drawing
        radius_s = None
        side = None
        if best_square:
            x_s, y_s, side = best_square
            center_s = (x_s + side // 2, y_s + side // 2)
            radius_s = side // 2
            cv2.rectangle(spiral_img, (x_s, y_s), (x_s + side, y_s + side), (255, 127, 0), 2) # Draw square
            cv2.circle(spiral_img, center_s, radius_s, (255, 0, 0), 2) # Draw inscribed circle

        # Spiral parameters
        a, b, max_theta, step = 0, 2, 320 * np.pi, 0.05
        spiral_points_1, spiral_points_2 = [], []
        intersection_point_1 = intersection_point_2 = None
        theta_intersected_1 = theta_intersected_2 = None

        if radius_s is not None:
            # Spiral 1: Intersects with the inscribed circle
            theta = 0
            while theta <= max_theta:
                r = a + b * theta
                x = centroid_x_crop + r * np.cos(theta)
                y = centroid_y_crop + r * np.sin(theta)
                spiral_points_1.append((x, y))
                # Check if spiral point is outside the inscribed circle
                if np.linalg.norm([x - center_s[0], y - center_s[1]]) >= radius_s:
                    intersection_point_1 = (x, y)
                    theta_intersected_1 = theta
                    break
                theta += step

            # Spiral 2: Intersects with the coconut hull contour
            inside_before = True
            theta = 0
            while theta <= max_theta:
                r = a + b * theta
                x = centroid_x_crop + r * np.cos(theta)
                y = centroid_y_crop + r * np.sin(theta)
                spiral_points_2.append((x, y))

                # Check if the spiral point is inside the hull
                inside_now = cv2.pointPolygonTest(hull_crop, (x, y), False) >= 0

                # If it was inside and now it's outside, we found the intersection
                if inside_before and not inside_now:
                    intersection_point_2 = (x, y)
                    theta_intersected_2 = theta
                    break

                inside_before = inside_now
                theta += step

        # Calculate angles to intersection points
        if intersection_point_1:
            angle_deg_1 = compute_angle_from_centroid(intersection_point_1, centroid_x_crop, centroid_y_crop)
        else:
            angle_deg_1 = 0

        if intersection_point_2:
            angle_deg_2 = compute_angle_from_centroid(intersection_point_2, centroid_x_crop, centroid_y_crop)
        else:
            angle_deg_2 = 0

        # Draw intersection points and lines
        if intersection_point_1:
            cv2.circle(spiral_img, (int(intersection_point_1[0]), int(intersection_point_1[1])), 5, (255, 0, 255), -1)
        if intersection_point_2:
            cv2.circle(spiral_img, (int(intersection_point_2[0]), int(intersection_point_2[1])), 5, (0, 255, 255), -1)

        # Draw spirals
        if len(spiral_points_1) > 1:
            spiral_pts_array = np.array(spiral_points_1, dtype=np.int32)
            cv2.polylines(spiral_img, [spiral_pts_array], False, (0, 0, 255), 2)

        if len(spiral_points_2) > 1:
            spiral_pts_array_2 = np.array(spiral_points_2, dtype=np.int32)
            cv2.polylines(spiral_img, [spiral_pts_array_2], False, (0, 0, 255), 2)

        # Draw lines from centroid to intersection points
        if intersection_point_1:
            cv2.line(spiral_img, (int(centroid_x_crop), int(centroid_y_crop)), (int(intersection_point_1[0]), int(intersection_point_1[1])), (255, 0, 255), 2)
        if intersection_point_2:
            cv2.line(spiral_img, (int(centroid_x_crop), int(centroid_y_crop)), (int(intersection_point_2[0]), int(intersection_point_2[1])), (0, 255, 255), 2)

        # Calculate areas
        circle_area = np.pi * (radius_s ** 2) if radius_s else 0
        square_area = side * side if best_square else 0
        coconut_area = cv2.contourArea(hull_crop)

        # Calculate distances from centroid to intersections
        dist_centroid_to_intersection_1 = np.linalg.norm([intersection_point_1[0] - centroid_x_crop, intersection_point_1[1] - centroid_y_crop]) if intersection_point_1 else 0
        dist_centroid_to_intersection_2 = np.linalg.norm([intersection_point_2[0] - centroid_x_crop, intersection_point_2[1] - centroid_y_crop]) if intersection_point_2 else 0

        # Calculate spiral lengths
        spiral_length_1 = compute_spiral_length(a, b, theta_intersected_1, step) if theta_intersected_1 else 0
        spiral_length_2 = compute_spiral_length(a, b, theta_intersected_2, step) if theta_intersected_2 else 0

        # Calculate number of turns
        turns_1 = theta_intersected_1 / (2 * np.pi) if theta_intersected_1 else 0
        turns_2 = theta_intersected_2 / (2 * np.pi) if theta_intersected_2 else 0

        # Calculate thickness using the provided formula
        thickness = 0.016667 * 2 * ((turns_2 * 2 * np.pi) - (turns_1 * 2 * np.pi))

        # Save processed images for display on the web page
        output_filename_original = "original_" + unique_filename
        output_filename_cropped = "cropped_" + unique_filename
        output_filename_spiral = "spiral_" + unique_filename

        cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], output_filename_original), cv2.cvtColor(img_rgb_full, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], output_filename_cropped), cv2.cvtColor(img_crop_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], output_filename_spiral), cv2.cvtColor(spiral_img, cv2.COLOR_RGB2BGR))

        # Prepare results dictionary to pass to the template
        results = {
            "Circle Area": f"{circle_area:.0f} px²",
            "Square Area": f"{square_area} px²",
            "Coconut Area": f"{int(coconut_area)} px²",
            "Spiral 1 Distance to Intersection": f"{dist_centroid_to_intersection_1:.0f} px",
            "Spiral 2 Distance to Intersection": f"{dist_centroid_to_intersection_2:.0f} px",
            "Spiral 1 Angle": f"{angle_deg_1:.1f}°",
            "Spiral 2 Angle": f"{angle_deg_2:.1f}°",
            "Spiral 1 Length": f"{spiral_length_1:.2f} px",
            "Spiral 2 Length": f"{spiral_length_2:.2f} px",
            "Spiral 1 Number of Turns": f"{turns_1:.2f}",
            "Spiral 2 Number of Turns": f"{turns_2:.2f}",
            "Calculated Thickness": f"{thickness:.2f}",
            "original_image": output_filename_original,
            "cropped_image": output_filename_cropped,
            "spiral_image": output_filename_spiral
        }

        # Clean up the uploaded file after processing
        os.remove(filepath)

        # Render the index page with results and current datetime
        return render_template('index.html', results=results, now=datetime.now())

# Route for the "About" page
@app.route('/about')
def about():
    """
    Renders the about page.
    Passes the current datetime object to the template for footer year display.
    """
    return render_template('about.html', now=datetime.now())

# Route for the "Contact" page
@app.route('/contact')
def contact():
    """
    Renders the contact page.
    Passes the current datetime object to the template for footer year display.
    """
    return render_template('contact.html', now=datetime.now())
@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('404.html'), 404
# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
