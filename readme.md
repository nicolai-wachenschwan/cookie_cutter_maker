# 🍪 Advanced Cookie Cutter Generator

This project provides a web-based tool to convert any black and white image or scribble into a 3D-printable cookie cutter. The application is built with Streamlit and uses OpenCV for image processing and Trimesh for 3D mesh generation.

Try it out here: [Cookie Cutter Maker](https://cookiecuttermaker.streamlit.app/)

## Motivation

The goal of this project is to provide a fast and easy way to create custom cookie cutters. You can draw a shape, take a picture, and have a 3D model ready for printing in minutes.

## How it Works

The application follows a simple workflow to generate the cookie cutter:

1.  **Image Upload**: The user uploads an image (PNG, JPG, etc.).
2.  **Image Processing**: The uploaded image is processed to create a heightmap.
    *   The image is converted to grayscale.
    *   Adaptive thresholding is applied to create a binary image.
    *   Morphological operations are used to clean up the image.
    *   A heightmap is generated where different grayscale values represent different heights of the cookie cutter (e.g., cutting edge, rim).
3.  **3D Mesh Generation**: The heightmap is converted into a 3D mesh.
    *   A 3D surface is generated from the heightmap.
    *   A flat base is created using Delaunay triangulation to ensure the model is watertight.
    *   The surface and base are merged into a single mesh.
    *   The final mesh is scaled and centered according to the user's specifications.
4.  **3D Preview and Download**:
    *   An interactive 3D preview of the cookie cutter is displayed.
    *   The user can download the final model as an STL file, ready for 3D printing.

## Function Structure

The codebase is organized into three main Python files:

*   `app.py`: Handles the Streamlit user interface, user inputs, and orchestrates the overall process.
*   `heightmap.py`: Contains all the functions related to image processing. The main function is `process_image()`, which takes a PIL image and a set of parameters and returns the heightmap and an insert map.
*   `mesh.py`: Contains all the functions for 3D mesh generation. The main function is `generate_mesh()`, which takes a heightmap and parameters and returns a `trimesh` object.

## Local Installation

To run the application locally, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Comment out `start_xvfb()`**:
    In `app.py`, find the line `start_xvfb()` and comment it out. This is only needed for headless rendering on servers like Streamlit Cloud.
    ```python
    # In app.py
    # start_xvfb()
    ```

4.  **Run the app**:
    ```bash
    streamlit run app.py
    ```

## Usage

1.  Open the web application in your browser (or run it locally).
2.  Use the sidebar to adjust the parameters for your cookie cutter, such as size, height, and wall thickness.
3.  Upload a black and white image of the desired shape.
4.  The application will process the image and display the generated heightmap.
5.  Click the "Generate 3D Mesh" button to create the 3D model.
6.  Preview the 3D model in the interactive viewer.
7.  Download the STL file and send it to your 3D printer.

## Dependencies

The project relies on the following main libraries:

*   **Streamlit**: For the web application framework.
*   **OpenCV**: For image processing.
*   **Trimesh**: For 3D mesh creation and manipulation.
*   **NumPy, SciPy, Scikit-image**: For numerical operations and image processing.
*   **stpyvista, PyVista, VTK**: For 3D visualization within Streamlit.

For a full list of Python dependencies, see `requirements.txt`. For system-level dependencies (for Streamlit Cloud), see `packages.txt`.
