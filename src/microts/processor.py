"""
Time Series Processor Plugin for Napari

This plugin provides utilities for navigating through time-series microscopy images
and saving segmentation masks and tracking data for further analysis.
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
from contextlib import contextmanager

import napari
from napari.layers import Image, Labels, Tracks
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import (QVBoxLayout, QPushButton, QWidget, QFileDialog,
                            QLabel, QHBoxLayout, QGroupBox)
from qtpy.QtCore import Qt


class TimeSeriesProcessor(QWidget):
    """
    A Napari plugin for processing microscopy time series data.

    This plugin allows users to:
    - Navigate through multiple TIF files in a directory
    - Save segmentation masks and tracking data
    - Use keyboard shortcuts for efficient workflow
    """

    def __init__(self, napari_viewer: napari.Viewer):
        """
        Initialize the Time Series Processor widget.

        Parameters
        ----------
        napari_viewer : napari.Viewer
            The napari viewer instance
        """
        super().__init__()
        self.viewer = napari_viewer
        self.input_dir = ""
        self.output_dir = ""
        self.tif_files: List[str] = []
        self.current_index = -1

        # Cache for last loaded filename
        self.last_loaded_file = ""

        # Set up the UI
        self.setup_ui()

        # Set up keyboard shortcuts
        self.setup_shortcuts()

    def setup_ui(self) -> None:
        """Create and arrange the UI elements for the plugin."""
        main_layout = QVBoxLayout()

        # Directory input/output group
        dir_group = QGroupBox("Directories")
        dir_layout = QVBoxLayout()

        # Input directory selection
        input_layout = QVBoxLayout()
        self.input_label = QLabel("Input Directory:")
        input_layout.addWidget(self.input_label)

        self.input_btn = QPushButton("Select Input Directory")
        self.input_btn.clicked.connect(self.select_input_dir)
        input_layout.addWidget(self.input_btn)

        self.input_path_label = QLabel("No directory selected")
        self.input_path_label.setWordWrap(True)
        input_layout.addWidget(self.input_path_label)
        dir_layout.addLayout(input_layout)

        # Output directory selection
        output_layout = QVBoxLayout()
        self.output_label = QLabel("Output Directory:")
        output_layout.addWidget(self.output_label)

        self.output_btn = QPushButton("Select Output Directory")
        self.output_btn.clicked.connect(self.select_output_dir)
        output_layout.addWidget(self.output_btn)

        self.output_path_label = QLabel("No directory selected")
        self.output_path_label.setWordWrap(True)
        output_layout.addWidget(self.output_path_label)
        dir_layout.addLayout(output_layout)

        dir_group.setLayout(dir_layout)
        main_layout.addWidget(dir_group)

        # Navigation group
        nav_group = QGroupBox("Navigation")
        nav_layout = QVBoxLayout()

        # Navigation buttons
        button_layout = QHBoxLayout()

        self.prev_btn = QPushButton("Previous File [P]")
        self.prev_btn.clicked.connect(self.load_previous_file)
        button_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next File [N]")
        self.next_btn.clicked.connect(self.load_next_file)
        button_layout.addWidget(self.next_btn)

        nav_layout.addLayout(button_layout)

        # Save button
        self.save_btn = QPushButton("Save Mask + Track [W]")
        self.save_btn.clicked.connect(self.save_mask_and_track)
        nav_layout.addWidget(self.save_btn)

        nav_group.setLayout(nav_layout)
        main_layout.addWidget(nav_group)

        # File info label
        self.file_info_label = QLabel("No file loaded")
        self.file_info_label.setWordWrap(True)
        self.file_info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.file_info_label)

        # Status label for operation feedback
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

        # Set minimal width for better visibility
        self.setMinimumWidth(300)
        self.setLayout(main_layout)

    def setup_shortcuts(self) -> None:
        """Set up keyboard shortcuts for the plugin."""
        try:
            # Next file shortcut
            @self.viewer.bind_key('N')
            def next_file(viewer):
                self.load_next_file()

            # Previous file shortcut
            @self.viewer.bind_key('P')
            def prev_file(viewer):
                self.load_previous_file()

            # Save shortcut
            @self.viewer.bind_key('W')
            def save_mask_track(viewer):
                self.save_mask_and_track()

        except Exception as e:
            self.update_status(f"Error setting up shortcuts: {str(e)}", is_error=True)

    def select_input_dir(self) -> None:
        """Open a file dialog to select the input directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.input_dir = directory
            self.input_path_label.setText(directory)
            self._refresh_file_list()

    def select_output_dir(self) -> None:
        """Open a file dialog to select the output directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir = directory
            self.output_path_label.setText(directory)
            self.update_status("Output directory set")

    def _refresh_file_list(self) -> None:
        """Refresh the list of .tif files from the input directory."""
        if not self.input_dir:
            return

        try:
            path = Path(self.input_dir)
            self.tif_files = sorted([str(f) for f in path.glob("*.tif")])
            self.current_index = -1

            if self.tif_files:
                self.update_status(f"Found {len(self.tif_files)} .tif files")
            else:
                self.update_status("No .tif files found in directory", is_error=True)
        except Exception as e:
            self.update_status(f"Error refreshing file list: {str(e)}", is_error=True)

    def load_next_file(self) -> None:
        """Load the next .tif file in the list."""
        if not self.tif_files:
            self.update_status("No .tif files available", is_error=True)
            return

        if self.current_index < len(self.tif_files) - 1:
            # Remove previous image layer
            self._remove_image_layers()

            self.current_index += 1
            self._load_current_file()
        else:
            self.update_status("Already at the last file", is_error=True)

    def load_previous_file(self) -> None:
        """Load the previous .tif file in the list."""
        if not self.tif_files:
            self.update_status("No .tif files available", is_error=True)
            return

        if self.current_index > 0:
            # Remove previous image layer
            self._remove_image_layers()

            self.current_index -= 1
            self._load_current_file()
        else:
            self.update_status("Already at the first file", is_error=True)

    def _remove_image_layers(self) -> None:
        """Remove all image layers from the viewer."""
        for layer in list(self.viewer.layers):
            if isinstance(layer, Image):
                self.viewer.layers.remove(layer.name)

    def _load_current_file(self) -> None:
        """Load the current file into napari."""
        if not self.tif_files or self.current_index < 0 or self.current_index >= len(self.tif_files):
            return

        file_path = self.tif_files[self.current_index]
        file_name = os.path.basename(file_path)

        # Cache the filename
        self.last_loaded_file = file_name

        try:
            # Use napari's built-in TIF reader to load image layer
            self.update_status(f"Loading {file_name}...")
            layer_data = self.viewer.open(file_path, plugin="napari")[0]

            # Ensure image layer is at the bottom by moving it
            for layer in self.viewer.layers:
                if isinstance(layer, Image):
                    self.viewer.layers.move(len(self.viewer.layers) - 1, 0)  # Move to bottom (index 0)
                    break

            # Update info label
            self.file_info_label.setText(f"{file_name} ({self.current_index + 1}/{len(self.tif_files)})")
            self.update_status("File loaded successfully")
        except Exception as e:
            self.update_status(f"Error loading {file_name}: {str(e)}", is_error=True)

    def save_mask_and_track(self) -> None:
        """Save the committed_objects masks and tracks to the output directory."""
        if not self.output_dir:
            self.update_status("No output directory selected", is_error=True)
            return

        if not self.tif_files or self.current_index < 0 or self.current_index >= len(self.tif_files):
            self.update_status("No file loaded", is_error=True)
            return

        file_path = self.tif_files[self.current_index]
        file_name = os.path.splitext(os.path.basename(file_path))[0]  # Get filename without extension

        saved_items = []

        # Find and save the required layers
        committed_layer = self._get_layer_by_name_and_type("committed_objects", Labels)
        tracks_layer = self._get_layer_by_name_and_type("tracks", Tracks)

        # Save the committed_objects layer (segmentation masks)
        if committed_layer is not None:
            if self._save_mask_layer(file_name, committed_layer):
                saved_items.append("masks")
        else:
            self.update_status("No 'committed_objects' layer found", is_error=True)

        # Save the tracks layer
        if tracks_layer is not None and len(tracks_layer.data) > 0:
            if self._save_tracks_layer(file_name, tracks_layer):
                saved_items.append("tracks")
        else:
            self.update_status("No tracks data found", is_error=True)

        # Final status update
        if saved_items:
            self.update_status(f"Saved {' and '.join(saved_items)} for {file_name}")
        else:
            self.update_status("No data was saved", is_error=True)

    def _get_layer_by_name_and_type(self, layer_name: str, layer_type) -> Optional[Any]:
        """
        Get a layer by name and type from the viewer's layers.

        Parameters
        ----------
        layer_name : str
            Name of the layer to find
        layer_type : type
            Type of the layer to find

        Returns
        -------
        Optional[Any]
            The found layer or None
        """
        for layer in self.viewer.layers:
            if layer.name == layer_name and isinstance(layer, layer_type):
                return layer
        return None

    def _save_mask_layer(self, file_name: str, committed_layer: Labels) -> bool:
        """
        Save the segmentation mask layer to a TIFF file.

        Parameters
        ----------
        file_name : str
            Base name for the output file
        committed_layer : Labels
            The Labels layer to save

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        committed_path = os.path.join(self.output_dir, f"{file_name}_masks.tif")
        try:
            self.update_status(f"Saving masks to {committed_path}...")
            # Save as tif using napari's built-in function
            napari.save_layers(committed_path, [committed_layer])
            return True
        except Exception as e:
            self.update_status(f"Error saving masks: {str(e)}", is_error=True)
            return False

    def _save_tracks_layer(self, file_name: str, tracks_layer: Tracks) -> bool:
        """
        Save the tracks layer to a CSV file.

        Parameters
        ----------
        file_name : str
            Base name for the output file
        tracks_layer : Tracks
            The Tracks layer to save

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        tracks_path = os.path.join(self.output_dir, f"{file_name}_tracks.csv")
        try:
            self.update_status(f"Saving tracks to {tracks_path}...")
            # Get the tracks data
            tracks_data = tracks_layer.data.copy()

            # Convert ID and T columns to integers, keep Y and X as floats
            # The format is: (ID, T, Y, X)
            formatted_tracks = self._format_tracks_data(tracks_data)

            # Save as CSV with proper formats
            np.savetxt(
                tracks_path,
                formatted_tracks,
                delimiter=',',
                header='ID,T,Y,X',
                comments='',
                fmt='%d,%d,%.4f,%.4f'  # Integer format for ID and T, float format for Y and X
            )
            return True
        except Exception as e:
            self.update_status(f"Error saving tracks: {str(e)}", is_error=True)
            return False

    def _format_tracks_data(self, tracks_data: np.ndarray) -> np.ndarray:
        """
        Format the tracks data with proper data types.

        Parameters
        ----------
        tracks_data : np.ndarray
            Raw tracks data

        Returns
        -------
        np.ndarray
            Formatted tracks data with proper dtypes
        """
        # Create a structured array with proper dtypes
        formatted_tracks = np.zeros(tracks_data.shape[0],
                                    dtype=[('ID', np.int32),
                                           ('T', np.int32),
                                           ('Y', np.float32),
                                           ('X', np.float32)])

        # Fill the structured array with data, converting ID and T to integers
        formatted_tracks['ID'] = tracks_data[:, 0].astype(np.int32)
        formatted_tracks['T'] = tracks_data[:, 1].astype(np.int32)
        formatted_tracks['Y'] = tracks_data[:, 2].astype(np.float32)
        formatted_tracks['X'] = tracks_data[:, 3].astype(np.float32)

        return formatted_tracks

    def update_status(self, message: str, is_error: bool = False) -> None:
        """
        Update the status message displayed to the user.

        Parameters
        ----------
        message : str
            Message to display
        is_error : bool
            Whether this is an error message (changes style)
        """
        if is_error:
            self.status_label.setStyleSheet("color: red;")
        else:
            self.status_label.setStyleSheet("color: black;")
        self.status_label.setText(message)
        self.status_label.repaint()  # Force update


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    """Register the TimeSeriesProcessor widget with napari."""
    return TimeSeriesProcessor


if __name__ == "__main__":
    # For development/testing
    viewer = napari.Viewer()
    widget = TimeSeriesProcessor(viewer)
    viewer.window.add_dock_widget(widget, name="Time Series Processor")
    napari.run()