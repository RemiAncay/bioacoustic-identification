"""
Audio Classification Game - Rémi Ancay - 08/07/2025

This script implements a simple audio classification game using Tkinter and Pygame.
It allows users to classify audio samples into predefined classes by dragging and dropping them into designated areas
and get feedback on their classification accuracy.

It prepares a dataset of audio files, creates a graphical user interface (GUI) for interaction,
and provides functionality to play audio, drag and drop files, validate user assignments, and reset the game state.
"""

# Imports
import os
import tkinter as tk
from tkinter import messagebox
import random
import pygame
import shutil
from itertools import permutations

# Constants
USED_CLASSES = ['dog_9', 'dog_10', 'dog_5']
SAMPLES_PER_CLASS = 5
DISPLAY_LABELS = ['A', 'B', 'C']

DATASET_PATH = "Codes/Datasets/BarkopediaIndividualDataset/test"
TEMP_PATH = "Codes/AudioClassificationGame/temp_audio"


def on_close():
    """
    Cleanup function to be called on application close.
    """
    pygame.mixer.quit()
    pygame.mixer.init()
    
    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)
    root.destroy()

def prepare_data():
    """
    Copies a sample of audio files from dataset into a temporary folder,
    assigns display labels and random display names for UI.
    Checks if the requested number of samples per class exists, else raises an error.
    Returns list of audio filenames, mapping of true labels, and display names.
    """
    audio_files = []
    true_labels = {}
    display_names = {}

    for real_class in USED_CLASSES:
        class_path = os.path.join(DATASET_PATH, real_class)
        files = [f for f in os.listdir(class_path) if f.endswith(".wav")]
        if len(files) < SAMPLES_PER_CLASS:
            raise RuntimeError(
                f"Class '{real_class}' has only {len(files)} audio files, "
                f"but {SAMPLES_PER_CLASS} samples are requested."
            )

    random_names = [f"{i+1}" for i in range(len(USED_CLASSES) * SAMPLES_PER_CLASS)]
    random.shuffle(random_names)

    counter = 1
    name_index = 0

    for real_class, display_label in zip(USED_CLASSES, DISPLAY_LABELS):
        class_path = os.path.join(DATASET_PATH, real_class)
        files = [f for f in os.listdir(class_path) if f.endswith(".wav")]
        selected = random.sample(files, min(SAMPLES_PER_CLASS, len(files)))
        for f in selected:
            new_name = f"{counter}.wav"
            shutil.copy(os.path.join(class_path, f), os.path.join(TEMP_PATH, new_name))
            audio_files.append(new_name)
            true_labels[new_name] = display_label
            display_names[new_name] = random_names[name_index]
            name_index += 1
            counter += 1

    random.shuffle(audio_files)
    return audio_files, true_labels, display_names

class AudioClassifierApp:
    """
    Main application class for audio classification.
    Handles UI layout, audio playback, drag-and-drop assignment,
    validation of user assignments, and resetting the experience.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Audio Classification Game")
        self.root.geometry("1100x600")
        self.root.configure(bg='white')
        self.root.resizable(True, True)

        self.assignments = {}
        self.audio_widgets = {}
        self.drag_data = {"widget": None, "filename": None}
        self.prevent_reposition = False

        self.class_areas = {}
        self.class_y = 160
        self.class_width = 300
        self.class_height = 350
        self.class_spacing = 40

        self.update_class_areas_position()
        self.create_audio_widgets()
        self.position_audio_widgets()

        self.validate_button = tk.Button(root, text="Validate", command=self.validate, font=("Arial", 14))
        self.validate_button.place(relx=0.35, rely=0.92, anchor=tk.CENTER)

        self.reset_button = tk.Button(root, text="Reset", command=self.reset_experience, font=("Arial", 14))
        self.reset_button.place(relx=0.65, rely=0.92, anchor=tk.CENTER)

        self.root.bind("<Motion>", self.on_motion)
        self.root.bind("<ButtonRelease-1>", self.on_drop)
        self.root.bind("<Configure>", self.on_resize)

    def update_class_areas_position(self):
        """
        Updates the position and size of class label areas based on window size.
        """
        window_width = self.root.winfo_width()
        total_width = len(DISPLAY_LABELS) * self.class_width + (len(DISPLAY_LABELS) - 1) * self.class_spacing
        start_x = max((window_width - total_width) // 2, 0)

        for i, label in enumerate(DISPLAY_LABELS):
            x = start_x + i * (self.class_width + self.class_spacing)
            if label in self.class_areas:
                self.class_areas[label].place(x=x, y=self.class_y, width=self.class_width, height=self.class_height)
            else:
                frame = tk.Label(self.root, text=label, bg=self.get_color(i), font=("Arial", 26), bd=4, relief=tk.RIDGE)
                frame.place(x=x, y=self.class_y, width=self.class_width, height=self.class_height)
                self.class_areas[label] = frame

    def get_color(self, index):
        """
        Returns a color string for a given index cycling through predefined colors.
        """
        colors = ['tomato', 'deepskyblue', 'orange', 'lightgreen', 'violet', 'gold']
        return colors[index % len(colors)]

    def create_audio_widgets(self):
        """
        Creates draggable audio widgets for all audio files.
        """
        for widget in self.audio_widgets.values():
            widget.destroy()
        self.audio_widgets.clear()
        self.assignments.clear()

        for filename in audio_files:
            self.add_audio_widget(filename, 0, 0)

    def position_audio_widgets(self):
        """
        Positions audio widgets in rows at the top if they are not assigned to any class area.
        Wraps to a second row if they exceed the window width.
        """
        if self.prevent_reposition:
            return

        margin_y = 20
        widget_width = 90
        widget_height = 50
        gap = 10
        window_width = self.root.winfo_width()

        x = gap
        y = margin_y
        max_x = window_width - widget_width - gap

        for filename in audio_files:
            if filename not in self.assignments:
                self.audio_widgets[filename].place_forget()
                if x > max_x:
                    x = gap
                    y += widget_height + gap
                self.audio_widgets[filename].place(x=x, y=y, width=widget_width, height=widget_height)
                x += widget_width + gap

    def add_audio_widget(self, filename, x, y, w=90, h=50):
        """
        Creates a draggable audio widget with play button and label.
        """
        if filename in self.audio_widgets:
            return

        wrapper = tk.Frame(self.root, bg='saddlebrown', bd=2, relief=tk.RAISED)
        wrapper.place(x=x, y=y, width=w, height=h)

        play_btn = tk.Button(wrapper, text='▶', command=lambda f=filename: self.play_audio(f), width=2, height=1)
        play_btn.place(x=5, y=10)

        label = tk.Label(wrapper, text=display_names[filename], bg='peru', fg='white', width=6, height=2)
        label.place(x=35, y=5)
        label.bind("<ButtonPress-1>", lambda e, w=wrapper, f=filename: self.start_drag(e, w, f))

        self.audio_widgets[filename] = wrapper

    def play_audio(self, filename):
        """
        Plays the audio file corresponding to the given filename.
        """
        path = os.path.join(TEMP_PATH, filename)
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()

    def start_drag(self, event, widget, filename):
        """
        Starts dragging the audio widget.
        """
        self.drag_data["widget"] = widget
        self.drag_data["filename"] = filename

    def on_motion(self, event):
        """
        Moves the dragged widget with the mouse cursor.
        """
        widget = self.drag_data["widget"]
        if widget:
            x = event.x_root - self.root.winfo_rootx() - 45
            y = event.y_root - self.root.winfo_rooty() - 25
            widget.place(x=x, y=y)

    def on_drop(self, event):
        """
        Drops the dragged widget. Assigns it to a class area if dropped inside one.
        Removes assignment if dropped outside.
        """
        widget = self.drag_data["widget"]
        filename = self.drag_data["filename"]
        if not widget or not filename:
            return

        x_root = event.x_root
        y_root = event.y_root

        dropped_in_area = False
        for label, frame in self.class_areas.items():
            fx, fy = frame.winfo_rootx(), frame.winfo_rooty()
            fw, fh = frame.winfo_width(), frame.winfo_height()
            if fx < x_root < fx + fw and fy < y_root < fy + fh:
                self.assignments[filename] = label
                dropped_in_area = True
                break

        if not dropped_in_area and filename in self.assignments:
            del self.assignments[filename]

        self.drag_data = {"widget": None, "filename": None}

    def validate(self):
        """
        Validates user groupings by comparing how samples are clustered.
        Full score is awarded if all samples from the same true class are grouped together,
        even if the group is placed in the wrong label zone.
        """
        self.prevent_reposition = True

        user_clusters = {}
        for filename, zone in self.assignments.items():
            user_clusters.setdefault(zone, set()).add(filename)

        true_clusters = {}
        for filename, true_label in true_labels.items():
            true_clusters.setdefault(true_label, set()).add(filename)

        # Try all permutations of DISPLAY_LABELS to match user zones to true labels
        best_match = 0
        for perm in permutations(DISPLAY_LABELS):
            match_count = 0
            mapping = dict(zip(DISPLAY_LABELS, perm))
            for zone, filenames in user_clusters.items():
                mapped_label = mapping.get(zone)
                if mapped_label and mapped_label in true_clusters:
                    match_count += len(filenames & true_clusters[mapped_label])
            if match_count > best_match:
                best_match = match_count

        # Color widgets based on correct clustering
        for filename in audio_files:
            true_label = true_labels[filename]
            color_index = DISPLAY_LABELS.index(true_label)
            self.audio_widgets[filename].configure(bg=self.get_color(color_index))


        total = len(true_labels)
        score = (best_match / total) * 100 if total > 0 else 0
        messagebox.showinfo("Result", f"Accuracy (based on grouping): {best_match}/{total} ({score:.2f}%)")

        self.prevent_reposition = False



    def reset_experience(self):
        """
        Stops all audio playback, unloads music, clears temp files,
        reloads data, clears assignments, recreates and repositions audio widgets.
        """
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception:
            pass

        pygame.mixer.quit()
        pygame.mixer.init()

        for f in os.listdir(TEMP_PATH):
            os.remove(os.path.join(TEMP_PATH, f))

        global audio_files, true_labels, display_names
        audio_files, true_labels, display_names = prepare_data()
        self.assignments.clear()
        self.prevent_reposition = False
        self.create_audio_widgets()
        self.position_audio_widgets()


    def on_resize(self, event):
        """
        Handles window resize events by repositioning class areas and audio widgets,
        unless a drag is in progress or repositioning is prevented.
        """
        if self.drag_data["widget"] is None and not self.prevent_reposition:
            self.update_class_areas_position()
            self.position_audio_widgets()


#Initialize
pygame.mixer.init()
os.makedirs(TEMP_PATH, exist_ok=True)
for f in os.listdir(TEMP_PATH):
    os.remove(os.path.join(TEMP_PATH, f))

audio_files, true_labels, display_names = prepare_data()


# Main entry point for the application
# Initializes the Tkinter root window, sets up the AudioClassifierApp,
# and handles cleanup on close.
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = AudioClassifierApp(root)
        root.protocol("WM_DELETE_WINDOW", on_close)
        root.mainloop()
    except RuntimeError as e:
        messagebox.showerror("Initialization Error", str(e))