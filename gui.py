import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import yaml
import os
import sys
import subprocess
import chardet
import threading
import json
import tempfile
from themes import Themes
import customtkinter as ctk
import logging
from datetime import datetime

# Add the Windows-specific flag import
if sys.platform == "win32":
    from subprocess import CREATE_NO_WINDOW

class CreateToolTip:
    """Create a tooltip for a given widget"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind('<Enter>', self.enter)
        self.widget.bind('<Leave>', self.leave)

    def enter(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        label = ttk.Label(self.tooltip, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1)
        label.pack()

    def leave(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class FluxTrainingGUI(ctk.CTkFrame):
    def __init__(self, master=None, config=None):
        # Setup logging
        self.setup_logging()
        
        # Initialize theme first
        self.themes = Themes()
        self.is_dark_mode = self.load_theme_preference()
        
        # Default paths
        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        self.settings_file = os.path.join(self.app_dir, "last_settings.json")
        self.current_train_file = os.path.join(self.app_dir, "current_train.yaml")
        
        # Configure root window background color (using _configure_window for tk.Tk)
        if isinstance(master, tk.Tk):
            master.configure(bg=self.themes.get_theme(self.is_dark_mode)['bg'])
            master.minsize(800, 600)  # Set minimum size for root window
            # Set window protocol for root window
            master.protocol("WM_DELETE_WINDOW", self.on_closing)
        else:
            master.configure(fg_color=self.themes.get_theme(self.is_dark_mode)['bg'])
            # Find root window and set protocol
            root = self._find_root_window(master)
            if root:
                root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        super().__init__(master)
        self.master = master
        self._raw_config = config if config is not None else {}
        self.current_config_path = None
        self.current_config_label_var = tk.StringVar(value='No config loaded')
        self.batch_configs = []
        
        # Configure frame colors
        self.configure(fg_color=self.themes.get_theme(self.is_dark_mode)['bg'])
        
        # Initialize variables
        self.initialize_variables()
        
        # Create main layout
        self.create_widgets()
        
        # Pack main frame
        self.pack(fill="both", expand=True)
        self.grid_columnconfigure(0, weight=1)
        
        # Add trace to data_folder_var before loading settings
        self.data_folder_var.trace_add("write", lambda *args: self.update_dataset_stats(self.data_folder_var.get()))
        
        # Add trace to lr_var to update steps when learning rate changes
        self.lr_var.trace_add("write", lambda *args: self.update_steps_on_lr_change())
        
        # Add traces to all variables to auto-save settings
        self.setup_auto_save_traces()
        
        # Load settings after widgets are created
        self.load_last_settings()
        
        if config:
            self.load_config(config)
        
        # Explicitly update dataset stats after loading settings/config
        self.update_dataset_stats(self.data_folder_var.get())
        
        # Now apply theme after all widgets are created
        self.themes.apply_theme(self.master, self.is_dark_mode)
        self.add_theme_toggle()

        self.training_config_path = None
        self.current_config = {}  # Store the current configuration state
        
        # Add process tracking
        self.training_process = None

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gui.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _find_root_window(self, widget):
        """Find the root window by traversing up the widget hierarchy"""
        parent = widget.master
        while parent and not isinstance(parent, tk.Tk):
            parent = parent.master
        return parent

    def initialize_variables(self):
        """Initialize all variables used in the GUI"""
        # Initialize learning rate presets
        self.lr_presets = {
            '1e-4 (Standard)': {
                'value': '1e-4', 
                'description': 'Standard learning rate for most cases',
                'repeats': 150,
                'info': 'Each image will be seen ~150 times'
            },
            '2e-4 (Faster)': {
                'value': '2e-4', 
                'description': 'Faster learning, may be less stable',
                'repeats': 100,
                'info': 'Each image will be seen ~100 times'
            },
            '5e-5 (Conservative)': {
                'value': '5e-5', 
                'description': 'More conservative, more stable',
                'repeats': 200,
                'info': 'Each image will be seen ~200 times'
            },
            '3e-5 (Very Conservative)': {
                'value': '3e-5', 
                'description': 'Very conservative, very stable',
                'repeats': 250,
                'info': 'Each image will be seen ~250 times'
            },
            '2e-5 (Minimal)': {
                'value': '2e-5', 
                'description': 'Minimal changes, highest stability',
                'repeats': 300,
                'info': 'Each image will be seen ~300 times'
            }
        }
        
        # Initialize containers
        self.step_buttons = []
        self.resolution_vars = {
            "512x512": {"var": tk.BooleanVar(value=True), "description": "Standard training resolution"},
            "1024x1024": {"var": tk.BooleanVar(value=False), "description": "High resolution training"},
            "1280x1280": {"var": tk.BooleanVar(value=False), "description": "Very high resolution"},
            "1536x1536": {"var": tk.BooleanVar(value=False), "description": "Ultra high resolution"}
        }
        
        # Initialize all variables with defaults
        self.name_var = tk.StringVar(value='')
        self.trigger_word_var = tk.StringVar(value='')
        self.data_folder_var = tk.StringVar(value='')
        self.output_folder_var = tk.StringVar(value='')
        self.lr_var = tk.StringVar(value='1e-4 (Standard)')
        self.custom_lr_var = tk.StringVar()
        self.training_steps_var = tk.StringVar(value='1000')
        self.auto_steps_var = tk.BooleanVar(value=True)
        self.convert_utf8_var = tk.BooleanVar(value=True)
        self.prompt_style_var = tk.StringVar(value="Natural")
        self.sampling_steps_var = tk.StringVar(value="20")
        self.cfg_scale_var = tk.StringVar(value="3.5")
        self.save_dtype_var = tk.StringVar(value="float16")
        self.save_every_var = tk.StringVar(value="250")
        self.max_saves_var = tk.StringVar(value="4")
        self.model_path_var = tk.StringVar(value="")
        self.is_flux_var = tk.BooleanVar(value=True)
        self.quantize_var = tk.BooleanVar(value=True)
        self.dataset_stats_var = tk.StringVar(value='')
        self.enable_sampling_var = tk.BooleanVar(value=True)
        self.sample_every_var = tk.StringVar(value="500")
        self.seed_var = tk.StringVar(value="42")
        self.use_fixed_seed_var = tk.BooleanVar(value=True)
        
        # Network variables
        self.network_type_var = tk.StringVar(value="lora")
        self.linear_var = tk.StringVar(value="16")
        self.linear_alpha_var = tk.StringVar(value="16")
        
        # Training variables
        self.batch_size_var = tk.StringVar(value="1")
        self.steps_var = tk.StringVar(value="5000")
        self.grad_accum_var = tk.StringVar(value="1")
        self.train_unet_var = tk.BooleanVar(value=True)
        self.train_text_encoder_var = tk.BooleanVar(value=False)
        self.grad_checkpointing_var = tk.BooleanVar(value=True)
        self.noise_scheduler_var = tk.StringVar(value="flowmatch")
        self.optimizer_var = tk.StringVar(value="adamw8bit")
        
        # Add progress tracking variables
        self.step_var = tk.StringVar(value="Step: 0/0")
        self.status_var = tk.StringVar(value="Ready")

        # Add batch processing variables
        self.batch_configs = []  # List to store batch configurations
        self.batch_list_var = tk.StringVar(value=[])  # For listbox
        self.batch_status_var = tk.StringVar(value="Ready for batch processing")

        # Add caption management variables
        self.caption_folder_var = tk.StringVar(value='')
        self.caption_template_var = tk.StringVar(value='')
        self.caption_prefix_var = tk.StringVar(value='')
        self.caption_suffix_var = tk.StringVar(value='')
        self.overwrite_captions_var = tk.BooleanVar(value=False)

        # Add LLM vision variables
        self.use_llm_vision_var = tk.BooleanVar(value=False)
        self.llm_model_var = tk.StringVar(value="gpt-4-vision-preview")  # Default model
        self.llm_api_key_var = tk.StringVar(value="")
        self.llm_prompt_var = tk.StringVar(value="Describe this image in detail, focusing on visual elements")

        # Add local LLM options
        self.use_local_llm_var = tk.BooleanVar(value=False)
        self.local_api_url_var = tk.StringVar(value="http://localhost:1234/v1")
        self.local_model_temp_var = tk.StringVar(value="0.7")

        # Add variable for tracking current tab
        self.current_tab = None
        self.train_button = None  # Will store reference to train button

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        if event.num == 5 or event.delta < 0:  # Scroll down
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:  # Scroll up
            self.canvas.yview_scroll(-1, "units")

    def setup_scrollable_frame(self):
        """Set up the scrollable frame"""
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure grid weights for main frame
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Create window in canvas and make it expand
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=self.canvas.winfo_width())
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Configure canvas to expand
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Pack main frame
        self.pack(fill="both", expand=True)
        
        # Bind canvas resize
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        
        # Bind mouse wheel
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)
        
    def _on_canvas_configure(self, event):
        """Handle canvas resize"""
        # Update the width of the window in the canvas
        self.canvas.itemconfig(self.canvas.find_withtag('all')[0], width=event.width)

    def create_widgets(self):
        """Create all widgets for the GUI"""
        # Create main notebook
        self.main_notebook = ctk.CTkTabview(self)
        self.main_notebook.pack(fill="both", expand=True, padx=10, pady=10)
        self.main_notebook.configure(fg_color=self.themes.get_theme(self.is_dark_mode)['frame'])
        
        # Add tabs first
        training_tab = self.main_notebook.add("Training")
        sample_tab = self.main_notebook.add("Sample")
        batch_tab = self.main_notebook.add("Batch")
        caption_tab = self.main_notebook.add("Captions")
        
        # Set tab change callback after adding tabs
        self.main_notebook.configure(command=self.on_tab_change)
        
        # Configure tab colors
        training_tab.configure(fg_color=self.themes.get_theme(self.is_dark_mode)['frame'])
        sample_tab.configure(fg_color=self.themes.get_theme(self.is_dark_mode)['frame'])
        batch_tab.configure(fg_color=self.themes.get_theme(self.is_dark_mode)['frame'])
        caption_tab.configure(fg_color=self.themes.get_theme(self.is_dark_mode)['frame'])
        
        # Training tab
        training_tab.configure(fg_color=self.themes.get_theme(self.is_dark_mode)['frame'])
        
        # Left column in Training tab
        left_column = ctk.CTkFrame(training_tab)
        left_column.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Right column in Training tab
        right_column = ctk.CTkFrame(training_tab)
        right_column.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Basic Configuration (Left Column)
        basic_frame = ctk.CTkFrame(left_column)
        basic_frame.pack(fill='x', padx=5, pady=5)
        
        # Name and Trigger Word
        name_frame = ctk.CTkFrame(basic_frame)
        name_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(name_frame, text="Name:").pack(side='left')
        ctk.CTkEntry(name_frame, textvariable=self.name_var).pack(side='left', fill='x', expand=True, padx=5)
        
        trigger_frame = ctk.CTkFrame(basic_frame)
        trigger_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(trigger_frame, text="Trigger Word:").pack(side='left')
        ctk.CTkEntry(trigger_frame, textvariable=self.trigger_word_var).pack(side='left', fill='x', expand=True, padx=5)
        
        # Model Configuration
        model_frame = ctk.CTkFrame(left_column)
        model_frame.pack(fill='x', padx=5, pady=5)
        
        model_path_frame = ctk.CTkFrame(model_frame)
        model_path_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(model_path_frame, text="Model Path:").pack(side='left')
        ctk.CTkEntry(model_path_frame, textvariable=self.model_path_var).pack(side='left', fill='x', expand=True, padx=5)
        
        # Apply consistent button styling
        button_style = {
            'fg_color': self.themes.get_theme(self.is_dark_mode)['button'],
            'hover_color': self.themes.get_theme(self.is_dark_mode)['highlight'],
            'border_width': 0,
            'corner_radius': 6
        }
        
        # Apply consistent style to option menus
        option_menu_style = {
            'fg_color': self.themes.get_theme(self.is_dark_mode)['button'],
            'button_color': self.themes.get_theme(self.is_dark_mode)['button'],
            'button_hover_color': self.themes.get_theme(self.is_dark_mode)['highlight'],
            'dropdown_fg_color': self.themes.get_theme(self.is_dark_mode)['frame'],
            'dropdown_hover_color': self.themes.get_theme(self.is_dark_mode)['highlight'],
            'text_color': self.themes.get_theme(self.is_dark_mode)['text']
        }
        
        ctk.CTkButton(model_path_frame, text="Browse", command=lambda: self.browse_folder('model'), **button_style).pack(side='right')
        
        # Model Options
        model_options_frame = ctk.CTkFrame(model_frame)
        model_options_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkCheckBox(model_options_frame, text="Is Flux", variable=self.is_flux_var).pack(side='left', padx=5)
        ctk.CTkCheckBox(model_options_frame, text="Quantize", variable=self.quantize_var).pack(side='left', padx=5)
        
        # Dataset Configuration
        dataset_frame = ctk.CTkFrame(left_column)
        dataset_frame.pack(fill='x', padx=5, pady=5)
        
        ctk.CTkLabel(dataset_frame, text="Dataset Settings").pack(anchor='w', padx=5, pady=2)
        
        # Data folder selection
        data_folder_frame = ctk.CTkFrame(dataset_frame)
        data_folder_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(data_folder_frame, text="Data Folder:").pack(side='left')
        ctk.CTkEntry(data_folder_frame, textvariable=self.data_folder_var).pack(side='left', fill='x', expand=True, padx=5)
        ctk.CTkButton(data_folder_frame, text="Browse", command=lambda: self.browse_folder('data'), **button_style).pack(side='right')
        
        # Dataset statistics
        stats_frame = ctk.CTkFrame(dataset_frame)
        stats_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(stats_frame, text="Dataset Stats:").pack(side='left', padx=5)
        ctk.CTkLabel(stats_frame, textvariable=self.dataset_stats_var).pack(side='left', padx=5)
        
        # Output folder selection
        output_folder_frame = ctk.CTkFrame(dataset_frame)
        output_folder_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(output_folder_frame, text="Output Folder:").pack(side='left')
        ctk.CTkEntry(output_folder_frame, textvariable=self.output_folder_var).pack(side='left', fill='x', expand=True, padx=5)
        ctk.CTkButton(output_folder_frame, text="Browse", command=lambda: self.browse_folder('output'), **button_style).pack(side='right')
        
        # Dataset options
        dataset_options_frame = ctk.CTkFrame(dataset_frame)
        dataset_options_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkCheckBox(dataset_options_frame, text="Convert Captions to UTF-8", variable=self.convert_utf8_var).pack(side='left', padx=5)
        
        # Add resolution settings to dataset frame
        resolution_frame = ctk.CTkFrame(dataset_frame)
        resolution_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(resolution_frame, text="Training Resolutions:").pack(anchor='w', padx=5)
        
        # Create resolution checkboxes
        for res, data in self.resolution_vars.items():
            checkbox = ctk.CTkCheckBox(resolution_frame, text=res, variable=data['var'])
            checkbox.pack(side='left', padx=5)
            CreateToolTip(checkbox, data['description'])
        
        # Network Settings
        network_frame = ctk.CTkFrame(left_column)
        network_frame.pack(fill='x', padx=5, pady=5)
        
        ctk.CTkLabel(network_frame, text="Network Settings").pack(anchor='w', padx=5, pady=2)
        
        linear_frame = ctk.CTkFrame(network_frame)
        linear_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(linear_frame, text="Linear:").pack(side='left')
        ctk.CTkEntry(linear_frame, textvariable=self.linear_var, width=80).pack(side='left', padx=5)
        
        alpha_frame = ctk.CTkFrame(network_frame)
        alpha_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(alpha_frame, text="Alpha:").pack(side='left')
        ctk.CTkEntry(alpha_frame, textvariable=self.linear_alpha_var, width=80).pack(side='left', padx=5)
        
        # Training Configuration (Right Column)
        training_config_frame = ctk.CTkFrame(right_column)
        training_config_frame.pack(fill='x', padx=5, pady=5)
        
        ctk.CTkLabel(training_config_frame, text="Training Settings").pack(anchor='w', padx=5, pady=2)
        
        # Batch Size
        batch_frame = ctk.CTkFrame(training_config_frame)
        batch_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(batch_frame, text="Batch Size:").pack(side='left')
        ctk.CTkEntry(batch_frame, textvariable=self.batch_size_var, width=80).pack(side='left', padx=5)
        
        # Steps
        steps_frame = ctk.CTkFrame(training_config_frame)
        steps_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(steps_frame, text="Steps:").pack(side='left')
        self.steps_entry = ctk.CTkEntry(steps_frame, textvariable=self.steps_var, width=80)
        self.steps_entry.pack(side='left', padx=5)
        ctk.CTkCheckBox(steps_frame, text="Auto", variable=self.auto_steps_var, command=self.toggle_steps_entry).pack(side='left', padx=5)
        
        # Learning Rate
        lr_frame = ctk.CTkFrame(training_config_frame)
        lr_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(lr_frame, text="Learning Rate:").pack(side='left')
        lr_menu = ctk.CTkOptionMenu(lr_frame, 
                         variable=self.lr_var, 
                         values=list(self.lr_presets.keys()),
                         **option_menu_style)
        lr_menu.pack(side='left', padx=5)
        
        # Add info label for learning rate
        self.lr_info_label = ctk.CTkLabel(lr_frame, text=self.lr_presets[self.lr_var.get()]['info'])
        self.lr_info_label.pack(side='left', padx=5)
        
        # Update info label when learning rate changes
        def update_lr_info(*args):
            self.lr_info_label.configure(text=self.lr_presets[self.lr_var.get()]['info'])
            self.update_steps_on_lr_change()
        
        self.lr_var.trace_add("write", update_lr_info)
        
        # Training Options
        options_frame = ctk.CTkFrame(training_config_frame)
        options_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkCheckBox(options_frame, text="Train U-Net", variable=self.train_unet_var).pack(anchor='w', padx=5, pady=2)
        ctk.CTkCheckBox(options_frame, text="Train Text Encoder", variable=self.train_text_encoder_var).pack(anchor='w', padx=5, pady=2)
        ctk.CTkCheckBox(options_frame, text="Gradient Checkpointing", variable=self.grad_checkpointing_var).pack(anchor='w', padx=5, pady=2)
        
        # Save Configuration
        save_frame = ctk.CTkFrame(right_column)
        save_frame.pack(fill='x', padx=5, pady=5)
        
        ctk.CTkLabel(save_frame, text="Save Settings").pack(anchor='w', padx=5, pady=2)
        
        save_type_frame = ctk.CTkFrame(save_frame)
        save_type_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(save_type_frame, text="Save Precision:").pack(side='left')
        ctk.CTkOptionMenu(save_type_frame, 
                         variable=self.save_dtype_var, 
                         values=["float16", "float32"],
                         **option_menu_style).pack(side='left', padx=5)
        
        # Bottom Buttons
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill='x', padx=10, pady=5)
        
        # Apply consistent button styling
        button_style = {
            'fg_color': self.themes.get_theme(self.is_dark_mode)['button'],
            'hover_color': self.themes.get_theme(self.is_dark_mode)['highlight'],
            'border_width': 0,
            'corner_radius': 6
        }
        
        # Left side buttons
        left_buttons = ctk.CTkFrame(button_frame)
        left_buttons.pack(side='left')
        ctk.CTkButton(left_buttons, text="Load Config", command=self.load_config_file, **button_style).pack(side='left', padx=5)
        ctk.CTkButton(left_buttons, text="Save Config", command=self.save_config, **button_style).pack(side='left', padx=5)
        
        # Right side buttons
        right_buttons = ctk.CTkFrame(button_frame)
        right_buttons.pack(side='right')
        self.train_button = ctk.CTkButton(right_buttons, 
                                     text="Start Training", 
                                     command=self.handle_train_button,
                                     **button_style)
        self.train_button.pack(side='left', padx=5)
        
        self.stop_button = ctk.CTkButton(right_buttons, text="Stop Training", command=self.stop_training, 
                                        fg_color="#FF4444", hover_color="#CC3333", state="disabled")
        self.stop_button.pack(side='left', padx=5)
        
        # Progress Frame
        progress_frame = ctk.CTkFrame(self)
        progress_frame.pack(fill='x', padx=10, pady=5)
        progress_frame.configure(fg_color=self.themes.get_theme(self.is_dark_mode)['frame'])
        
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(fill='x', padx=5, pady=5)
        self.progress_bar.set(0)
        
        self.status_label = ctk.CTkLabel(progress_frame, textvariable=self.status_var)
        self.status_label.pack(anchor='w', padx=5)
        
        # Sample tab
        sample_tab.configure(fg_color=self.themes.get_theme(self.is_dark_mode)['frame'])
        
        # Left column for prompt settings
        sample_left = ctk.CTkFrame(sample_tab)
        sample_left.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Right column for template options
        sample_right = ctk.CTkFrame(sample_tab)
        sample_right.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Prompt Style Selection
        style_frame = ctk.CTkFrame(sample_left)
        style_frame.pack(fill='x', padx=5, pady=5)
        
        ctk.CTkLabel(style_frame, text="Prompt Style:").pack(anchor='w', padx=5, pady=2)
        
        # Radio buttons for prompt styles
        styles = ["Natural", "Danbooru", "Tags"]
        for style in styles:
            ctk.CTkRadioButton(style_frame, text=style, variable=self.prompt_style_var, 
                             value=style, command=self.update_prompt_style).pack(anchor='w', padx=20, pady=2)
        
        # Sampling Settings
        sampling_frame = ctk.CTkFrame(sample_left)
        sampling_frame.pack(fill='x', padx=5, pady=5)
        
        ctk.CTkLabel(sampling_frame, text="Sampling Settings").pack(anchor='w', padx=5, pady=2)
        
        # Enable sampling checkbox
        ctk.CTkCheckBox(sampling_frame, text="Enable Sampling", 
                       variable=self.enable_sampling_var).pack(anchor='w', padx=5, pady=2)
        
        # Sample every N steps
        sample_every_frame = ctk.CTkFrame(sampling_frame)
        sample_every_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(sample_every_frame, text="Sample Every:").pack(side='left')
        ctk.CTkEntry(sample_every_frame, textvariable=self.sample_every_var, 
                    width=80).pack(side='left', padx=5)
        ctk.CTkLabel(sample_every_frame, text="steps").pack(side='left')
        
        # Seed settings
        seed_frame = ctk.CTkFrame(sampling_frame)
        seed_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(seed_frame, text="Seed:").pack(side='left')
        ctk.CTkEntry(seed_frame, textvariable=self.seed_var, 
                    width=80).pack(side='left', padx=5)
        ctk.CTkCheckBox(seed_frame, text="Fixed Seed", 
                       variable=self.use_fixed_seed_var).pack(side='left', padx=5)
        
        # Sampling steps
        steps_frame = ctk.CTkFrame(sampling_frame)
        steps_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(steps_frame, text="Sampling Steps:").pack(side='left')
        ctk.CTkEntry(steps_frame, textvariable=self.sampling_steps_var, 
                    width=80).pack(side='left', padx=5)
        
        # CFG Scale
        cfg_frame = ctk.CTkFrame(sampling_frame)
        cfg_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(cfg_frame, text="CFG Scale:").pack(side='left')
        ctk.CTkEntry(cfg_frame, textvariable=self.cfg_scale_var, 
                    width=80).pack(side='left', padx=5)
        
        # Template Options (Right Column)
        template_frame = ctk.CTkFrame(sample_right)
        template_frame.pack(fill='x', padx=5, pady=5)
        
        ctk.CTkLabel(template_frame, text="Add Test Prompt:").pack(anchor='w', padx=5, pady=2)
        
        # Template buttons with descriptions
        templates = {
            "Person (Woman)": "Test the LoRA with a female character",
            "Person (Man)": "Test the LoRA with a male character",
            "Place": "Test the LoRA with a location or scene",
            "Thing": "Test the LoRA with an object or item"
        }
        
        for template_name, tooltip in templates.items():
            template_button = ctk.CTkButton(
                template_frame,
                text=template_name,
                command=lambda t=template_name: self.add_test_prompt(t),
                fg_color=self.themes.get_theme(self.is_dark_mode)['button'],
                hover_color=self.themes.get_theme(self.is_dark_mode)['highlight'],
                border_width=0,
                corner_radius=6
            )
            template_button.pack(anchor='w', padx=5, pady=2)
            CreateToolTip(template_button, tooltip)
        
        # Prompt Text Area
        prompt_frame = ctk.CTkFrame(sample_right)
        prompt_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Prompt header frame with label and delete button
        prompt_header = ctk.CTkFrame(prompt_frame)
        prompt_header.pack(fill='x', padx=5, pady=2)
        
        ctk.CTkLabel(prompt_header, text="Prompt:").pack(side='left', padx=5)
        
        # Delete prompt button
        ctk.CTkButton(
            prompt_header,
            text="Delete Prompt",
            command=lambda: self.prompt_text.delete('1.0', tk.END),
            fg_color=self.themes.get_theme(self.is_dark_mode)['button'],
            hover_color=self.themes.get_theme(self.is_dark_mode)['highlight'],
            border_width=0,
            corner_radius=6,
            width=100
        ).pack(side='right', padx=5)
        
        self.prompt_text = tk.Text(prompt_frame, height=10, wrap=tk.WORD)
        self.prompt_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Initialize default prompts with trigger word placeholder
        self.default_prompts = {
            "Natural": {
                "Person (Woman)": "A beautiful photograph of a <trigger> woman with long hair, standing in natural lighting, professional portrait",
                "Person (Man)": "A professional photograph of a <trigger> man in casual attire, confident pose, studio lighting",
                "Place": "A scenic view of a <trigger> location, beautiful lighting, high detail, professional photography",
                "Thing": "A detailed photograph of a <trigger> object, studio lighting, professional product photography"
            },
            "Danbooru": {
                "Person (Woman)": "1girl, <trigger>, solo, best quality, masterpiece, detailed, intricate details",
                "Person (Man)": "1boy, <trigger>, solo, best quality, masterpiece, detailed, intricate details",
                "Place": "<trigger>, landscape, scenery, no humans, best quality, masterpiece",
                "Thing": "<trigger>, object focus, no humans, best quality, masterpiece, detailed"
            },
            "Tags": {
                "Person (Woman)": "<trigger>, woman, portrait, (high quality), (best quality), (realistic), (photorealistic), (detailed)",
                "Person (Man)": "<trigger>, man, portrait, (high quality), (best quality), (realistic), (photorealistic), (detailed)",
                "Place": "<trigger>, location, (high quality), (best quality), (realistic), (photorealistic), (detailed)",
                "Thing": "<trigger>, object, (high quality), (best quality), (realistic), (photorealistic), (detailed)"
            }
        }

        # Add Batch tab
        batch_tab.configure(fg_color=self.themes.get_theme(self.is_dark_mode)['frame'])
        
        # Left column for batch list
        batch_left = ctk.CTkFrame(batch_tab)
        batch_left.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Right column for controls
        self.batch_right = ctk.CTkFrame(batch_tab)  # Store as instance variable
        self.batch_right.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Batch List
        list_frame = ctk.CTkFrame(batch_left)
        list_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(list_frame, text="Batch Queue:").pack(anchor='w', padx=5, pady=2)
        
        # Create listbox for batch items
        self.batch_listbox = tk.Listbox(list_frame, selectmode=tk.SINGLE, 
                                       bg=self.themes.get_theme(self.is_dark_mode)['button'],
                                       fg=self.themes.get_theme(self.is_dark_mode)['text'])
        self.batch_listbox.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Batch Controls
        control_frame = ctk.CTkFrame(self.batch_right)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        # Add current config to batch
        ctk.CTkButton(control_frame, 
                      text="Add Current Config", 
                      command=self.add_to_batch,
                      **button_style).pack(fill='x', padx=5, pady=2)
        
        # Load config to batch
        ctk.CTkButton(control_frame, 
                      text="Load Config to Batch", 
                      command=self.load_config_to_batch,
                      **button_style).pack(fill='x', padx=5, pady=2)
        
        # Reorder buttons frame
        reorder_frame = ctk.CTkFrame(control_frame)
        reorder_frame.pack(fill='x', padx=5, pady=2)
        
        # Move Up button
        ctk.CTkButton(reorder_frame, 
                      text="▲ Move Up",
                      command=self.move_batch_item_up,
                      width=100,
                      **button_style).pack(side='left', expand=True, padx=2)
        
        # Move Down button
        ctk.CTkButton(reorder_frame, 
                      text="▼ Move Down",
                      command=self.move_batch_item_down,
                      width=100,
                      **button_style).pack(side='left', expand=True, padx=2)
        
        # Remove selected from batch
        ctk.CTkButton(control_frame, 
                      text="Remove Selected", 
                      command=self.remove_from_batch,
                      **button_style).pack(fill='x', padx=5, pady=2)
        
        # Clear batch
        ctk.CTkButton(control_frame, 
                      text="Clear Batch", 
                      command=self.clear_batch,
                      **button_style).pack(fill='x', padx=5, pady=2)
        
        # Start batch processing
        ctk.CTkButton(control_frame, 
                      text="Start Batch Processing", 
                      command=self.start_batch_processing,
                      **button_style).pack(fill='x', padx=5, pady=2)
        
        # Batch status
        status_frame = ctk.CTkFrame(self.batch_right)
        status_frame.pack(fill='x', padx=5, pady=5)
        ctk.CTkLabel(status_frame, textvariable=self.batch_status_var).pack(anchor='w', padx=5)

        # Add Caption Management tab
        caption_tab.configure(fg_color=self.themes.get_theme(self.is_dark_mode)['frame'])
        
        # Left column for folder selection and options
        caption_left = ctk.CTkFrame(caption_tab)
        caption_left.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Right column for template and preview
        caption_right = ctk.CTkFrame(caption_tab)
        caption_right.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Folder Selection
        folder_frame = ctk.CTkFrame(caption_left)
        folder_frame.pack(fill='x', padx=5, pady=5)
        
        ctk.CTkLabel(folder_frame, text="Images Folder:").pack(anchor='w', padx=5, pady=2)
        folder_select = ctk.CTkFrame(folder_frame)
        folder_select.pack(fill='x', padx=5, pady=2)
        ctk.CTkEntry(folder_select, textvariable=self.caption_folder_var).pack(side='left', fill='x', expand=True, padx=5)
        ctk.CTkButton(folder_select, text="Browse", command=self.browse_caption_folder, **button_style).pack(side='right')
        
        # Caption Options
        options_frame = ctk.CTkFrame(caption_left)
        options_frame.pack(fill='x', padx=5, pady=5)
        
        ctk.CTkLabel(options_frame, text="Caption Options:").pack(anchor='w', padx=5, pady=2)
        
        # Prefix
        prefix_frame = ctk.CTkFrame(options_frame)
        prefix_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(prefix_frame, text="Prefix:").pack(side='left')
        ctk.CTkEntry(prefix_frame, textvariable=self.caption_prefix_var).pack(side='left', fill='x', expand=True, padx=5)
        
        # Suffix
        suffix_frame = ctk.CTkFrame(options_frame)
        suffix_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(suffix_frame, text="Suffix:").pack(side='left')
        ctk.CTkEntry(suffix_frame, textvariable=self.caption_suffix_var).pack(side='left', fill='x', expand=True, padx=5)
        
        # Overwrite option
        ctk.CTkCheckBox(options_frame, text="Overwrite existing captions", 
                        variable=self.overwrite_captions_var).pack(anchor='w', padx=5, pady=2)
        
        # Template Frame
        template_frame = ctk.CTkFrame(caption_right)
        template_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(template_frame, text="Caption Template:").pack(anchor='w', padx=5, pady=2)
        
        # Template text area
        self.caption_template = tk.Text(template_frame, height=10, width=40,
                                      bg=self.themes.get_theme(self.is_dark_mode)['button'],
                                      fg=self.themes.get_theme(self.is_dark_mode)['text'])
        self.caption_template.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Template buttons
        template_buttons = ctk.CTkFrame(template_frame)
        template_buttons.pack(fill='x', padx=5, pady=5)
        
        ctk.CTkButton(template_buttons, text="Load Template", 
                      command=self.load_caption_template, **button_style).pack(side='left', padx=5)
        ctk.CTkButton(template_buttons, text="Save Template", 
                      command=self.save_caption_template, **button_style).pack(side='left', padx=5)
        ctk.CTkButton(template_buttons, text="Generate Captions", 
                      command=self.generate_captions, **button_style).pack(side='right', padx=5)

        # Add LLM Vision section to caption_left
        llm_frame = ctk.CTkFrame(caption_left)
        llm_frame.pack(fill='x', padx=5, pady=5)
        
        # LLM Vision toggle
        llm_header = ctk.CTkFrame(llm_frame)
        llm_header.pack(fill='x', padx=5, pady=2)
        ctk.CTkCheckBox(llm_header, text="Use LLM Vision for Auto-Captioning", 
                        variable=self.use_llm_vision_var,
                        command=self.toggle_llm_options).pack(side='left', padx=5)
        
        # LLM Options container
        self.llm_options = ctk.CTkFrame(llm_frame)
        self.llm_options.pack(fill='x', padx=5, pady=2)
        
        # Model selection
        model_frame = ctk.CTkFrame(self.llm_options)
        model_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(model_frame, text="Model:").pack(side='left')
        ctk.CTkOptionMenu(model_frame, 
                         variable=self.llm_model_var,
                         values=["gpt-4-vision-preview", "claude-3-opus-20240229"],
                         **option_menu_style).pack(side='left', padx=5)
        
        # API Key
        api_frame = ctk.CTkFrame(self.llm_options)
        api_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(api_frame, text="API Key:").pack(side='left')
        api_entry = ctk.CTkEntry(api_frame, textvariable=self.llm_api_key_var, show="*")
        api_entry.pack(side='left', fill='x', expand=True, padx=5)
        
        # System Prompt
        prompt_frame = ctk.CTkFrame(self.llm_options)
        prompt_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(prompt_frame, text="System Prompt:").pack(anchor='w', padx=5)
        ctk.CTkEntry(prompt_frame, textvariable=self.llm_prompt_var).pack(fill='x', padx=5, pady=2)
        
        # Initially disable LLM options
        self.toggle_llm_options()

        # Add Local LLM section
        local_llm_frame = ctk.CTkFrame(self.llm_options)
        local_llm_frame.pack(fill='x', padx=5, pady=2)
        
        # Local LLM toggle
        ctk.CTkCheckBox(local_llm_frame, text="Use Local LLM", 
                        variable=self.use_local_llm_var,
                        command=self.toggle_llm_type).pack(anchor='w', padx=5)
        
        # Local API settings container
        self.local_api_frame = ctk.CTkFrame(local_llm_frame)
        self.local_api_frame.pack(fill='x', padx=5, pady=2)
        
        # API URL
        url_frame = ctk.CTkFrame(self.local_api_frame)
        url_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(url_frame, text="API URL:").pack(side='left')
        ctk.CTkEntry(url_frame, textvariable=self.local_api_url_var).pack(side='left', fill='x', expand=True, padx=5)
        
        # Temperature
        temp_frame = ctk.CTkFrame(self.local_api_frame)
        temp_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(temp_frame, text="Temperature:").pack(side='left')
        ctk.CTkEntry(temp_frame, textvariable=self.local_model_temp_var, width=80).pack(side='left', padx=5)
        
        # Initially update states
        self.toggle_llm_type()

    def toggle_llm_options(self):
        """Enable/disable LLM options based on checkbox"""
        state = "normal" if self.use_llm_vision_var.get() else "disabled"
        for widget in self.llm_options.winfo_children():
            for child in widget.winfo_children():
                if isinstance(child, (ctk.CTkEntry, ctk.CTkOptionMenu)):
                    child.configure(state=state)

    def add_test_prompt(self, template_type):
        """Add a test prompt with the current trigger word"""
        style = self.prompt_style_var.get()
        trigger_word = self.trigger_word_var.get() or "<trigger>"
        
        # Get the template and replace the trigger placeholder
        template = self.default_prompts[style][template_type]
        prompt = template.replace("<trigger>", trigger_word)
        
        # Add newline if there's already text
        if self.prompt_text.get('1.0', tk.END).strip():
            self.prompt_text.insert(tk.END, '\n\n')
        self.prompt_text.insert(tk.END, prompt)

    def update_prompt_style(self):
        """Update the prompt text based on selected style"""
        style = self.prompt_style_var.get()
        trigger_word = self.trigger_word_var.get() or "<trigger>"
        
        # Clear existing text
        self.prompt_text.delete('1.0', tk.END)
        
        # Add example prompt for the selected style
        example = self.default_prompts[style]["Person (Woman)"].replace("<trigger>", trigger_word)
        self.prompt_text.insert('1.0', example)

    def load_default_config(self):
        """Load master template configuration from train_lora_flux_24gb.yaml"""
        try:
            default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                             'ai-toolkit', 'config', 'examples', 
                                             'train_lora_flux_24gb.yaml')
            if os.path.exists(default_config_path):
                with open(default_config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                self.logger.error(f"Master template not found at: {default_config_path}")
                raise FileNotFoundError(f"Required master template not found: {default_config_path}")
        except Exception as e:
            self.logger.error(f"Error loading master template: {str(e)}")
            raise

    def prepare_config(self):
        """Prepare the configuration dictionary using master template as base"""
        # Load master template configuration
        config = self.load_default_config()
        process_config = config['config']['process'][0]
        
        # Only update values that are explicitly set in the GUI
        # Everything else will keep the master template values
        
        # Update basic config if set
        if self.name_var.get():
            config['config']['name'] = self.name_var.get()
        if self.trigger_word_var.get():
            config['config']['trigger_word'] = self.trigger_word_var.get()
        
        # Update training folder if set
        if self.output_folder_var.get():
            process_config['training_folder'] = self.output_folder_var.get()
        
        # Update network settings that are configurable in GUI
        process_config['network'].update({
            'type': self.network_type_var.get(),
            'linear': int(self.linear_var.get()),
            'linear_alpha': int(self.linear_alpha_var.get())
        })
        
        # Update save settings that are configurable in GUI
        process_config['save'].update({
            'dtype': self.save_dtype_var.get(),
            'save_every': int(self.save_every_var.get()),
            'max_step_saves_to_keep': int(self.max_saves_var.get())
        })
        
        # Update dataset settings if path is set
        if self.data_folder_var.get():
            # Get selected resolutions
            selected_resolutions = []
            for res, data in self.resolution_vars.items():
                if data['var'].get():
                    size = int(res.split('x')[0])
                    selected_resolutions.append(size)
            
            if not selected_resolutions:
                selected_resolutions = [512]
            
            process_config['datasets'][0].update({
                'folder_path': self.data_folder_var.get(),
                'resolution': selected_resolutions
            })
        
        # Update train settings that are configurable in GUI
        selected_lr = self.lr_var.get()
        lr_value = self.lr_presets[selected_lr]['value'] if selected_lr in self.lr_presets else selected_lr
        
        process_config['train'].update({
            'batch_size': int(self.batch_size_var.get()),
            'steps': int(self.steps_var.get()),
            'gradient_accumulation_steps': int(self.grad_accum_var.get()),
            'train_unet': self.train_unet_var.get(),
            'train_text_encoder': self.train_text_encoder_var.get(),
            'gradient_checkpointing': self.grad_checkpointing_var.get(),
            'noise_scheduler': self.noise_scheduler_var.get(),
            'optimizer': self.optimizer_var.get(),
            'lr': lr_value
        })
        
        # Update model settings if path is set
        if self.model_path_var.get():
            process_config['model'].update({
                'name_or_path': self.model_path_var.get(),
                'is_flux': self.is_flux_var.get(),
                'quantize': self.quantize_var.get()
            })
        
        # Update sample settings if enabled
        if self.enable_sampling_var.get():
            prompts = [p.strip() for p in self.prompt_text.get('1.0', tk.END).split('\n\n')]
            prompts = [p for p in prompts if p]
            
            # Only update sampling values that are configurable in GUI
            process_config['sample'].update({
                'sample_every': int(self.sample_every_var.get()),
                'prompts': prompts,
                'seed': int(self.seed_var.get()),
                'walk_seed': self.use_fixed_seed_var.get(),
                'guidance_scale': float(self.cfg_scale_var.get()),
                'sample_steps': int(self.sampling_steps_var.get())
            })
        
        return config

    def save_config(self):
        """Save the current configuration to a YAML file"""
        try:
            config = self.prepare_config()
            
            output_folder = self.output_folder_var.get()
            name = self.name_var.get()
            
            if not output_folder:
                raise ValueError("Please set an output folder first.")
                
            if not name:
                raise ValueError("Please set a name for the configuration.")
                
            os.makedirs(output_folder, exist_ok=True)
            
            default_path = os.path.join(output_folder, f"{name}_config.yaml")
            
            file_path = filedialog.asksaveasfilename(
                initialdir=output_folder,
                initialfile=f"{name}_config.yaml",
                defaultextension=".yaml",
                filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
            )
            
            if not file_path:
                file_path = default_path
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            self.current_config_path = file_path
            self.current_config = config
            self.current_config_label_var.set(f"Loaded: {os.path.basename(file_path)}")
            self.save_current_train()  # Save as current training config
            
            messagebox.showinfo("Success", "Configuration saved successfully!")
            
        except Exception as e:
            error_msg = f"Failed to save configuration: {str(e)}"
            self.logger.error(error_msg)
            messagebox.showerror("Error", error_msg)

    def convert_captions_to_utf8(self, folder_path):
        """Convert all caption files in the folder to UTF-8"""
        caption_files = []
        for file in os.listdir(folder_path):
            if file.lower().endswith('.txt'):
                caption_files.append(os.path.join(folder_path, file))
        
        converted_count = 0
        failed_count = 0
        
        for file_path in caption_files:
            try:
                # First try to detect the encoding
                with open(file_path, 'rb') as file:
                    raw = file.read()
                    result = chardet.detect(raw)
                    detected_encoding = result['encoding'] if result['confidence'] > 0.7 else 'latin-1'
                
                # Try to read with detected encoding
                try:
                    with open(file_path, 'r', encoding=detected_encoding) as file:
                        content = file.read()
                except UnicodeDecodeError:
                    # If that fails, try common encodings
                    for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                        try:
                            with open(file_path, 'r', encoding=encoding) as file:
                                content = file.read()
                                detected_encoding = encoding
                                break
                        except UnicodeDecodeError:
                            continue
                
                # Write back as UTF-8 if we successfully read the file
                if content:
                    with open(file_path, 'w', encoding='utf-8', errors='ignore') as file:
                        file.write(content)
                    converted_count += 1
                    self.logger.info(f"Converted {file_path} from {detected_encoding} to UTF-8")
                else:
                    failed_count += 1
                    self.logger.error(f"Failed to read {file_path} with any encoding")
                
            except Exception as e:
                failed_count += 1
                self.logger.error(f"Failed to convert {file_path}: {str(e)}")
        
        if failed_count > 0:
            self.logger.warning(f"Failed to convert {failed_count} files")
        
        return converted_count

    def start_training(self):
        """Start training with current configuration"""
        try:
            # Check if UTF-8 conversion is needed
            if self.convert_utf8_var.get():
                data_folder = self.data_folder_var.get()
                if data_folder and os.path.exists(data_folder):
                    self.status_var.set("Converting captions to UTF-8...")
                    converted = self.convert_captions_to_utf8(data_folder)
                    if converted > 0:
                        self.logger.info(f"Converted {converted} caption files to UTF-8")
                    self.status_var.set("UTF-8 conversion completed")
            
            # Save current UI state to current_train.yaml
            config = self.prepare_config()
            
            with open(self.current_train_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            # Load paths from config.yaml
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
            
            if not os.path.exists(config_path):
                raise FileNotFoundError("config.yaml not found. Please run start.bat first.")
            
            with open(config_path, 'r') as f:
                paths_config = yaml.safe_load(f)
            
            # Validate required fields
            if not self.model_path_var.get():
                raise ValueError("Please select a model path")
            if not self.data_folder_var.get():
                raise ValueError("Please select a data folder")
            if not self.output_folder_var.get():
                raise ValueError("Please select an output folder")
            if not self.name_var.get():
                raise ValueError("Please enter a name for the training")
            
            # Start training process with default console output
            self.training_process = subprocess.Popen([
                paths_config['python_path'],
                paths_config['train_script_path'],
                self.current_train_file
            ])
            
            # Update button states
            self.train_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            
            self.status_var.set("Training started... Check console for progress")
            
        except Exception as e:
            error_msg = "Failed to start training: " + str(e)
            self.logger.error(error_msg)
            messagebox.showerror("Error", error_msg)

    def stop_training(self):
        """Stop the training process"""
        try:
            if self.training_process:
                # Try to terminate gracefully first
                self.training_process.terminate()
                
                # Give it a moment to terminate
                try:
                    self.training_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # If it doesn't terminate, force kill
                    self.training_process.kill()
                
                self.training_process = None
                
                # Update button states
                self.train_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                
                self.status_var.set("Training stopped by user")
                self.logger.info("Training process stopped by user")
                
        except Exception as e:
            error_msg = f"Failed to stop training: {str(e)}"
            self.logger.error(error_msg)
            messagebox.showerror("Error", error_msg)

    def update_progress(self, current_step, total_steps, status="Training"):
        """Update the progress display"""
        progress = (current_step / total_steps) * 100 if total_steps > 0 else 0
        self.progress_bar.set(progress)
        self.step_var.set(f"Step: {current_step}/{total_steps}")
        self.status_var.set(status)
        self.master.update_idletasks()

    def save_last_settings(self):
        """Save current settings to JSON file"""
        settings = {
            'name': self.name_var.get(),
            'trigger_word': self.trigger_word_var.get(),
            'data_folder': self.data_folder_var.get(),
            'output_folder': self.output_folder_var.get(),
            'model_path': self.model_path_var.get(),
            'is_flux': self.is_flux_var.get(),
            'quantize': self.quantize_var.get(),
            'convert_utf8': self.convert_utf8_var.get(),
            'learning_rate': self.lr_var.get(),
            'batch_size': self.batch_size_var.get(),
            'steps': self.steps_var.get(),
            'auto_steps': self.auto_steps_var.get(),
            'grad_accum': self.grad_accum_var.get(),
            'train_unet': self.train_unet_var.get(),
            'train_text_encoder': self.train_text_encoder_var.get(),
            'grad_checkpointing': self.grad_checkpointing_var.get(),
            'noise_scheduler': self.noise_scheduler_var.get(),
            'optimizer': self.optimizer_var.get(),
            'save_dtype': self.save_dtype_var.get(),
            'save_every': self.save_every_var.get(),
            'max_saves': self.max_saves_var.get(),
            'network_type': self.network_type_var.get(),
            'linear': self.linear_var.get(),
            'linear_alpha': self.linear_alpha_var.get(),
            # Sampling settings
            'enable_sampling': self.enable_sampling_var.get(),
            'sample_every': self.sample_every_var.get(),
            'sampling_steps': self.sampling_steps_var.get(),
            'cfg_scale': self.cfg_scale_var.get(),
            'seed': self.seed_var.get(),
            'use_fixed_seed': self.use_fixed_seed_var.get(),
            'prompt_style': self.prompt_style_var.get(),
            'prompt_text': self.prompt_text.get('1.0', tk.END).strip(),
            'dark_mode': self.is_dark_mode,
            # Ensure these critical paths are saved
            'last_model_path': self.model_path_var.get(),
            'last_output_folder': self.output_folder_var.get(),
            # Save resolution settings
            'resolutions': {res: data['var'].get() for res, data in self.resolution_vars.items()},
        }
        
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"Failed to save settings: {str(e)}")

    def load_last_settings(self):
        """Load last used settings from JSON file"""
        if not os.path.exists(self.settings_file):
            return
            
        try:
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
                
            # Load basic settings
            self.name_var.set(settings.get('name', ''))
            self.trigger_word_var.set(settings.get('trigger_word', ''))
            self.data_folder_var.set(settings.get('data_folder', ''))
            
            # Load critical paths with priority from last saved paths
            self.model_path_var.set(settings.get('last_model_path', settings.get('model_path', '')))
            self.output_folder_var.set(settings.get('last_output_folder', settings.get('output_folder', '')))
            
            self.is_flux_var.set(settings.get('is_flux', True))
            self.quantize_var.set(settings.get('quantize', True))
            self.convert_utf8_var.set(settings.get('convert_utf8', True))
            self.lr_var.set(settings.get('learning_rate', '1e-4 (Standard)'))
            self.batch_size_var.set(settings.get('batch_size', '1'))
            self.steps_var.set(settings.get('steps', '5000'))
            self.auto_steps_var.set(settings.get('auto_steps', True))
            self.grad_accum_var.set(settings.get('grad_accum', '1'))
            self.train_unet_var.set(settings.get('train_unet', True))
            self.train_text_encoder_var.set(settings.get('train_text_encoder', False))
            self.grad_checkpointing_var.set(settings.get('grad_checkpointing', True))
            self.noise_scheduler_var.set(settings.get('noise_scheduler', 'flowmatch'))
            self.optimizer_var.set(settings.get('optimizer', 'adamw8bit'))
            self.save_dtype_var.set(settings.get('save_dtype', 'float16'))
            self.save_every_var.set(settings.get('save_every', '250'))
            self.max_saves_var.set(settings.get('max_saves', '4'))
            self.network_type_var.set(settings.get('network_type', 'lora'))
            self.linear_var.set(settings.get('linear', '16'))
            self.linear_alpha_var.set(settings.get('linear_alpha', '16'))
            
            # Load sampling settings
            self.enable_sampling_var.set(settings.get('enable_sampling', True))
            self.sample_every_var.set(settings.get('sample_every', '500'))
            self.sampling_steps_var.set(settings.get('sampling_steps', '20'))
            self.cfg_scale_var.set(settings.get('cfg_scale', '3.5'))
            self.seed_var.set(settings.get('seed', '42'))
            self.use_fixed_seed_var.set(settings.get('use_fixed_seed', True))
            self.prompt_style_var.set(settings.get('prompt_style', 'Natural'))
            
            # Load prompt text if it exists
            if 'prompt_text' in settings:
                self.prompt_text.delete('1.0', tk.END)
                self.prompt_text.insert('1.0', settings['prompt_text'])
                        
            # Update steps entry state
            if hasattr(self, 'steps_entry'):
                self.steps_entry.configure(state='disabled' if self.auto_steps_var.get() else 'normal')
            
            # Update dataset stats if data folder exists
            self.update_dataset_stats(settings.get('data_folder', ''))
            
            # Load theme preference
            self.is_dark_mode = settings.get('dark_mode', True)
            
            # Load resolution settings
            if 'resolutions' in settings:
                for res, value in settings['resolutions'].items():
                    if res in self.resolution_vars:
                        self.resolution_vars[res]['var'].set(value)
                
        except Exception as e:
            print(f"Failed to load settings: {str(e)}")

    def on_closing(self):
        """Handle window closing"""
        try:
            # Stop training if it's running
            if self.training_process:
                self.stop_training()
            
            # Save settings
            self.save_last_settings()
            
            # Destroy window
            if isinstance(self.master, tk.Tk):
                self.master.destroy()
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")

    def load_config_file(self):
        """Load configuration file to prefill UI fields"""
        file_path = filedialog.askopenfilename(
            title="Load Configuration Template",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = yaml.safe_load(f)
                # Load config into UI fields only
                self.load_config(config)
                messagebox.showinfo("Success", "Configuration template loaded successfully!")
            except Exception as e:
                error_msg = f"Failed to load configuration template: {str(e)}"
                self.logger.error(error_msg)
                messagebox.showerror("Error", error_msg)

    def update_config(self, key, value):
        """Update the current configuration with new values"""
        self.current_config[key] = value

    def update_dataset_stats(self, folder_path):
        """Update dataset statistics for a given folder path"""
        if folder_path and os.path.exists(folder_path):
            image_count, caption_count = self.count_dataset_files(folder_path)
            self.dataset_stats_var.set(f"Images: {image_count}, Captions: {caption_count}")
            if self.auto_steps_var.get():
                suggested_steps = self.calculate_suggested_steps(image_count)
                self.steps_var.set(str(suggested_steps))
        else:
            self.dataset_stats_var.set("No valid folder selected")
            
    def browse_folder(self, folder_type):
        """Browse for a folder based on type"""
        folder = filedialog.askdirectory()
        if folder:
            if folder_type == 'model':
                self.model_path_var.set(folder)
            elif folder_type == 'data':
                self.data_folder_var.set(folder)
                # Explicitly update dataset stats when data folder is selected
                self.update_dataset_stats(folder)
            elif folder_type == 'output':
                self.output_folder_var.set(folder)

    def count_dataset_files(self, folder):
        """Count the number of images and caption files in the dataset folder"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
        caption_extensions = {'.txt', '.caption'}
        
        image_count = 0
        caption_count = 0
        
        try:
            for file in os.listdir(folder):
                lower_file = file.lower()
                ext = os.path.splitext(lower_file)[1]
                if ext in image_extensions:
                    image_count += 1
                elif ext in caption_extensions:
                    caption_count += 1
            
            return image_count, caption_count
        except Exception as e:
            messagebox.showerror("Error", f"Failed to count files: {str(e)}")
            return 0, 0

    def load_config(self, config):
        """Load configuration values into GUI fields"""
        try:
            # Extract the first process config (assuming single process)
            process_config = config.get('config', {}).get('process', [{}])[0]
            
            # Basic settings
            self.name_var.set(config.get('config', {}).get('name', ''))
            self.trigger_word_var.set(config.get('config', {}).get('trigger_word', ''))
            
            # Model settings
            model_config = process_config.get('model', {})
            self.model_path_var.set(model_config.get('name_or_path', ''))
            self.is_flux_var.set(model_config.get('is_flux', True))
            self.quantize_var.set(model_config.get('quantize', True))
            
            # Network settings
            network_config = process_config.get('network', {})
            self.network_type_var.set(network_config.get('type', 'lora'))
            self.linear_var.set(str(network_config.get('linear', 16)))
            self.linear_alpha_var.set(str(network_config.get('linear_alpha', 16)))
            
            # Dataset settings
            dataset_config = process_config.get('datasets', [{}])[0]
            self.data_folder_var.set(dataset_config.get('folder_path', ''))
            
            # Training settings
            train_config = process_config.get('train', {})
            self.batch_size_var.set(str(train_config.get('batch_size', 1)))
            self.steps_var.set(str(train_config.get('steps', 5000)))
            self.grad_accum_var.set(str(train_config.get('gradient_accumulation_steps', 1)))
            self.train_unet_var.set(train_config.get('train_unet', True))
            self.train_text_encoder_var.set(train_config.get('train_text_encoder', False))
            self.grad_checkpointing_var.set(train_config.get('gradient_checkpointing', True))
            self.noise_scheduler_var.set(train_config.get('noise_scheduler', 'flowmatch'))
            self.optimizer_var.set(train_config.get('optimizer', 'adamw8bit'))
            
            # Learning rate
            lr_value = str(train_config.get('lr', '1e-4'))
            # Find matching preset or use custom value
            preset_found = False
            for preset_name, preset_data in self.lr_presets.items():
                if preset_data['value'] == lr_value:
                    self.lr_var.set(preset_name)
                    preset_found = True
                    break
            if not preset_found:
                self.lr_var.set(lr_value)
            
            # Save settings
            save_config = process_config.get('save', {})
            self.save_dtype_var.set(save_config.get('dtype', 'float16'))
            self.save_every_var.set(str(save_config.get('save_every', 250)))
            self.max_saves_var.set(str(save_config.get('max_step_saves_to_keep', 4)))
            
            # Output folder
            self.output_folder_var.set(process_config.get('training_folder', ''))
            
            # Sampling settings
            sample_config = process_config.get('sample', {})
            if sample_config:
                self.enable_sampling_var.set(True)
                self.sample_every_var.set(str(sample_config.get('sample_every', 500)))
                self.sampling_steps_var.set(str(sample_config.get('sample_steps', 20)))
                self.cfg_scale_var.set(str(sample_config.get('guidance_scale', 3.5)))
                self.seed_var.set(str(sample_config.get('seed', 42)))
                self.use_fixed_seed_var.set(sample_config.get('walk_seed', True))
                
                # Load prompts if they exist
                if 'prompts' in sample_config:
                    self.prompt_text.delete('1.0', tk.END)
                    self.prompt_text.insert('1.0', '\n\n'.join(sample_config['prompts']))
            else:
                self.enable_sampling_var.set(False)
            
            # Update dataset stats
            self.update_dataset_stats(self.data_folder_var.get())
            
            # After loading, save as current training config
            self.current_config = config
            self.save_current_train()
            
        except Exception as e:
            error_msg = f"Failed to load configuration: {str(e)}"
            self.logger.error(error_msg)
            messagebox.showerror("Error", error_msg)

    def add_theme_toggle(self):
        """Add theme toggle button to the GUI"""
        self.theme_button = ctk.CTkButton(
            self.master,
            text="🌙" if self.is_dark_mode else "☀️",
            command=self.toggle_theme,
            width=30,
            height=30
        )
        self.theme_button.pack(anchor='ne', padx=5, pady=5)

    def toggle_theme(self):
        """Toggle between light and dark themes"""
        self.is_dark_mode = not self.is_dark_mode
        
        # Get new theme colors
        theme = self.themes.get_theme(self.is_dark_mode)
        
        # Update root window and title bar
        if isinstance(self.master, tk.Tk):
            self.master.configure(bg=theme['bg'])
            # Update title bar
            title_bar = next((w for w in self.master.winfo_children() 
                             if isinstance(w, ctk.CTkFrame) and w.winfo_height() == 30), None)
            if title_bar:
                title_bar.configure(fg_color=theme['frame'])
                # Update title bar children (close button and title)
                for child in title_bar.winfo_children():
                    if isinstance(child, ctk.CTkButton):  # Close button
                        child.configure(
                            fg_color="transparent",
                            hover_color="#FF4444",
                            text_color=theme['text']
                        )
                    elif isinstance(child, ctk.CTkLabel):  # Title label
                        child.configure(text_color=theme['text'])
        
        # Update main frame color
        self.configure(fg_color=theme['bg'])
        
        # Update notebook and its tabs
        self.main_notebook.configure(fg_color=theme['frame'])
        self.main_notebook._segmented_button.configure(
            fg_color=theme['frame'],
            selected_color=theme['highlight'],
            unselected_color=theme['button']
        )
        
        # Update all tab frames
        for tab in ["Training", "Sample"]:
            tab_frame = self.main_notebook.tab(tab)
            if tab_frame:
                tab_frame.configure(fg_color=theme['frame'])
                # Update all child frames in the tab
                for child in tab_frame.winfo_children():
                    if isinstance(child, (ctk.CTkFrame, tk.Frame)):
                        child.configure(fg_color=theme['frame'])
                        # Update nested frames
                        for nested in child.winfo_children():
                            if isinstance(nested, (ctk.CTkFrame, tk.Frame)):
                                nested.configure(fg_color=theme['frame'])
        
        # Update text widget colors
        if hasattr(self, 'prompt_text'):
            self.prompt_text.configure(
                bg=theme['bg'],
                fg=theme['text'],
                insertbackground=theme['text']
            )
        
        # Update progress frame and bar
        for widget in self.winfo_children():
            if isinstance(widget, ctk.CTkFrame):
                widget.configure(fg_color=theme['frame'])
                # Update progress bar if it exists
                for child in widget.winfo_children():
                    if isinstance(child, ctk.CTkProgressBar):
                        child.configure(
                            fg_color=theme['button'],
                            progress_color=theme['highlight']
                        )
        
        # Update button styles
        button_style = {
            'fg_color': theme['button'],
            'hover_color': theme['highlight'],
            'text_color': theme['text'],
            'border_color': theme['highlight']
        }
        
        option_menu_style = {
            'fg_color': theme['button'],
            'button_color': theme['button'],
            'button_hover_color': theme['highlight'],
            'dropdown_fg_color': theme['frame'],
            'dropdown_hover_color': theme['highlight'],
            'text_color': theme['text']
        }
        
        # Update all widgets recursively with proper styling
        def update_widget_styles(widget):
            if isinstance(widget, ctk.CTkButton):
                widget.configure(**button_style)
            elif isinstance(widget, ctk.CTkOptionMenu):
                widget.configure(**option_menu_style)
            elif isinstance(widget, ctk.CTkFrame):
                widget.configure(fg_color=theme['frame'])
            elif isinstance(widget, ctk.CTkEntry):
                widget.configure(
                    fg_color=theme['button'],
                    text_color=theme['text'],
                    border_color=theme['highlight']
                )
            elif isinstance(widget, ctk.CTkCheckBox):
                widget.configure(
                    fg_color=theme['button'],
                    text_color=theme['text'],
                    hover_color=theme['highlight']
                )
            elif isinstance(widget, ctk.CTkRadioButton):
                widget.configure(
                    fg_color=theme['button'],
                    text_color=theme['text'],
                    hover_color=theme['highlight']
                )
            elif isinstance(widget, ctk.CTkLabel):
                widget.configure(text_color=theme['text'])
            elif isinstance(widget, tk.Text):
                widget.configure(
                    bg=theme['bg'],
                    fg=theme['text'],
                    insertbackground=theme['text']
                )
        
            # Recursively update child widgets
            for child in widget.winfo_children():
                update_widget_styles(child)
        
        # Apply styles to all widgets
        update_widget_styles(self)
        
        # Update theme button text
        self.theme_button.configure(text="🌙" if self.is_dark_mode else "☀️")
        
        # Save theme preference
        self.save_theme_preference()

    def load_theme_preference(self):
        """Load theme preference from settings file"""
        try:
            if os.path.exists('last_settings.json'):
                with open('last_settings.json', 'r') as f:
                    settings = json.load(f)
                    return settings.get('dark_mode', True)
        except:
            pass
        return True  # Default to dark mode

    def save_theme_preference(self):
        """Save theme preference to settings file"""
        try:
            settings = {}
            if os.path.exists('last_settings.json'):
                with open('last_settings.json', 'r') as f:
                    settings = json.load(f)
            
            settings['dark_mode'] = self.is_dark_mode
            
            with open('last_settings.json', 'w') as f:
                json.dump(settings, f, indent=4)
        except:
            pass

    def toggle_steps_entry(self):
        """Toggle the steps entry between auto and manual"""
        if self.auto_steps_var.get():
            # If switching to auto, update with suggested steps
            if self.data_folder_var.get():
                image_count, _ = self.count_dataset_files(self.data_folder_var.get())
                self.update_suggested_steps(image_count)
        
        # Enable/disable steps entry based on auto_steps_var
        if hasattr(self, 'steps_entry'):
            self.steps_entry.configure(state='disabled' if self.auto_steps_var.get() else 'normal')

    def calculate_suggested_steps(self, image_count):
        """Calculate suggested training steps based on image count and learning rate"""
        if image_count <= 0:
            return 5000  # Default value
        
        # Get current learning rate value and convert to float
        selected_lr = self.lr_var.get()
        try:
            lr_value = float(self.lr_presets[selected_lr]['value'] if selected_lr in self.lr_presets else selected_lr)
        except (ValueError, TypeError):
            # If conversion fails, use default value
            self.logger.warning(f"Invalid learning rate value: {selected_lr}, using default")
            lr_value = 1e-4
        
        # Define recommended repeats based on learning rate
        if lr_value >= 2e-4:  # Fast learning
            repeats = 100  # Less repeats needed for fast learning
        elif lr_value >= 1e-4:  # Standard
            repeats = 150
        elif lr_value >= 5e-5:  # Conservative
            repeats = 200
        elif lr_value >= 3e-5:  # Very Conservative
            repeats = 250
        else:  # Minimal
            repeats = 300
        
        # Calculate total steps based on images and batch size
        try:
            batch_size = int(self.batch_size_var.get())
        except ValueError:
            batch_size = 1  # Default if invalid
        
        steps_per_epoch = (image_count + batch_size - 1) // batch_size
        total_steps = steps_per_epoch * repeats
        
        # Round to nearest 100
        total_steps = round(total_steps / 100) * 100
        
        # Set minimum and maximum values
        total_steps = max(1000, min(total_steps, 20000))
        
        return total_steps

    def update_suggested_steps(self, image_count):
        """Update steps based on auto calculation"""
        if self.auto_steps_var.get():
            suggested = self.calculate_suggested_steps(image_count)
            self.steps_var.set(str(suggested))

    def get_template_text(self, template_type):
        """Get the template text based on the template type"""
        templates = {
            "Character": """Character Description:
Gender: [male/female]
Character Type: [human/fantasy/sci-fi]
Age Range: [child/young/adult/elderly]

Physical Appearance:
- Hair: [color, style, length, texture]
- Eyes: [color, shape, unique features]
- Face: [shape, complexion, notable features]
- Build: [height, body type, physique]

Attire:
- Style: [casual/formal/fantasy/futuristic]
- Main Clothing: [specific items, colors]
- Accessories: [jewelry, gadgets, props]

Pose & Expression:
- Pose: [standing/sitting/action/dynamic]
- Expression: [emotion, intensity]
- Gesture: [hand position, body language]

Additional Details:
- Lighting: [natural/studio/dramatic]
- Background: [simple/detailed/environment]
- Special Effects: [glow/particles/aura]

Quality & Style Tags:
(masterpiece, best quality, highly detailed, ultra realistic, photorealistic, 8k, sharp focus)
Style Options: [cartoon/anime/photorealistic/hyperrealistic]
Additional Tags: [cinematic, dramatic lighting, professional photography]""",
            
            "Style": """Art Style Description:
Base Style:
- Type: [cartoon/anime/photorealistic/hyperrealistic]
- Medium: [digital/traditional/mixed media]
- Rendering: [2D/3D/hybrid]

Visual Quality:
- Resolution: [8k/4k/high resolution]
- Detail Level: [highly detailed/ultra detailed/intricate]
- Sharpness: [sharp focus/crystal clear]

Lighting & Atmosphere:
- Main Light: [natural/studio/dramatic/cinematic]
- Time of Day: [golden hour/blue hour/midday/night]
- Mood: [bright/moody/atmospheric]

Color & Technique:
- Color Palette: [warm/cool/vibrant/muted/pastel]
- Color Grading: [HDR/film-like/stylized]
- Technique: [brush strokes/lineart/painterly/photographic]

Professional Elements:
- Camera: [professional photography/cinematic/portrait lens]
- Post-processing: [color grading/professional retouching]
- Composition: [rule of thirds/dynamic/balanced]

Quality Tags:
(masterpiece, best quality, highly detailed)
(professional lighting, perfect composition)
(ultra realistic, photorealistic, hyperrealistic)
Style-specific Tags: [add based on chosen style]""",
            
            "Scene": """Scene Description:
Location: [indoor/outdoor setting]
Time: [day/night/sunset]
Weather: [clear/rainy/snowy]
Lighting: [natural/artificial/dramatic]
Atmosphere: [peaceful/tense/magical]
Background Elements: [furniture/nature/buildings]
Additional Details: [mood, special effects]""",
            
            "Object": """Object Description:
Type: [item category]
Material: [metal/wood/fabric/etc]
Size: [small/medium/large]
Color: [main colors]
Texture: [smooth/rough/patterned]
Condition: [new/worn/antique]
Special Features: [unique characteristics]""",
            
            "Custom": """[Your custom template structure]
Element 1: [description]
Element 2: [description]
Element 3: [description]
...
Additional Notes: [any special instructions]"""
        }
        
        return templates.get(template_type, "Template not found")

    def setup_auto_save_traces(self):
        """Setup traces on all variables to auto-save settings when they change"""
        # List of all StringVar variables
        string_vars = [
            'name_var', 'trigger_word_var', 'data_folder_var', 'output_folder_var',
            'lr_var', 'batch_size_var', 'steps_var', 'grad_accum_var',
            'noise_scheduler_var', 'optimizer_var', 'save_dtype_var',
            'save_every_var', 'max_saves_var', 'network_type_var',
            'linear_var', 'linear_alpha_var', 'sample_every_var',
            'sampling_steps_var', 'cfg_scale_var', 'seed_var',
            'model_path_var'
        ]
        
        # List of all BooleanVar variables
        bool_vars = [
            'is_flux_var', 'quantize_var', 'convert_utf8_var', 'auto_steps_var',
            'train_unet_var', 'train_text_encoder_var', 'grad_checkpointing_var',
            'enable_sampling_var', 'use_fixed_seed_var'
        ]
        
        # Add trace to each StringVar
        for var_name in string_vars:
            if hasattr(self, var_name):
                getattr(self, var_name).trace_add("write", self._on_setting_change)
        
        # Add trace to each BooleanVar
        for var_name in bool_vars:
            if hasattr(self, var_name):
                getattr(self, var_name).trace_add("write", self._on_setting_change)
        
        # Add trace for prompt text changes
        self.prompt_text.bind('<<Modified>>', self._on_prompt_change)

    def _on_setting_change(self, *args):
        """Called when any setting changes to save current state"""
        self.save_last_settings()
        self.save_current_train()  # Also save current training config
        
    def _on_prompt_change(self, event):
        """Called when prompt text changes"""
        if self.prompt_text.edit_modified():
            self.save_last_settings()
            self.prompt_text.edit_modified(False)

    def update_steps_on_lr_change(self):
        """Update steps when learning rate changes and auto steps is enabled"""
        if self.auto_steps_var.get():
            folder_path = self.data_folder_var.get()
            if folder_path and os.path.exists(folder_path):
                image_count, _ = self.count_dataset_files(folder_path)
                suggested_steps = self.calculate_suggested_steps(image_count)
                self.steps_var.set(str(suggested_steps))

    def save_current_train(self):
        """Save current training configuration"""
        try:
            config = self.prepare_config()
            with open(self.current_train_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            self.logger.error(f"Failed to save current training config: {str(e)}")

    def load_current_train(self):
        """Load current training configuration if it exists"""
        try:
            if os.path.exists(self.current_train_file):
                with open(self.current_train_file, 'r') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load current training config: {str(e)}")
        return None

    def add_to_batch(self):
        """Add current configuration to batch queue"""
        try:
            config = self.prepare_config()
            name = config['config']['name']
            
            # Add to batch list
            self.batch_configs.append(config)
            self.batch_listbox.insert(tk.END, f"Training: {name}")
            
            self.batch_status_var.set(f"Added {name} to batch queue")
            self.logger.info(f"Added configuration '{name}' to batch queue")
            
            # Update button if we're in batch tab
            if self.current_tab == "Batch":
                self.train_button.configure(
                    text="Start Batch Processing",
                    command=self.start_batch_processing
                )
            
        except Exception as e:
            error_msg = f"Failed to add to batch: {str(e)}"
            self.logger.error(error_msg)
            messagebox.showerror("Error", error_msg)

    def remove_from_batch(self):
        """Remove selected configuration from batch queue"""
        try:
            selection = self.batch_listbox.curselection()
            if selection:
                index = selection[0]
                name = self.batch_configs[index]['config']['name']
                
                self.batch_listbox.delete(index)
                self.batch_configs.pop(index)
                
                self.batch_status_var.set(f"Removed {name} from batch queue")
                self.logger.info(f"Removed configuration '{name}' from batch queue")
                
        except Exception as e:
            error_msg = f"Failed to remove from batch: {str(e)}"
            self.logger.error(error_msg)
            messagebox.showerror("Error", error_msg)

    def clear_batch(self):
        """Clear all configurations from batch queue"""
        try:
            self.batch_configs.clear()
            self.batch_listbox.delete(0, tk.END)
            self.batch_status_var.set("Batch queue cleared")
            self.logger.info("Batch queue cleared")
            
            # Update button if we're in batch tab
            if self.current_tab == "Batch":
                self.train_button.configure(
                    text="Start Training",
                    command=self.start_training
                )
            
        except Exception as e:
            error_msg = f"Failed to clear batch: {str(e)}"
            self.logger.error(error_msg)
            messagebox.showerror("Error", error_msg)

    def start_batch_processing(self):
        """Start processing all configurations in the batch queue"""
        if not self.batch_configs:
            messagebox.showwarning("Warning", "Batch queue is empty")
            return
        
        try:
            # Load paths from config.yaml
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
            if not os.path.exists(config_path):
                raise FileNotFoundError("config.yaml not found. Please run start.bat first.")
            
            with open(config_path, 'r') as f:
                paths_config = yaml.safe_load(f)
            
            # Disable batch controls during processing
            self.disable_batch_controls()
            
            for i, config in enumerate(self.batch_configs):
                name = config['config']['name']
                self.batch_status_var.set(f"Processing {name} ({i+1}/{len(self.batch_configs)})")
                
                try:
                    # Get dataset folder path
                    if 'datasets' in config['config']['process'][0]:
                        data_folder = config['config']['process'][0]['datasets'][0]['folder_path']
                        
                        # Check if UTF-8 conversion is needed
                        convert_utf8 = False
                        for process in config['config']['process']:
                            if 'convert_utf8' in process and process['convert_utf8']:
                                convert_utf8 = True
                                break
                        
                        if convert_utf8 and data_folder and os.path.exists(data_folder):
                            self.batch_status_var.set(f"Converting captions to UTF-8 for {name}...")
                            self.logger.info(f"Starting UTF-8 conversion for {name} in folder: {data_folder}")
                            
                            try:
                                converted = self.convert_captions_to_utf8(data_folder)
                                if converted > 0:
                                    self.logger.info(f"Converted {converted} caption files to UTF-8 for {name}")
                                self.batch_status_var.set(f"UTF-8 conversion completed for {name}")
                            except Exception as e:
                                self.logger.error(f"UTF-8 conversion failed for {name}: {str(e)}")
                                raise
                
                    # Save current config to temp file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                        config_path = f.name
                    
                    # Start training process and wait for completion
                    process = subprocess.Popen([
                        paths_config['python_path'],
                        paths_config['train_script_path'],
                        config_path
                    ])
                    
                    # Wait for this job to complete before starting next one
                    process.wait()
                    
                    if process.returncode != 0:
                        raise Exception(f"Training process for {name} failed with return code {process.returncode}")
                    
                finally:
                    # Clean up temp file
                    if os.path.exists(config_path):
                        os.unlink(config_path)
            
            self.batch_status_var.set("Batch processing completed successfully")
            messagebox.showinfo("Success", "All batch jobs completed")
            
        except Exception as e:
            error_msg = f"Batch processing failed: {str(e)}"
            self.logger.error(error_msg)
            messagebox.showerror("Error", error_msg)
        finally:
            self.enable_batch_controls()

    def disable_batch_controls(self):
        """Disable batch control buttons during processing"""
        for widget in self.batch_right.winfo_children():
            if isinstance(widget, ctk.CTkButton):
                widget.configure(state="disabled")

    def enable_batch_controls(self):
        """Enable batch control buttons after processing"""
        for widget in self.batch_right.winfo_children():
            if isinstance(widget, ctk.CTkButton):
                widget.configure(state="normal")

    def browse_caption_folder(self):
        """Browse for folder containing images to caption"""
        folder = filedialog.askdirectory()
        if folder:
            self.caption_folder_var.set(folder)

    def load_caption_template(self):
        """Load a caption template from file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.caption_template.delete('1.0', tk.END)
                    self.caption_template.insert('1.0', f.read())
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load template: {str(e)}")

    def save_caption_template(self):
        """Save current caption template to file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.caption_template.get('1.0', tk.END))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save template: {str(e)}")

    def generate_captions(self):
        """Generate caption files for all images in the selected folder"""
        folder = self.caption_folder_var.get()
        if not folder or not os.path.exists(folder):
            messagebox.showerror("Error", "Please select a valid folder")
            return
        
        if self.use_llm_vision_var.get():
            if not self.llm_api_key_var.get():
                messagebox.showerror("Error", "Please enter an API key for LLM Vision")
                return
        else:
            template = self.caption_template.get('1.0', tk.END).strip()
            if not template:
                messagebox.showerror("Error", "Caption template is empty")
                return
        
        try:
            # Get all image files
            image_files = [f for f in os.listdir(folder) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                messagebox.showwarning("Warning", "No image files found in folder")
                return
            
            # Process each image
            created = 0
            skipped = 0
            for img_file in image_files:
                caption_file = os.path.splitext(img_file)[0] + '.txt'
                caption_path = os.path.join(folder, caption_file)
                
                # Skip if file exists and overwrite is false
                if os.path.exists(caption_path) and not self.overwrite_captions_var.get():
                    skipped += 1
                    continue
                
                if self.use_llm_vision_var.get():
                    # Generate caption using LLM Vision
                    caption = self.generate_llm_caption(os.path.join(folder, img_file))
                else:
                    # Use template-based caption
                    caption = self.caption_template.get('1.0', tk.END).strip()
                
                # Add prefix/suffix
                if self.caption_prefix_var.get():
                    caption = self.caption_prefix_var.get() + ", " + caption
                if self.caption_suffix_var.get():
                    caption = caption + ", " + self.caption_suffix_var.get()
                
                # Save caption file
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(caption)
                created += 1
            
            status = f"Created {created} caption files"
            if skipped > 0:
                status += f", skipped {skipped} existing files"
            messagebox.showinfo("Success", status)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate captions: {str(e)}")

    def generate_llm_caption(self, image_path):
        """Generate caption for an image using LLM Vision API"""
        try:
            import base64
            import requests
            import json
            
            # Read and encode image
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            if self.use_local_llm_var.get():
                # Use local LLM API (LMStudio compatible endpoint)
                headers = {
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self.llm_prompt_var.get()
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{encoded_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": float(self.local_model_temp_var.get()),
                    "max_tokens": 500
                }
                
                response = requests.post(
                    f"{self.local_api_url_var.get()}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                result = response.json()
                caption = result['choices'][0]['message']['content']
                
            else:
                # Existing cloud API code...
                if "gpt-4" in self.llm_model_var.get():
                    # OpenAI GPT-4 Vision API code...
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.llm_api_key_var.get()}"
                    }
                    
                    payload = {
                        "model": self.llm_model_var.get(),
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": self.llm_prompt_var.get()
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{encoded_image}"
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 500
                    }
                    
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=payload
                    )
                    
                    result = response.json()
                    caption = result['choices'][0]['message']['content']
                
                elif "claude" in self.llm_model_var.get():
                    # Anthropic Claude API code...
                    headers = {
                        "Content-Type": "application/json",
                        "x-api-key": self.llm_api_key_var.get(),
                        "anthropic-version": "2023-06-01"
                    }
                    
                    payload = {
                        "model": self.llm_model_var.get(),
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": self.llm_prompt_var.get()
                                    },
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": encoded_image
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 500
                    }
                    
                    response = requests.post(
                        "https://api.anthropic.com/v1/messages",
                        headers=headers,
                        json=payload
                    )
                    
                    result = response.json()
                    caption = result['content'][0]['text']
                
            return caption.strip()
            
        except Exception as e:
            self.logger.error(f"LLM Vision API error: {str(e)}")
            raise Exception(f"Failed to generate caption using LLM Vision: {str(e)}")

    def toggle_llm_type(self):
        """Toggle between cloud and local LLM options"""
        if self.use_llm_vision_var.get():
            cloud_state = "disabled" if self.use_local_llm_var.get() else "normal"
            local_state = "normal" if self.use_local_llm_var.get() else "disabled"
            
            # Update cloud options
            for widget in self.llm_options.winfo_children():
                if widget != self.local_api_frame:
                    for child in widget.winfo_children():
                        if isinstance(child, (ctk.CTkEntry, ctk.CTkOptionMenu)):
                            child.configure(state=cloud_state)
            
            # Update local options
            for widget in self.local_api_frame.winfo_children():
                for child in widget.winfo_children():
                    if isinstance(child, ctk.CTkEntry):
                        child.configure(state=local_state)
        else:
            # Disable all if LLM vision is off
            for widget in self.llm_options.winfo_children():
                for child in widget.winfo_children():
                    if isinstance(child, (ctk.CTkEntry, ctk.CTkOptionMenu)):
                        child.configure(state="disabled")

    def load_config_to_batch(self):
        """Load a saved configuration file into the batch queue"""
        try:
            # Open file dialog
            file_path = filedialog.askopenfilename(
                title="Select Configuration File",
                filetypes=[
                    ("YAML files", "*.yaml *.yml"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                # Load the configuration
                with open(file_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Validate basic structure
                if not isinstance(config, dict) or 'config' not in config:
                    raise ValueError("Invalid configuration file format")
                
                # Get name from config
                name = config['config'].get('name', os.path.basename(file_path))
                
                # Add to batch list
                self.batch_configs.append(config)
                self.batch_listbox.insert(tk.END, f"Training: {name}")
                
                self.batch_status_var.set(f"Added {name} to batch queue")
                self.logger.info(f"Loaded configuration '{name}' into batch queue from {file_path}")
                
        except Exception as e:
            error_msg = f"Failed to load configuration: {str(e)}"
            self.logger.error(error_msg)
            messagebox.showerror("Error", error_msg)

    def move_batch_item_up(self):
        """Move selected batch item up in the queue"""
        try:
            selection = self.batch_listbox.curselection()
            if not selection:
                return
            
            index = selection[0]
            if index == 0:  # Already at top
                return
            
            # Swap items in configs list
            self.batch_configs[index], self.batch_configs[index-1] = \
                self.batch_configs[index-1], self.batch_configs[index]
            
            # Get item text
            item_text = self.batch_listbox.get(index)
            
            # Delete and reinsert in listbox
            self.batch_listbox.delete(index)
            self.batch_listbox.insert(index-1, item_text)
            
            # Update selection
            self.batch_listbox.selection_clear(0, tk.END)
            self.batch_listbox.selection_set(index-1)
            self.batch_listbox.see(index-1)
            
            self.batch_status_var.set(f"Moved {item_text} up")
            
        except Exception as e:
            self.logger.error(f"Failed to move item up: {str(e)}")
            messagebox.showerror("Error", "Failed to move item")

    def move_batch_item_down(self):
        """Move selected batch item down in the queue"""
        try:
            selection = self.batch_listbox.curselection()
            if not selection:
                return
            
            index = selection[0]
            if index >= len(self.batch_configs) - 1:  # Already at bottom
                return
            
            # Swap items in configs list
            self.batch_configs[index], self.batch_configs[index+1] = \
                self.batch_configs[index+1], self.batch_configs[index]
            
            # Get item text
            item_text = self.batch_listbox.get(index)
            
            # Delete and reinsert in listbox
            self.batch_listbox.delete(index)
            self.batch_listbox.insert(index+1, item_text)
            
            # Update selection
            self.batch_listbox.selection_clear(0, tk.END)
            self.batch_listbox.selection_set(index+1)
            self.batch_listbox.see(index+1)
            
            self.batch_status_var.set(f"Moved {item_text} down")
            
        except Exception as e:
            self.logger.error(f"Failed to move item down: {str(e)}")
            messagebox.showerror("Error", "Failed to move item")

    def handle_train_button(self):
        """Handle train button click based on current tab"""
        if self.current_tab == "Batch" and self.batch_configs:
            self.start_batch_processing()
        else:
            self.start_training()

    def on_tab_change(self):
        """Handle tab changes and update button text/function"""
        try:
            # Get current tab name from CTkTabview
            tab_name = self.main_notebook.get()
            self.current_tab = tab_name
            
            if tab_name == "Batch" and self.batch_configs:
                # In Batch tab with configs queued
                self.train_button.configure(
                    text="Start Batch Processing",
                    command=self.start_batch_processing
                )
            else:
                # In any other tab or no batch configs
                self.train_button.configure(
                    text="Start Training",
                    command=self.start_training
                )
            
            self.logger.debug(f"Changed to {tab_name} tab, updated training button")
            
        except Exception as e:
            self.logger.error(f"Error updating button on tab change: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Flux Training GUI")
    
    # Configure root window
    root.minsize(800, 600)
    root.overrideredirect(True)
    root.configure(bg='#1e1e1e')
    
    # Make window draggable
    def start_move(event):
        root.x = event.x_root
        root.y = event.y_root

    def stop_move(event):
        root.x = None
        root.y = None

    def do_move(event):
        if hasattr(root, 'x') and hasattr(root, 'y'):
            deltax = event.x_root - root.x
            deltay = event.y_root - root.y
            x = root.winfo_x() + deltax
            y = root.winfo_y() + deltay
            root.geometry(f"+{x}+{y}")
            root.x = event.x_root
            root.y = event.y_root
    
    # Create title bar with theme-aware colors
    title_bar = ctk.CTkFrame(root, height=30)
    title_bar.pack(fill='x', side='top')
    title_bar.configure(fg_color='#1e1e1e')  # Will be updated by theme toggle
    
    # Add title with theme-aware colors
    title_label = ctk.CTkLabel(
        title_bar, 
        text="Flux Training GUI",
        text_color="white"  # Will be updated by theme toggle
    )
    title_label.pack(side='left', padx=10)
    
    # Add close button with theme-aware colors
    close_button = ctk.CTkButton(
        title_bar,
        text="×",
        width=30,
        height=30,
        command=root.destroy,
        fg_color="transparent",
        hover_color="#FF4444",
        text_color="white"  # Will be updated by theme toggle
    )
    close_button.pack(side='right', padx=5)
    
    # Bind dragging to title bar
    title_bar.bind("<ButtonPress-1>", start_move)
    title_bar.bind("<ButtonRelease-1>", stop_move)
    title_bar.bind("<B1-Motion>", do_move)
    
    # Center the window but let it size to content
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - 500) // 2
    y = (screen_height - 512) // 2
    root.geometry(f"+{x}+{y}")
    
    app = FluxTrainingGUI(root)
    
    # Update window size to fit content
    root.update_idletasks()
    root.geometry('')
    
    root.mainloop()