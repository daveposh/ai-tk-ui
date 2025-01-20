import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import yaml
import os
import chardet
import threading
import subprocess
import sys
import json
import tempfile
from themes import Themes
import customtkinter as ctk

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
        super().__init__(master)
        self.master = master
        self._raw_config = config if config is not None else {}
        self.current_config_path = None
        self.current_config_label_var = tk.StringVar(value='No config loaded')
        self.batch_configs = []
        
        # Initialize theme first
        self.themes = Themes()
        self.is_dark_mode = self.load_theme_preference()
        
        # Settings file path
        self.settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_settings.json")
        
        # Initialize variables
        self.initialize_variables()
        
        # Add seed variable
        self.seed_var = tk.StringVar(value="-1")
        self.use_fixed_seed_var = tk.BooleanVar(value=False)
        
        # Configure window properties
        self.master.minsize(800, 600)  # Adjusted minimum size
        
        # Create main layout
        self.create_widgets()
        
        # Pack main frame
        self.pack(fill="both", expand=True)
        self.grid_columnconfigure(0, weight=1)
        
        # Load settings after widgets are created
        self.load_last_settings()
        
        if config:
            self.load_config(config)
        
        # Save settings when window is closed
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Add trace to data_folder_var
        self.data_folder_var.trace_add("write", lambda *args: self.update_dataset_stats(self.data_folder_var.get()))
        
        # Now apply theme after all widgets are created
        self.themes.apply_theme(self.master, self.is_dark_mode)
        self.add_theme_toggle()

    def initialize_variables(self):
        """Initialize all variables used in the GUI"""
        # Initialize learning rate presets
        self.lr_presets = {
            '1e-4 (Standard)': {'value': '1e-4', 'description': 'Standard learning rate for most cases'},
            '2e-4 (Faster)': {'value': '2e-4', 'description': 'Faster learning, may be less stable'},
            '5e-5 (Conservative)': {'value': '5e-5', 'description': 'More conservative, more stable'},
            '3e-5 (Very Conservative)': {'value': '3e-5', 'description': 'Very conservative, very stable'},
            '2e-5 (Minimal)': {'value': '2e-5', 'description': 'Minimal changes, highest stability'}
        }
        
        # Initialize containers
        self.step_buttons = []
        self.resolution_vars = {}
        
        # Initialize all variables with defaults
        self.name_var = tk.StringVar(value='')
        self.trigger_word_var = tk.StringVar(value='')
        self.data_folder_var = tk.StringVar(value='')
        self.output_folder_var = tk.StringVar(value='')
        self.lr_var = tk.StringVar(value='1e-4')
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
        
        # Training tab
        training_tab = self.main_notebook.add("Training")
        
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
            'text_color': self.themes.get_theme(self.is_dark_mode)['text'],
            'border_width': 0,
            'corner_radius': 6
        }
        
        ctk.CTkButton(model_path_frame, text="Browse", command=lambda: self.browse_folder('model'), **button_style).pack(side='right')
        
        # Model Options
        model_options_frame = ctk.CTkFrame(model_frame)
        model_options_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkCheckBox(model_options_frame, text="Is Flux", variable=self.is_flux_var).pack(side='left', padx=5)
        ctk.CTkCheckBox(model_options_frame, text="Quantize", variable=self.quantize_var).pack(side='left', padx=5)
        
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
        ctk.CTkOptionMenu(lr_frame, 
                         variable=self.lr_var, 
                         values=list(self.lr_presets.keys()),
                         **option_menu_style).pack(side='left', padx=5)
        
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
        
        ctk.CTkButton(button_frame, text="Load Config", command=self.load_config_file, **button_style).pack(side='left', padx=5)
        ctk.CTkButton(button_frame, text="Save Config", command=self.save_config, **button_style).pack(side='left', padx=5)
        ctk.CTkButton(button_frame, text="Start Training", command=self.start_training, **button_style).pack(side='right', padx=5)
        
        # Progress Frame
        progress_frame = ctk.CTkFrame(self)
        progress_frame.pack(fill='x', padx=10, pady=5)
        
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(fill='x', padx=5, pady=5)
        self.progress_bar.set(0)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ctk.CTkLabel(progress_frame, textvariable=self.status_var)
        self.status_label.pack(anchor='w', padx=5)
        
        # Sample tab
        sample_tab = self.main_notebook.add("Sample")
        
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
        
        ctk.CTkLabel(prompt_frame, text="Prompt:").pack(anchor='w', padx=5, pady=2)
        
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

    def prepare_config(self):
        """Prepare the configuration dictionary"""
        # Get the actual learning rate value from the preset
        selected_lr = self.lr_var.get()
        lr_value = self.lr_presets[selected_lr]['value'] if selected_lr in self.lr_presets else selected_lr
        
        # Get selected resolutions
        selected_resolutions = [res for res, data in self.resolution_vars.items() if data['var'].get()]
        
        # Prepare sample config
        sample_config = None
        if self.enable_sampling_var.get():
            # Split prompts by blank lines and remove empty strings
            prompts = [p.strip() for p in self.prompt_text.get('1.0', tk.END).split('\n\n')]
            prompts = [p for p in prompts if p]  # Remove empty prompts
            
            sample_config = {
                'sampler': 'flowmatch',  # Must match train.noise_scheduler
                'sample_every': int(self.sample_every_var.get()),
                'width': 1024,  # Default size
                'height': 1024,  # Default size
                'prompts': prompts,
                'neg': '',  # Not used for flux
                'seed': int(self.seed_var.get()),
                'walk_seed': self.use_fixed_seed_var.get(),
                'guidance_scale': float(self.cfg_scale_var.get()),
                'sample_steps': int(self.sampling_steps_var.get())
            }
        
        return {
            'job': 'extension',
            'config': {
                'name': self.name_var.get(),
                'trigger_word': self.trigger_word_var.get(),
                'process': [{
                    'type': 'sd_trainer',
                    'training_folder': self.output_folder_var.get(),
                    'device': 'cuda:0',
                    'network': {
                        'type': self.network_type_var.get(),
                        'linear': int(self.linear_var.get()),
                        'linear_alpha': int(self.linear_alpha_var.get())
                    },
                    'save': {
                        'dtype': self.save_dtype_var.get(),
                        'save_every': int(self.save_every_var.get()),
                        'max_step_saves_to_keep': int(self.max_saves_var.get()),
                        'push_to_hub': False
                    },
                    'datasets': [{
                        'folder_path': self.data_folder_var.get(),
                        'caption_ext': 'txt',
                        'caption_dropout_rate': 0.05,
                        'shuffle_tokens': False,
                        'cache_latents_to_disk': True,
                        'resolution': [int(res.split('x')[0]) for res, data in self.resolution_vars.items() if data['var'].get()]
                    }],
                    'train': {
                        'batch_size': int(self.batch_size_var.get()),
                        'steps': int(self.steps_var.get()),
                        'gradient_accumulation_steps': int(self.grad_accum_var.get()),
                        'train_unet': self.train_unet_var.get(),
                        'train_text_encoder': self.train_text_encoder_var.get(),
                        'gradient_checkpointing': self.grad_checkpointing_var.get(),
                        'noise_scheduler': self.noise_scheduler_var.get(),
                        'optimizer': self.optimizer_var.get(),
                        'lr': lr_value,
                        'ema_config': {
                            'use_ema': True,
                            'ema_decay': 0.99
                        },
                        'dtype': 'bf16'
                    },
                    'model': {
                        'name_or_path': self.model_path_var.get(),
                        'is_flux': self.is_flux_var.get(),
                        'quantize': self.quantize_var.get()
                    },
                    'sample': sample_config
                }]
            },
            'meta': {
                'name': '[name]',
                'version': '1.0'
            }
        }

    def save_config(self):
        """Save the current configuration to a YAML file"""
        config = self.prepare_config()
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                self.current_config_path = file_path
                self.current_config_label_var.set(f"Loaded: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", "Configuration saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")

    def start_training(self):
        """Start the training process with progress updates"""
        try:
            config = self.prepare_config()
            total_steps = int(self.steps_var.get())
            
            # Create output folder if it doesn't exist
            os.makedirs(self.output_folder_var.get(), exist_ok=True)
            
            # Save or update config file
            if self.current_config_path:
                config_path = self.current_config_path
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            else:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_config:
                    yaml.dump(config, temp_config, default_flow_style=False, sort_keys=False)
                    config_path = temp_config.name
            
            # Get the path to run.py
            script_dir = os.path.dirname(os.path.abspath(__file__))
            train_script = os.path.join(script_dir, "run.py")
            
            # Start training process with pipe for output
            process = subprocess.Popen([sys.executable, train_script, config_path],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True,
                                     bufsize=1)
            
            def monitor_progress():
                """Monitor training progress from process output"""
                current_step = 0
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    
                    # Parse progress from output
                    if "Step:" in line:
                        try:
                            current_step = int(line.split("Step:")[1].split("/")[0])
                            self.update_progress(current_step, total_steps)
                        except:
                            pass
                    
                    # Update status with any errors
                    if process.poll() is not None:
                        error = process.stderr.read()
                        if error:
                            self.status_var.set(f"Error: {error.strip()}")
                        break
                
                # Training completed
                if process.returncode == 0:
                    self.update_progress(total_steps, total_steps, "Training completed")
                else:
                    self.status_var.set("Training failed")
            
            # Start progress monitoring in a separate thread
            progress_thread = threading.Thread(target=monitor_progress)
            progress_thread.daemon = True
            progress_thread.start()
            
            messagebox.showinfo("Success", f"Training started with config: {os.path.basename(config_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {str(e)}")

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
            'sampling_steps': self.sampling_steps_var.get(),
            'cfg_scale': self.cfg_scale_var.get(),
            'prompt_style': self.prompt_style_var.get(),
            'prompt_text': self.prompt_text.get('1.0', tk.END).strip(),
            'resolutions': {size: data['var'].get() for size, data in self.resolution_vars.items()},
            'batch_configs': self.batch_configs,  # Save batch configs
            'dark_mode': self.is_dark_mode
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
                
            # Load all the settings
            self.name_var.set(settings.get('name', ''))
            self.trigger_word_var.set(settings.get('trigger_word', ''))
            self.data_folder_var.set(settings.get('data_folder', ''))
            self.output_folder_var.set(settings.get('output_folder', ''))
            self.model_path_var.set(settings.get('model_path', ''))
            self.is_flux_var.set(settings.get('is_flux', True))
            self.quantize_var.set(settings.get('quantize', True))
            self.convert_utf8_var.set(settings.get('convert_utf8', True))
            self.lr_var.set(settings.get('learning_rate', '1e-4'))
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
            self.sampling_steps_var.set(settings.get('sampling_steps', '20'))
            self.cfg_scale_var.set(settings.get('cfg_scale', '3.5'))
            self.prompt_style_var.set(settings.get('prompt_style', 'Natural'))
            
            # Load prompt text
            if 'prompt_text' in settings:
                self.prompt_text.delete('1.0', tk.END)
                self.prompt_text.insert('1.0', settings['prompt_text'])
            
            # Load resolutions
            if 'resolutions' in settings:
                for size, value in settings['resolutions'].items():
                    if size in self.resolution_vars:
                        self.resolution_vars[size]['var'].set(value)
                        
            # Update steps entry state
            if hasattr(self, 'steps_entry'):
                self.steps_entry.configure(state='disabled' if self.auto_steps_var.get() else 'normal')
            
            # Update dataset stats if data folder exists
            self.update_dataset_stats(settings.get('data_folder', ''))
            
            # Load batch configs
            self.batch_configs = settings.get('batch_configs', [])
            for config_path in self.batch_configs:
                if os.path.exists(config_path):
                    self.batch_tree.insert('', 'end', values=(os.path.basename(config_path), config_path))
                
            # Load theme preference
            self.is_dark_mode = settings.get('dark_mode', True)
            
        except Exception as e:
            print(f"Failed to load settings: {str(e)}")

    def on_closing(self):
        """Handle window closing"""
        self.save_last_settings()
        self.master.destroy()

    def load_config_file(self):
        """Load configuration from a YAML file"""
        file_path = filedialog.askopenfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self.load_config(config)
                self.current_config_path = file_path
                self.current_config_label_var.set(f"Loaded: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", "Configuration loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")

    def update_config(self):
        """Update the currently loaded config file"""
        if not self.current_config_path:
            messagebox.showwarning("Warning", "No configuration file loaded. Please load a config file first.")
            return
            
        try:
            config = self.prepare_config()
            
            with open(self.current_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            self.current_config_label_var.set(f"Updated: {os.path.basename(self.current_config_path)}")
            messagebox.showinfo("Success", "Configuration updated successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update configuration: {str(e)}")

    def update_dataset_stats(self, folder_path):
        """Update dataset statistics for a given folder path"""
        if folder_path and os.path.exists(folder_path):
            image_count, caption_count = self.count_dataset_files(folder_path)
            self.dataset_stats_var.set(f"Images: {image_count}, Captions: {caption_count}")
            if self.auto_steps_var.get():
                self.update_suggested_steps(image_count)

    def add_to_batch(self):
        """Add current configuration to batch"""
        if not self.current_config_path:
            messagebox.showwarning("Warning", "Please save or load a configuration first.")
            return
            
        # Check if config is already in batch
        for item in self.batch_tree.get_children():
            if self.batch_tree.item(item)['values'][1] == self.current_config_path:
                messagebox.showinfo("Info", "This configuration is already in the batch.")
                return
        
        # Add to treeview
        self.batch_tree.insert('', 'end', values=(os.path.basename(self.current_config_path), 
                                                 self.current_config_path))
        self.batch_configs.append(self.current_config_path)

    def add_config_to_batch(self):
        """Add configuration file to batch"""
        file_paths = filedialog.askopenfilenames(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        
        for file_path in file_paths:
            # Check if config is already in batch
            if any(file_path == self.batch_tree.item(item)['values'][1] 
                  for item in self.batch_tree.get_children()):
                continue
                
            # Add to treeview
            self.batch_tree.insert('', 'end', values=(os.path.basename(file_path), file_path))
            self.batch_configs.append(file_path)

    def remove_from_batch(self):
        """Remove selected configuration from batch"""
        selected = self.batch_tree.selection()
        if not selected:
            return
            
        for item in selected:
            config_path = self.batch_tree.item(item)['values'][1]
            self.batch_configs.remove(config_path)
            self.batch_tree.delete(item)

    def clear_batch(self):
        """Clear all configurations from batch"""
        self.batch_tree.delete(*self.batch_tree.get_children())
        self.batch_configs.clear()

    def start_batch_training(self):
        """Start batch training process"""
        if not self.batch_configs:
            messagebox.showwarning("Warning", "Batch queue is empty. Please add configurations first.")
            return
            
        try:
            # Get the path to run.py in the same directory as the GUI script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            train_script = os.path.join(script_dir, "run.py")
            
            def run_batch():
                for config_path in self.batch_configs:
                    try:
                        # Start training process
                        process = subprocess.Popen([sys.executable, train_script, config_path])
                        process.wait()  # Wait for the current training to complete
                    except Exception as e:
                        print(f"Error running config {config_path}: {str(e)}")
                
                messagebox.showinfo("Success", "Batch training completed!")
            
            # Start batch training in a separate thread
            batch_thread = threading.Thread(target=run_batch)
            batch_thread.daemon = True
            batch_thread.start()
            
            messagebox.showinfo("Success", "Batch training started!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start batch training: {str(e)}")

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
        self.themes.apply_theme(self.master, self.is_dark_mode)
        
        # Update frame colors
        frame_color = self.themes.get_theme(self.is_dark_mode)['frame']
        for widget in self.scrollable_frame.winfo_children():
            if isinstance(widget, ctk.CTkFrame):
                widget.configure(fg_color=frame_color)
                # Update nested frames
                for child in widget.winfo_children():
                    if isinstance(child, ctk.CTkFrame):
                        child.configure(fg_color=frame_color)
        
        self.theme_button.configure(text="🌙" if self.is_dark_mode else "☀️")
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
        """Calculate suggested training steps based on image count"""
        if image_count <= 0:
            return 5000  # Default value
        
        # Get current learning rate value
        selected_lr = self.lr_var.get()
        lr_value = float(self.lr_presets[selected_lr]['value'] if selected_lr in self.lr_presets else selected_lr)
        
        # Base calculation: roughly 100 steps per image
        suggested = image_count * 100
        
        # Adjust steps based on learning rate
        # Use 1e-4 as the baseline learning rate
        lr_multiplier = 1e-4 / lr_value
        suggested = int(suggested * lr_multiplier)
        
        # Round to nearest 500
        suggested = round(suggested / 500) * 500
        
        # Set minimum and maximum values
        suggested = max(1000, min(suggested, 10000))
        
        return suggested

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

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Flux Training GUI")
    
    # Make window borderless
    root.overrideredirect(True)
    
    # Add a close button in the top-right corner
    close_button = ctk.CTkButton(
        root,
        text="×",
        width=30,
        height=30,
        command=root.destroy,
        fg_color="transparent",
        hover_color="#FF4444"
    )
    close_button.pack(anchor='ne', padx=5, pady=5)
    
    # Make window draggable
    def start_move(event):
        root.x = event.x
        root.y = event.y

    def stop_move(event):
        root.x = None
        root.y = None

    def do_move(event):
        deltax = event.x - root.x
        deltay = event.y - root.y
        x = root.winfo_x() + deltax
        y = root.winfo_y() + deltay
        root.geometry(f"+{x}+{y}")

    root.bind("<ButtonPress-1>", start_move)
    root.bind("<ButtonRelease-1>", stop_move)
    root.bind("<B1-Motion>", do_move)
    
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